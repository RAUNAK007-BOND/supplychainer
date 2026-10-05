"""
Username/password access for the Supplychainer app.

Accounts come from three places (first match wins):
  1. ADMIN_USERNAME / ADMIN_PASSWORD env vars - the owner's admin account. If neither is set, an
     `admin` account with a random password is created on first boot and the password is printed
     to the server log once.
  2. SUPPLYCHAINER_USERS env var - `username:<hash>` entries separated by `;` or new lines. Use this
     on hosts whose disk is wiped on restart (e.g. Render free), so accounts survive redeploys.
     Generate a line with:  python -m backend.engine.auth hash <username>
  3. The `users` table - accounts the admin creates from the Admin page.

Sessions are stateless signed cookies (HMAC-SHA256 with SESSION_SECRET). Each token carries a
fingerprint of the password hash, so resetting a password or disabling a user signs them out.
Passwords are stored as PBKDF2-SHA256 hashes only.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
from typing import Any, Dict, List, Optional

ITERATIONS = 260_000
COOKIE_NAME = "sc_session"
SESSION_DAYS = float(os.getenv("SESSION_DAYS", "14"))
LOGIN_WINDOW_S, LOGIN_MAX_FAILURES = 15 * 60, 8
USERNAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789._-")


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iters))
        return hmac.compare_digest(base64.b64encode(dk).decode(), digest)
    except (ValueError, TypeError):
        return False


def generate_password(n_words: int = 4) -> str:
    """Readable random password, e.g. 'harbor-vivid-tango-48'."""
    words = ("anchor harbor cargo route vessel atlas cobalt amber delta orbit falcon summit pilot meadow "
             "canyon ember lunar nova quartz river tango vivid willow zephyr cedar coral harvest").split()
    return "-".join(secrets.choice(words) for _ in range(n_words - 1)) + f"-{secrets.randbelow(90) + 10}"


def normalize_username(u: str) -> str:
    u = (u or "").strip().lower()
    if not (2 <= len(u) <= 40) or not set(u) <= USERNAME_CHARS:
        raise ValueError("Usernames are 2-40 characters: letters, numbers, dot, dash or underscore.")
    return u


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class AuthManager:
    def __init__(self, storage):
        self.storage = storage
        self.secret = (os.getenv("SESSION_SECRET") or self._stored_secret()).encode()
        self.env_users = self._parse_env_users(os.getenv("SUPPLYCHAINER_USERS", ""))
        self._failures: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._bootstrap_admin()

    # ------------------------------------------------------------------ setup
    def _stored_secret(self) -> str:
        s = self.storage.get_state("session_secret")
        if not s:
            s = secrets.token_hex(32)
            self.storage.set_state("session_secret", s)
        return s

    @staticmethod
    def _parse_env_users(raw: str) -> Dict[str, Dict[str, Any]]:
        users = {}
        for entry in raw.replace("\n", ";").split(";"):
            if ":" not in entry:
                continue
            name, h = entry.strip().split(":", 1)
            try:
                name = normalize_username(name)
            except ValueError:
                continue
            users[name] = {"username": name, "password_hash": h.strip(), "display_name": name, "email": None,
                           "is_admin": False, "active": True, "source": "env"}
        return users

    def _bootstrap_admin(self):
        name = os.getenv("ADMIN_USERNAME", "").strip().lower() or "admin"
        password = os.getenv("ADMIN_PASSWORD", "")
        existing = self.storage.get_user(name)
        if password:
            if not existing:
                self.storage.add_user(name, hash_password(password), display_name="Administrator", is_admin=True)
            elif not verify_password(password, existing["password_hash"]) or not existing["is_admin"]:
                # The env var is the source of truth for the owner's account.
                self.storage.update_user(name, password_hash=hash_password(password), is_admin=True, active=True)
        elif not any(u["is_admin"] for u in self.storage.list_users()):
            pw = generate_password()
            self.storage.add_user(name, hash_password(pw), display_name="Administrator", is_admin=True)
            print("=" * 72 + f"\n[AUTH] Created admin account  username: {name}  password: {pw}\n"
                  "[AUTH] Set ADMIN_USERNAME / ADMIN_PASSWORD to choose your own.\n" + "=" * 72)
        self.admin_username = name

    # ------------------------------------------------------------------ users
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        u = self.storage.get_user(username)
        if u:
            return {**u, "source": "db"}
        return self.env_users.get(username)

    def list_users(self) -> List[Dict[str, Any]]:
        db = [{**u, "source": "db"} for u in self.storage.list_users()]
        names = {u["username"] for u in db}
        return db + [u for n, u in self.env_users.items() if n not in names]

    @staticmethod
    def public_user(u: Dict[str, Any]) -> Dict[str, Any]:
        return {k: u.get(k) for k in ("username", "display_name", "email", "is_admin", "active", "created_at",
                                      "last_login", "source", "request_id")}

    # ------------------------------------------------------------------ login
    def _throttled(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if now - t < LOGIN_WINDOW_S]
            self._failures[key] = recent
            return len(recent) >= LOGIN_MAX_FAILURES

    def _fail(self, key: str):
        with self._lock:
            self._failures.setdefault(key, []).append(time.time())

    def authenticate(self, username: str, password: str, ip: str) -> Dict[str, Any]:
        """-> {"user": ...} or {"error": message, "status": http_status}"""
        try:
            username = normalize_username(username)
        except ValueError:
            username = (username or "").strip().lower()[:40]
        keys = (f"ip:{ip}", f"user:{username}")
        if any(self._throttled(k) for k in keys):
            return {"error": "Too many failed sign-in attempts. Try again in 15 minutes.", "status": 429}
        u = self.get_user(username)
        ok = verify_password(password or "", u["password_hash"] if u else hash_password("x", b"0" * 16))
        if not u or not ok or not u["active"]:
            for k in keys:
                self._fail(k)
            msg = "This account has been disabled." if u and ok and not u["active"] else "Wrong username or password."
            return {"error": msg, "status": 401}
        with self._lock:
            for k in keys:
                self._failures.pop(k, None)
        if u.get("source") == "db":
            self.storage.update_user(username, last_login=time.time())
        return {"user": u}

    # ------------------------------------------------------------------ sessions
    @staticmethod
    def _fingerprint(u: Dict[str, Any]) -> str:
        return hashlib.sha256(u["password_hash"].encode()).hexdigest()[:12]

    def issue(self, u: Dict[str, Any]) -> str:
        body = _b64(json.dumps({"u": u["username"], "exp": int(time.time() + SESSION_DAYS * 86400),
                                "fp": self._fingerprint(u)}, separators=(",", ":")).encode())
        sig = _b64(hmac.new(self.secret, body.encode(), hashlib.sha256).digest())
        return f"{body}.{sig}"

    def verify(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token or "." not in token:
            return None
        body, sig = token.rsplit(".", 1)
        expected = _b64(hmac.new(self.secret, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        try:
            data = json.loads(_unb64(body))
        except ValueError:
            return None
        if data.get("exp", 0) < time.time():
            return None
        u = self.get_user(data.get("u", ""))
        if not u or not u["active"] or data.get("fp") != self._fingerprint(u):
            return None
        return u


def _cli():
    """python -m backend.engine.auth hash <username> [password]  -> a SUPPLYCHAINER_USERS entry"""
    if len(sys.argv) >= 3 and sys.argv[1] == "hash":
        name = normalize_username(sys.argv[2])
        pw = sys.argv[3] if len(sys.argv) > 3 else generate_password()
        print(f"username: {name}\npassword: {pw}\n\nAdd to SUPPLYCHAINER_USERS (separate entries with ;):\n"
              f"{name}:{hash_password(pw)}")
    else:
        print(_cli.__doc__)


if __name__ == "__main__":
    _cli()
