"""
Supplychainer persistence layer (SQLite).

Replaces in-memory state: route runs (history), watched routes, alerts and webhook
subscriptions survive restarts. SQLite ships with Python, needs no server, and is safe for the
FastAPI threadpool when each call opens its own short-lived connection (WAL mode).
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "supplychainer.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    scenario TEXT,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    label TEXT,
    watched INTEGER NOT NULL DEFAULT 0,
    selected_index INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    run_id TEXT,
    scenario_id TEXT,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);

CREATE TABLE IF NOT EXISTS webhooks (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    url TEXT NOT NULL,
    secret TEXT NOT NULL,
    events TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    last_status TEXT
);

CREATE TABLE IF NOT EXISTS platform_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class Storage:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("SUPPLYCHAINER_DB", DEFAULT_DB)
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.Lock()
        # A shared connection is required for ":memory:" (tests); files use per-call connections.
        self._shared = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)
            if self._shared is None:
                c.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _conn(self):
        if self._shared is not None:
            with self._lock:
                self._shared.row_factory = sqlite3.Row
                yield self._shared
                self._shared.commit()
            return
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs
    def save_run(self, request: Dict[str, Any], response: Dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO runs (id, created_at, origin, destination, scenario, request_json, response_json)"
                      " VALUES (?,?,?,?,?,?,?)",
                      (run_id, time.time(), request.get("source"), request.get("destination"),
                       request.get("scenario"), json.dumps(request), json.dumps(response)))
        return run_id

    @staticmethod
    def _run_row(r, full=False) -> Dict[str, Any]:
        resp = json.loads(r["response_json"])
        recs = resp.get("recommendations", [])
        out = {
            "id": r["id"], "created_at": r["created_at"], "origin": r["origin"], "destination": r["destination"],
            "scenario": r["scenario"], "label": r["label"], "watched": bool(r["watched"]),
            "selected_index": r["selected_index"],
            "summary": [{"personas": x.get("personas", [x.get("persona")]), "eta_p85": x.get("adjusted_eta"),
                         "eta_band": x.get("eta_band"), "cost": x.get("total_cost"),
                         "threat": x.get("threat_level"), "mode": x.get("primary_mode")} for x in recs],
            "request": json.loads(r["request_json"]),
        }
        if full:
            out["response"] = resp
        return out

    def list_runs(self, limit: int = 50, watched_only: bool = False) -> List[Dict[str, Any]]:
        q = "SELECT * FROM runs" + (" WHERE watched=1" if watched_only else "") + " ORDER BY created_at DESC LIMIT ?"
        with self._conn() as c:
            return [self._run_row(r) for r in c.execute(q, (limit,)).fetchall()]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._run_row(r, full=True) if r else None

    def update_run(self, run_id: str, watched: Optional[bool] = None, label: Optional[str] = None,
                   selected_index: Optional[int] = None) -> bool:
        sets, args = [], []
        if watched is not None: sets.append("watched=?"); args.append(int(watched))
        if label is not None: sets.append("label=?"); args.append(label)
        if selected_index is not None: sets.append("selected_index=?"); args.append(int(selected_index))
        if not sets: return False
        with self._conn() as c:
            cur = c.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", (*args, run_id))
            return cur.rowcount > 0

    def delete_run(self, run_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM runs WHERE id=?", (run_id,)).rowcount > 0

    # ------------------------------------------------------------------ alerts
    def add_alert(self, severity: str, title: str, message: str, run_id: str = None,
                  scenario_id: str = None, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        now = time.time()
        with self._conn() as c:
            cur = c.execute("INSERT INTO alerts (created_at, run_id, scenario_id, severity, title, message, payload_json)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (now, run_id, scenario_id, severity, title, message, json.dumps(payload or {})))
            alert_id = cur.lastrowid
        return {"id": alert_id, "created_at": now, "run_id": run_id, "scenario_id": scenario_id,
                "severity": severity, "title": title, "message": message, "payload": payload or {},
                "acknowledged": False}

    @staticmethod
    def _alert_row(r) -> Dict[str, Any]:
        return {"id": r["id"], "created_at": r["created_at"], "run_id": r["run_id"], "scenario_id": r["scenario_id"],
                "severity": r["severity"], "title": r["title"], "message": r["message"],
                "payload": json.loads(r["payload_json"] or "{}"), "acknowledged": bool(r["acknowledged"])}

    def list_alerts(self, since_id: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM alerts WHERE id>? ORDER BY id DESC LIMIT ?", (since_id, limit)).fetchall()
        return [self._alert_row(r) for r in rows]

    def latest_alert_id(self) -> int:
        with self._conn() as c:
            r = c.execute("SELECT MAX(id) AS m FROM alerts").fetchone()
        return r["m"] or 0

    def ack_alert(self, alert_id: int) -> bool:
        with self._conn() as c:
            return c.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,)).rowcount > 0

    def unacked_count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) AS n FROM alerts WHERE acknowledged=0").fetchone()["n"]

    # ------------------------------------------------------------------ webhooks
    def add_webhook(self, url: str, events: List[str]) -> Dict[str, Any]:
        hook = {"id": uuid.uuid4().hex[:10], "created_at": time.time(), "url": url,
                "secret": uuid.uuid4().hex, "events": events, "active": True, "last_status": None}
        with self._conn() as c:
            c.execute("INSERT INTO webhooks (id, created_at, url, secret, events, active) VALUES (?,?,?,?,?,1)",
                      (hook["id"], hook["created_at"], url, hook["secret"], json.dumps(events)))
        return hook

    def list_webhooks(self, event: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM webhooks WHERE active=1 ORDER BY created_at").fetchall()
        hooks = [{"id": r["id"], "created_at": r["created_at"], "url": r["url"], "secret": r["secret"],
                  "events": json.loads(r["events"]), "active": bool(r["active"]), "last_status": r["last_status"]}
                 for r in rows]
        return [h for h in hooks if event is None or event in h["events"] or "*" in h["events"]]

    def set_webhook_status(self, hook_id: str, status: str):
        with self._conn() as c:
            c.execute("UPDATE webhooks SET last_status=? WHERE id=?", (status, hook_id))

    def delete_webhook(self, hook_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM webhooks WHERE id=?", (hook_id,)).rowcount > 0

    # ------------------------------------------------------------------ platform state
    def get_state(self, key: str) -> Optional[str]:
        with self._conn() as c:
            r = c.execute("SELECT value FROM platform_state WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    def set_state(self, key: str, value: Optional[str]):
        with self._conn() as c:
            c.execute("INSERT INTO platform_state (key, value) VALUES (?,?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
