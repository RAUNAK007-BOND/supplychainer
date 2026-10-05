from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.responses import Response, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import json
import mimetypes
import os
import threading
import time
from contextlib import asynccontextmanager

import httpx
from fastapi.middleware.cors import CORSMiddleware
from .engine.threat_intelligence import ThreatIntelligencePredictor, classify_threat, keyword_severity
from .engine.multimodal_network import load_canonical_hubs
from .engine.route_recommender import RouteRecommender
from .engine.scenario_manager import ScenarioManager
from .engine.supplier_scorer import SupplierScorer
from .engine.supplier_sources import SupplierDirectory, SupplierSourceError
from .engine.world_registry import WorldRegistry
from .engine.live_conditions import LiveConditions
from .engine.storage import Storage
from .engine.auth import AuthManager, COOKIE_NAME, SESSION_DAYS, generate_password, hash_password, normalize_username
from .engine.integrations import AlertEngine, WebhookDispatcher, export_csv, export_pdf, export_tms

# Global Engine State
# (The legacy US-only simulator/graph_model pipeline was constructed here on every boot but never
#  used by any live route; it is no longer loaded. See README "Two things that look like...".)
predictor = ThreatIntelligencePredictor(lazy_load=True)

# Product layer (Supplychainer Architecture) - Canonical Hubs
scenario_mgr = ScenarioManager()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() == "true"  # shared publicly without login
recommender = RouteRecommender(None, predictor, None, scenario_mgr, demo_mode=DEMO_MODE)
canonical_hubs = load_canonical_hubs()
hub_index = {h["id"]: h for h in canonical_hubs}
world = WorldRegistry(canonical_hubs)          # every real port (UN/LOCODE + WPI) and airport (OurAirports)
recommender.world = world
conditions = LiveConditions()                  # live weather at ports (Open-Meteo)
supplier_directory = SupplierDirectory()       # live suppliers (Open Supply Hub, Wikidata fallback)
supplier_scorer = SupplierScorer(supplier_directory, recommender, world, conditions)

storage = Storage()
auth = AuthManager(storage)
dispatcher = WebhookDispatcher(storage)
alert_engine = AlertEngine(storage, recommender, scenario_mgr, dispatcher)
scenario_mgr.activate_scenario(storage.get_state("live_scenario"))  # survives restarts
# Render free (and similar) wipe the disk on restart: DB-created accounts would not survive.
STORAGE_EPHEMERAL = os.getenv("STORAGE_EPHEMERAL", "").lower() == "true" or \
    os.path.abspath(storage.path).replace("\\", "/").startswith("/tmp/")


def lookup_hub(hub_id: str):
    """Core hub, or a real port/airport from the world registry."""
    return hub_index.get(hub_id) or recommender.hubs.get(hub_id) or (
        WorldRegistry.public(world.get(hub_id)) if world.get(hub_id) else None)


class RecommendRequest(BaseModel):
    source: str # This should be a Canonical Hub ID or City Name
    destination: str # This should be a Canonical Hub ID or City Name
    cargo_type: str = "general"
    priority: str = "normal"
    budget_sensitivity: str = "medium"
    transport_preference: str = "any" # sea, air, rail, road, any
    routing_policy: str = "STRICT" # STRICT or PREFERRED
    scenario: Optional[str] = None
    overrides: Optional[dict] = None
    save: bool = True
    watch: bool = False

class SourcingRequest(BaseModel):
    category: str = "Electronics"
    destination: str = "PORT-ROTTERDAM"   # where the goods must arrive: core hub or real port/airport id
    country: Optional[str] = Field(None, pattern=r"^[A-Za-z]{2}$")  # only suppliers in this ISO country
    limit: int = Field(15, ge=3, le=30)
    current_inventory: int = 1000
    safety_stock: int = 1500
    demand_forecast: int = 800
    scenario: Optional[str] = None

class LoginBody(BaseModel):
    username: str = Field(..., max_length=80)
    password: str = Field(..., max_length=200)

class AccessRequestBody(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=200)
    company: Optional[str] = Field(None, max_length=160)
    use_case: Optional[str] = Field(None, max_length=2000)
    website: Optional[str] = None  # honeypot: real people never see or fill this field

class UserCreate(BaseModel):
    username: str
    password: Optional[str] = Field(None, min_length=8, max_length=200)
    display_name: Optional[str] = Field(None, max_length=120)
    email: Optional[str] = Field(None, max_length=200)
    is_admin: bool = False
    request_id: Optional[int] = None

class UserUpdate(BaseModel):
    active: Optional[bool] = None
    is_admin: Optional[bool] = None
    display_name: Optional[str] = Field(None, max_length=120)

class RequestUpdate(BaseModel):
    status: str = Field(..., pattern=r"^(pending|declined|approved)$")

class RunUpdate(BaseModel):
    watched: Optional[bool] = None
    label: Optional[str] = None
    selected_index: Optional[int] = None

class LiveScenario(BaseModel):
    scenario_id: Optional[str] = None

class WebhookCreate(BaseModel):
    url: str = Field(..., pattern=r"^https?://")
    events: List[str] = ["alert.created", "scenario.activated", "route.planned"]

class IntelScan(BaseModel):
    hubs: List[str]
    mode: str = "sea"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Supplychainer Engine Active: Canonical Global Registry Loaded.")
    if not DEMO_MODE:
        asyncio.create_task(asyncio.to_thread(recommender.run_background_warmup))
    world.refresh_in_background_if_stale()
    yield

app = FastAPI(title="Supplychainer API", version="2.0.0", lifespan=lifespan)

# Paths anyone may call without signing in (the product page, the login form, access requests).
PUBLIC_API = {"/api/health", "/api/auth/login", "/api/auth/logout", "/api/auth/me", "/api/access-requests"}


def _client_ip(request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in PUBLIC_API and not path.startswith("/api/public/"):
        user = auth.verify(request.cookies.get(COOKIE_NAME))
        if not user:
            return JSONResponse({"detail": "Sign in to use Supplychainer."}, status_code=401)
        if path.startswith("/api/admin/") and not user["is_admin"]:
            return JSONResponse({"detail": "Admins only."}, status_code=403)
        request.state.user = user
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _engine_status() -> str:
    if recommender.warmup_failed:
        return "WARM-UP FAILED"
    if not recommender.is_warmed_up:
        return "WARMING RISK ENGINE"
    return "FULLY OPERATIONAL"


def _live_scenario():
    return scenario_mgr.get(scenario_mgr.active_scenario_id)


# ============================================================================ auth
def _set_session(response: Response, request: Request, user) -> None:
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE_NAME, auth.issue(user), max_age=int(SESSION_DAYS * 86400), httponly=True,
                        samesite="lax", secure=secure, path="/")


@app.post("/api/auth/login")
def login(body: LoginBody, request: Request, response: Response):
    result = auth.authenticate(body.username, body.password, _client_ip(request))
    if "error" in result:
        raise HTTPException(result["status"], result["error"])
    _set_session(response, request, result["user"])
    return {"user": auth.public_user(result["user"])}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request):
    user = auth.verify(request.cookies.get(COOKIE_NAME))
    if not user:
        raise HTTPException(401, "Not signed in.")
    return {"user": auth.public_user(user)}


# ============================================================================ access requests (public)
_request_times: dict = {}
_request_lock = threading.Lock()


def _notify_access_request(req: dict) -> None:
    url = os.getenv("ACCESS_REQUEST_WEBHOOK", "").strip()
    if not url:
        return
    text = (f"New Supplychainer access request #{req['id']}: {req['name']} <{req['email']}>"
            f"{' - ' + req['company'] if req.get('company') else ''}\n{req.get('use_case') or ''}")
    try:  # Slack ("text") and Discord ("content") both accept this body
        httpx.post(url, json={"text": text, "content": text[:1900]}, timeout=8)
    except Exception as e:
        print(f"[ACCESS] Notification failed: {e}")


@app.post("/api/access-requests")
def create_access_request(body: AccessRequestBody, request: Request):
    if body.website:  # honeypot filled in: a bot. Pretend it worked.
        return {"ok": True}
    ip = _client_ip(request)
    now = time.time()
    with _request_lock:
        recent = [t for t in _request_times.get(ip, []) if now - t < 3600]
        if len(recent) >= 5:
            raise HTTPException(429, "Too many requests from this network. Please try again later.")
        _request_times[ip] = recent + [now]
    req = storage.add_access_request(body.name.strip(), body.email.strip().lower(), (body.company or "").strip() or None,
                                     (body.use_case or "").strip() or None, ip)
    threading.Thread(target=_notify_access_request, args=(req,), daemon=True).start()
    return {"ok": True}


# ============================================================================ admin
def _env_entry(username: str, password_hash: str) -> str:
    return f"{username}:{password_hash}"


@app.get("/api/admin/overview")
def admin_overview():
    return {
        "requests": storage.list_access_requests(),
        "users": [auth.public_user(u) for u in auth.list_users()],
        "storage_ephemeral": STORAGE_EPHEMERAL,
        "admin_username": auth.admin_username,
        "suppliers": supplier_directory.info(),
        "world": world.status(),
        "notify_webhook": bool(os.getenv("ACCESS_REQUEST_WEBHOOK")),
    }


@app.post("/api/admin/users")
def admin_create_user(body: UserCreate):
    try:
        username = normalize_username(body.username)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if auth.get_user(username):
        raise HTTPException(409, f"The username {username} is taken.")
    req = storage.get_access_request(body.request_id) if body.request_id else None
    password = body.password or generate_password()
    pw_hash = hash_password(password)
    storage.add_user(username, pw_hash, display_name=body.display_name or (req or {}).get("name"),
                     email=body.email or (req or {}).get("email"), is_admin=body.is_admin,
                     request_id=body.request_id)
    if req:
        storage.update_access_request(req["id"], "approved", username)
    return {"user": auth.public_user(auth.get_user(username)), "password": password,
            "env_entry": _env_entry(username, pw_hash), "storage_ephemeral": STORAGE_EPHEMERAL}


def _db_user_or_404(username: str):
    u = storage.get_user(username)
    if not u:
        if username in auth.env_users:
            raise HTTPException(400, "This account comes from the SUPPLYCHAINER_USERS env var; change it there.")
        raise HTTPException(404, "User not found")
    return u


@app.patch("/api/admin/users/{username}")
def admin_update_user(username: str, body: UserUpdate, request: Request):
    _db_user_or_404(username)
    if username == request.state.user["username"] and (body.active is False or body.is_admin is False):
        raise HTTPException(400, "You can't disable or demote your own account.")
    storage.update_user(username, active=body.active, is_admin=body.is_admin, display_name=body.display_name)
    return {"user": auth.public_user(auth.get_user(username))}


@app.post("/api/admin/users/{username}/reset-password")
def admin_reset_password(username: str):
    _db_user_or_404(username)
    password = generate_password()
    pw_hash = hash_password(password)
    storage.update_user(username, password_hash=pw_hash)
    return {"password": password, "env_entry": _env_entry(username, pw_hash), "storage_ephemeral": STORAGE_EPHEMERAL}


@app.delete("/api/admin/users/{username}")
def admin_delete_user(username: str, request: Request):
    _db_user_or_404(username)
    if username == request.state.user["username"]:
        raise HTTPException(400, "You can't delete your own account.")
    storage.delete_user(username)
    return {"deleted": username}


@app.patch("/api/admin/requests/{request_id}")
def admin_update_request(request_id: int, body: RequestUpdate):
    if not storage.update_access_request(request_id, body.status):
        raise HTTPException(404, "Request not found")
    return storage.get_access_request(request_id)


# ============================================================================ public (product page)
@app.get("/api/public/stats")
def public_stats():
    ws = world.status()
    G = recommender.unified_graph
    return {
        "ports": ws["ports"], "airports": ws["airports"], "countries": ws["countries"],
        "core_hubs": len(canonical_hubs),
        "lanes": sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "transit") // 2,
        "scenarios": len(scenario_mgr.get_all_scenarios()),
        "quantiles": sorted(predictor.models) or ["p50", "p85", "p95"],
        "engine_status": _engine_status(),
        "registry_updated_at": ws["generated_at"],
        "data_sources": ws["sources"] + [
            {"key": "openmeteo", "name": "Open-Meteo", "url": "https://open-meteo.com"},
            {"key": "news", "name": "Google News RSS", "url": "https://news.google.com"},
            {"key": "suppliers", "name": supplier_directory.info()["primary"] or "Wikidata",
             "url": "https://opensupplyhub.org" if supplier_directory.primary else "https://www.wikidata.org"},
        ],
    }


_example_cache: dict = {}


@app.get("/api/public/example")
def public_example():
    """A real plan (Shanghai -> Rotterdam during the Suez blockage) for the product page, cached for an hour."""
    if _example_cache.get("at", 0) < time.time() - 3600 or _example_cache.get("warm") != recommender.is_warmed_up:
        out = recommender.recommend("PORT-SHANGHAI", "PORT-ROTTERDAM", scenario="SUEZ_BLOCK", explain=False)
        if "error" in out:
            raise HTTPException(503, out["error"])
        routes = []
        for r in out["recommendations"]:
            transit = [l for l in r["legs"] if l["type"] == "transit"]
            via = [l["to_name"] for l in transit[:-1] if l.get("to_type") in ("choke_point", "port", "airport", "rail_hub")]
            routes.append({"personas": r["personas"], "eta_p85_h": r["adjusted_eta"], "eta_band": r["eta_band"],
                           "mode": r["primary_mode"], "via": via[:2], "cost": r["total_cost"],
                           "rerouted": (r.get("scenario_impact") or {}).get("rerouted", False),
                           "delta_eta_h": (r.get("scenario_impact") or {}).get("delta_eta")})
        _example_cache.update(at=time.time(), warm=recommender.is_warmed_up,
                              data={"origin": "Shanghai", "destination": "Rotterdam", "scenario": out["active_scenario"],
                                    "routes": routes, "solve_ms": out["engine"]["solve_ms"]})
    return _example_cache["data"]


_globe_cache: dict = {}


@app.get("/api/public/globe")
def public_globe():
    """Compact point cloud of every real port and airport plus the core sea/air lanes, for the 3D globe."""
    key = world.doc["generated_at"]
    if _globe_cache.get("key") != key:
        ports = [[round(p["lat"], 2), round(p["lon"], 2), p["importance"]] for p in world.doc["ports"]]
        airports = [[round(a["lat"], 2), round(a["lon"], 2)] for a in world.doc["airports"]]
        lanes, seen = [], set()
        for u, v, d in recommender.unified_graph.edges(data=True):
            if d.get("type") != "transit" or d["transport_mode"] not in ("sea", "air"):
                continue
            pu, pv = recommender.unified_graph.nodes[u]["physical_id"], recommender.unified_graph.nodes[v]["physical_id"]
            k = (min(pu, pv), max(pu, pv))
            if k in seen or d.get("distance", 0) < 800:
                continue
            seen.add(k)
            a, b = hub_index[pu], hub_index[pv]
            lanes.append([round(a["lat"], 2), round(a["lon"], 2), round(b["lat"], 2), round(b["lon"], 2),
                          d["transport_mode"][0]])
        _globe_cache.update(key=key, data={"ports": ports, "airports": airports, "lanes": lanes})
    return JSONResponse(_globe_cache["data"], headers={"Cache-Control": "public, max-age=3600"})


# ============================================================================ reference data
@app.get("/api/health")
def health():
    return {"ok": True, "engine_status": _engine_status()}

@app.get("/api/scenarios")
def get_scenarios():
    """Returns available disruption scenarios."""
    return scenario_mgr.get_all_scenarios()

@app.get("/api/hubs")
def get_hubs():
    """Returns the full canonical hub registry."""
    return canonical_hubs

@app.get("/api/hubs/search")
def search_hubs(q: str = Query(..., min_length=1), limit: int = 25, world_results: bool = True):
    """Search the core hubs, then every real port and airport (best matches first)."""
    q = q.lower().strip()
    scored = []
    for hub in canonical_hubs:
        fields = [hub["display_name"].lower(), hub["id"].lower(), (hub.get("parent_city") or "").lower(),
                  hub["country"].lower()] + [a.lower() for a in hub.get("aliases", [])]
        if not any(q in f for f in fields):
            continue
        rank = 0 if any(f.startswith(q) for f in fields) else 1
        scored.append((rank, -hub.get("importance", 5), hub["display_name"], hub))
    scored.sort(key=lambda x: x[:3])
    core = [h for *_, h in scored[:limit]]
    if not world_results or len(core) >= limit:
        return core
    # Real ports/airports that are the same place as a core hub are already covered by it.
    extra = [r for r in world.search(q, limit=limit * 2) if not r.get("core_alias")]
    return core + extra[:limit - len(core)]


@app.get("/api/world/status")
def world_status():
    return world.status()


@app.get("/api/conditions/{hub_id}")
def hub_conditions(hub_id: str):
    """Live weather (and sea state for ports) at a hub, port or airport."""
    hub = lookup_hub(hub_id)
    if not hub or hub.get("lat") is None:
        raise HTTPException(404, "Unknown hub")
    return {"hub": hub_id, "name": hub["display_name"],
            **conditions.at(hub["lat"], hub["lon"], marine=hub.get("type") == "port")}

@app.get("/api/network")
def get_network():
    """Physical hub graph for the map (virtual mode-nodes collapsed back to hubs)."""
    G = recommender.unified_graph
    nodes = [{"id": h["id"], "display_name": h["display_name"], "type": h["type"], "modes": h["modes"],
              "lat": h["lat"], "lon": h["lon"], "country": h["country"], "importance": h.get("importance", 5)}
             for h in canonical_hubs]
    edges, seen = [], set()
    for u, v, data in G.edges(data=True):
        if data.get("type") != "transit":
            continue
        pu, pv = G.nodes[u]["physical_id"], G.nodes[v]["physical_id"]
        key = (min(pu, pv), max(pu, pv), data["transport_mode"])
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": pu, "target": pv, "mode": data["transport_mode"],
                      "distance_km": data.get("distance"), "baseline_time": round(data.get("baseline_time", 0), 1)})
    return {"nodes": nodes, "edges": edges}

@app.get("/api/status")
def get_status():
    return {
        "ml_trained": predictor.is_trained,
        "quantiles": sorted(predictor.models),
        "nlp_ready": recommender.nlp.ready,
        "engine_status": _engine_status(),
        "is_supplychainer": True,
        "geo_scope": "Global (Canonical)",
        "hub_count": len(canonical_hubs),
        "live_scenario": _live_scenario(),
        "unacked_alerts": storage.unacked_count(),
    }

@app.get("/api/model")
def get_model_info():
    return predictor.model_info()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not auth.verify(websocket.cookies.get(COOKIE_NAME)):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    last_alert = storage.latest_alert_id()
    try:
        while True:
            fresh = storage.list_alerts(since_id=last_alert)
            if fresh:
                last_alert = max(a["id"] for a in fresh)
            live = _live_scenario()
            state = {
                "engine_status": _engine_status(),
                "ml_trained": predictor.is_trained,
                "nlp_ready": recommender.nlp.ready,
                "hub_registry": "Synchronized",
                "live_scenario": {"id": live["id"], "name": live["name"]} if live else None,
                "unacked_alerts": storage.unacked_count(),
                "new_alerts": list(reversed(fresh)),
            }
            await websocket.send_text(json.dumps(state))
            await asyncio.sleep(1.5)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        print(f"WebSocket closed: {e}")

@app.get("/api/cities")
def get_cities():
    """Returns the city-to-hub mapping for multimodal resolution."""
    path = os.path.join(os.path.dirname(__file__), 'data', 'canonical_locations.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

# ============================================================================ routing
@app.post("/api/recommend")
def recommend_routes(req: RecommendRequest):
    result = recommender.recommend(
        source=req.source,
        destination=req.destination,
        cargo_type=req.cargo_type,
        priority=req.priority,
        transport_preference=req.transport_preference,
        routing_policy=req.routing_policy,
        scenario=req.scenario,
        overrides=req.overrides
    )
    if req.save and "error" not in result:
        payload = req.model_dump(exclude={"save", "watch"})
        result["run_id"] = storage.save_run(payload, result)
        if req.watch:
            storage.update_run(result["run_id"], watched=True)
        dispatcher.emit("route.planned", {"run_id": result["run_id"], "origin": req.source,
                                          "destination": req.destination,
                                          "options": [{"personas": r["personas"], "eta_p85": r["adjusted_eta"],
                                                       "cost": r["total_cost"]} for r in result["recommendations"]]})
    return result

# ============================================================================ history
@app.get("/api/runs")
def list_runs(limit: int = 50, watched: bool = False):
    return storage.list_runs(limit=limit, watched_only=watched)

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run

@app.patch("/api/runs/{run_id}")
def update_run(run_id: str, upd: RunUpdate):
    if not storage.update_run(run_id, watched=upd.watched, label=upd.label, selected_index=upd.selected_index):
        raise HTTPException(404, "Run not found")
    run = storage.get_run(run_id)
    alert = None
    live = scenario_mgr.active_scenario_id
    if upd.watched and live:
        # Watching a route while an incident is already live must alert immediately.
        alert = alert_engine.evaluate_run(run, live)
    return {"run": {k: v for k, v in run.items() if k != "response"}, "alert": alert}

@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    if not storage.delete_run(run_id):
        raise HTTPException(404, "Run not found")
    return {"deleted": run_id}

@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str, format: str = Query("csv", pattern="^(csv|pdf|tms)$"), index: int = 0):
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    stem = f"supplychainer_{run['origin']}_{run['destination']}_{run_id}".replace(" ", "_")
    if format == "csv":
        return Response(export_csv(run, index), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})
    if format == "pdf":
        return Response(export_pdf(run, index), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.pdf"'})
    hubs = {**recommender.hubs, **hub_index}
    for leg in run["response"].get("recommendations", [{}])[index].get("legs", []):
        for hid in (leg["from"], leg["to"]):
            if hid not in hubs and lookup_hub(hid):
                hubs[hid] = lookup_hub(hid)
    return JSONResponse(export_tms(run, index, hubs=hubs),
                        headers={"Content-Disposition": f'attachment; filename="{stem}.tms.json"'})

# ============================================================================ live disruption + alerts
@app.get("/api/scenarios/live")
def get_live_scenario():
    return {"scenario": _live_scenario()}

@app.post("/api/scenarios/live")
def set_live_scenario(body: LiveScenario):
    if body.scenario_id and not scenario_mgr.get(body.scenario_id):
        raise HTTPException(400, f"Unknown scenario {body.scenario_id}")
    previous = scenario_mgr.active_scenario_id
    scenario_mgr.activate_scenario(body.scenario_id)
    storage.set_state("live_scenario", body.scenario_id)
    alerts = []
    if body.scenario_id and body.scenario_id != previous:
        alerts = alert_engine.on_scenario_activated(body.scenario_id)
    elif not body.scenario_id and previous:
        dispatcher.emit("scenario.cleared", {"scenario_id": previous})
    return {"scenario": _live_scenario(), "alerts_raised": alerts}

@app.get("/api/alerts")
def list_alerts(since: int = 0, limit: int = 100):
    return storage.list_alerts(since_id=since, limit=limit)

@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    if not storage.ack_alert(alert_id):
        raise HTTPException(404, "Alert not found")
    return {"acknowledged": alert_id}

@app.post("/api/alerts/ack-all")
def ack_all():
    n = 0
    for a in storage.list_alerts(limit=1000):
        if not a["acknowledged"]:
            storage.ack_alert(a["id"]); n += 1
    return {"acknowledged": n}

# ============================================================================ integrations
@app.get("/api/webhooks")
def list_webhooks():
    return [{**h, "secret": h["secret"][:6] + "…"} for h in storage.list_webhooks()]

def _block_on_public_demo():
    # Webhooks make this server send HTTP requests to arbitrary URLs. On a public, login-free
    # deployment that would let any visitor use the host machine to reach other addresses.
    if PUBLIC_MODE:
        raise HTTPException(403, "Webhooks are turned off on the public demo.")

@app.post("/api/webhooks")
def create_webhook(body: WebhookCreate):
    _block_on_public_demo()
    hook = storage.add_webhook(body.url, body.events)
    return hook  # the full secret is only returned once, at creation

@app.delete("/api/webhooks/{hook_id}")
def delete_webhook(hook_id: str):
    _block_on_public_demo()
    if not storage.delete_webhook(hook_id):
        raise HTTPException(404, "Webhook not found")
    return {"deleted": hook_id}

@app.post("/api/webhooks/test")
def test_webhooks():
    _block_on_public_demo()
    return {"deliveries": dispatcher.emit("ping", {"message": "Supplychainer webhook test"}, sync=True)}

@app.post("/api/intel/scan")
def intel_scan(body: IntelScan):
    """Live news scan for a route's hubs: Google News RSS -> contrastive NLP -> CARF -> threat type."""
    targets = [h for h in dict.fromkeys(body.hubs) if lookup_hub(h)][:8]

    def scan(hid):
        hub = lookup_hub(hid)
        mode = body.mode if body.mode in hub["modes"] else hub["modes"][0]
        place = hub.get("parent_city") or hub["display_name"]
        intel = recommender.news_ingestor.get_intel(place, mode)
        if intel["source"] == "FALLBACK":
            score = 0.0
        elif recommender.nlp.ready:
            score = recommender.nlp.get_semantic_score(intel["text"])
        else:
            score = keyword_severity(intel["text"])  # embedding model disabled on small hosts
        verdict = recommender.carf.assess(score, intel["text"], mode)
        cls = classify_threat(intel["text"]) if verdict["threat"] > 0 else {"category": "none", "evidence": []}
        return {"hub": hid, "name": hub["display_name"], "mode": mode, "source": intel["source"],
                "headlines": intel["headlines"], "semantic_score": round(score, 3), "carf": verdict,
                "threat": verdict["threat"], "category": cls["category"], "evidence": cls["evidence"],
                "fetched_at": datetime.fromtimestamp(intel["fetched_at"], timezone.utc).isoformat()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(scan, targets))
    return {"results": results, "nlp_ready": recommender.nlp.ready}

# ============================================================================ suppliers
@app.get("/api/suppliers/categories")
def supplier_categories():
    return supplier_scorer.categories()

@app.get("/api/suppliers/sources")
def supplier_sources():
    return supplier_directory.info()

@app.post("/api/suppliers")
def get_suppliers(req: SourcingRequest):
    # Stateless: the what-if scenario no longer overwrites the platform-wide live scenario.
    if req.category not in supplier_scorer.categories():
        raise HTTPException(400, f"Unknown category {req.category}")
    if "error" in recommender.resolve(req.destination):
        raise HTTPException(400, f"Unknown destination {req.destination}")
    active_disruptions = scenario_mgr.get_disruptions(req.scenario)
    try:
        ranked = supplier_scorer.rank(req.category, req.destination, iso=req.country,
                                      disruptions=active_disruptions, limit=req.limit)
    except SupplierSourceError as e:
        raise HTTPException(502, f"Live supplier data is unavailable right now: {e}")
    advice = supplier_scorer.get_procurement_advice(req.current_inventory, req.safety_stock, req.demand_forecast,
                                                    ranked=ranked["suppliers"])
    dest = lookup_hub(req.destination)
    return {
        **ranked,
        "destination": {"id": req.destination, "name": dest["display_name"] if dest else req.destination},
        "advice": advice,
        "active_disruptions": active_disruptions,
    }

# ============================================================================ built frontend (single-process deploy)
# dist/index.html is the public product page; dist/app/index.html is the signed-in tool.
# Windows can map .js to text/plain, which browsers refuse for service workers.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/manifest+json", ".webmanifest")
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(os.path.join(FRONTEND_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/app", include_in_schema=False)
    def app_root():
        return RedirectResponse("/app/")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        candidate = os.path.abspath(os.path.join(FRONTEND_DIST, full_path))
        if full_path and candidate.startswith(FRONTEND_DIST) and os.path.isfile(candidate):
            # The service worker must be revalidated on every load, or app updates stall.
            headers = {"Cache-Control": "no-cache"} if full_path.endswith(("sw.js", ".webmanifest")) else None
            return FileResponse(candidate, headers=headers)
        if full_path.startswith("app/") or full_path in ("login", "admin"):
            return FileResponse(os.path.join(FRONTEND_DIST, "app", "index.html"))
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
