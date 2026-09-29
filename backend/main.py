from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from .engine.threat_intelligence import ThreatIntelligencePredictor, classify_threat, keyword_severity
from .engine.multimodal_network import load_canonical_hubs
from .engine.route_recommender import RouteRecommender
from .engine.scenario_manager import ScenarioManager
from .engine.supplier_scorer import SupplierScorer
from .engine.storage import Storage
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
supplier_scorer = SupplierScorer(os.path.join(os.path.dirname(__file__), 'data', 'suppliers.json'))

storage = Storage()
dispatcher = WebhookDispatcher(storage)
alert_engine = AlertEngine(storage, recommender, scenario_mgr, dispatcher)
scenario_mgr.activate_scenario(storage.get_state("live_scenario"))  # survives restarts


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
    current_inventory: int = 1000
    safety_stock: int = 1500
    demand_forecast: int = 800
    scenario: Optional[str] = None

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
    yield

app = FastAPI(title="Supplychainer API", version="2.0.0", lifespan=lifespan)

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
def search_hubs(q: str = Query(..., min_length=1), limit: int = 25):
    """Search hubs by display_name, aliases, city, or country (best matches first)."""
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
    return [h for *_, h in scored[:limit]]

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
    return JSONResponse(export_tms(run, index, hubs=hub_index),
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
    targets = [h for h in dict.fromkeys(body.hubs) if h in hub_index][:8]

    def scan(hid):
        hub = hub_index[hid]
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

@app.post("/api/suppliers")
def get_suppliers(req: SourcingRequest):
    # Stateless: the what-if scenario no longer overwrites the platform-wide live scenario.
    active_disruptions = scenario_mgr.get_disruptions(req.scenario)
    ranked_suppliers = supplier_scorer.get_ranked_suppliers(req.category, active_disruptions)
    advice = supplier_scorer.get_procurement_advice(req.current_inventory, req.safety_stock, req.demand_forecast,
                                                    ranked=ranked_suppliers)
    return {
        "suppliers": ranked_suppliers,
        "advice": advice,
        "active_disruptions": active_disruptions
    }

# ============================================================================ built frontend (single-process deploy)
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(os.path.join(FRONTEND_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        candidate = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
