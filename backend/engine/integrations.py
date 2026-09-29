"""
Supplychainer integrations: real-time disruption alerts, webhooks and export formats.

* AlertEngine — when a scenario goes live, every *watched* route is checked for exposure; exposed
  routes get an alert that carries a re-plan computed under the new conditions.
* WebhookDispatcher — signed (HMAC-SHA256) JSON POSTs so a TMS/ERP can react without polling.
* Exporters — CSV audit trail, boardroom PDF, and `supplychainer.shipment_plan.v1` JSON, a
  TMS-ingestible plan with absolute planned timestamps per leg.
"""
import csv
import hashlib
import hmac
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

SEVERITY_BY_THREAT = [(0.85, "critical"), (0.6, "high"), (0.0, "medium")]


def _severity(threat: float) -> str:
    return next(s for t, s in SEVERITY_BY_THREAT if threat >= t)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ============================================================================ webhooks
class WebhookDispatcher:
    def __init__(self, storage, timeout: float = 5.0):
        self.storage = storage
        self.timeout = timeout

    @staticmethod
    def sign(secret: str, body: bytes) -> str:
        return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    def emit(self, event: str, data: Dict[str, Any], sync: bool = False) -> List[Dict[str, Any]]:
        hooks = self.storage.list_webhooks(event)
        if not hooks:
            return []
        envelope = {"event": event, "sent_at": _iso(datetime.now(timezone.utc)), "data": data}
        body = json.dumps(envelope, default=str).encode()
        results = []

        def deliver(hook):
            import httpx
            headers = {"Content-Type": "application/json", "X-Supplychainer-Event": event,
                       "X-Supplychainer-Signature": self.sign(hook["secret"], body)}
            try:
                r = httpx.post(hook["url"], content=body, headers=headers, timeout=self.timeout)
                status = f"{r.status_code} at {_iso(datetime.now(timezone.utc))}"
            except Exception as e:
                status = f"error: {type(e).__name__}"
            self.storage.set_webhook_status(hook["id"], status)
            results.append({"webhook_id": hook["id"], "status": status})

        for hook in hooks:
            if sync:
                deliver(hook)
            else:
                threading.Thread(target=deliver, args=(hook,), daemon=True).start()
        return results


# ============================================================================ alerts
class AlertEngine:
    def __init__(self, storage, recommender, scenario_mgr, dispatcher: WebhookDispatcher):
        self.storage = storage
        self.recommender = recommender
        self.scenario_mgr = scenario_mgr
        self.dispatcher = dispatcher

    @staticmethod
    def _selected(run) -> Optional[Dict[str, Any]]:
        recs = run["response"].get("recommendations", [])
        if not recs:
            return None
        return recs[min(run.get("selected_index", 0), len(recs) - 1)]

    def evaluate_run(self, run: Dict[str, Any], scenario_id: str) -> Optional[Dict[str, Any]]:
        """Creates an alert if the run's selected route passes through a hub the scenario disrupts."""
        disruptions = self.scenario_mgr.get_disruptions(scenario_id)
        route = self._selected(run)
        if not disruptions or not route:
            return None
        if run["request"].get("scenario") == scenario_id:
            return None  # the route was already planned under this scenario
        hit = [h for h in route.get("hubs", []) if h in disruptions]
        if not hit:
            return None

        scenario = self.scenario_mgr.get(scenario_id)
        worst = max(disruptions[h]["threat"] for h in hit)
        added = sum(disruptions[h]["delay"] for h in hit)
        persona = route.get("persona", "BALANCED")

        # Re-plan the same request under the new conditions.
        req = dict(run["request"])
        replan = self.recommender.recommend(
            source=req["source"], destination=req["destination"],
            transport_preference=req.get("transport_preference", "any"),
            routing_policy=req.get("routing_policy", "STRICT"), cargo_type=req.get("cargo_type", "general"),
            priority=req.get("priority", "normal"), scenario=scenario_id, overrides=req.get("overrides"),
            explain=False)
        suggestion = None
        for rec in replan.get("recommendations", []):
            if persona in rec.get("personas", []):
                suggestion = {
                    "persona": persona, "eta_p85": rec["adjusted_eta"], "eta_band": rec["eta_band"],
                    "total_cost": rec["total_cost"], "threat_level": rec["threat_level"],
                    "rerouted": rec.get("scenario_impact", {}).get("rerouted"),
                    "via": [l["to_name"] for l in rec["legs"] if l["type"] == "transit"],
                    "delta_eta_vs_original": round(rec["adjusted_eta"] - route["adjusted_eta"], 1),
                    "delta_cost_vs_original": round(rec["total_cost"] - route["total_cost"], 2),
                }
                break

        names = ", ".join(self.recommender.hubs.get(h, {}).get("display_name", h) for h in hit)
        label = {"FASTEST": "fastest", "SAFEST": "most reliable", "BALANCED": "best-value"}.get(persona, persona.lower())
        origin = self.recommender.hubs.get(run["origin"], {}).get("display_name", run["origin"])
        dest = self.recommender.hubs.get(run["destination"], {}).get("display_name", run["destination"])
        msg = (f"{scenario['name']} hits {names} on your {label} route from {origin} to {dest}. "
               f"Expect about {added / 24:.0f} extra days if nothing changes.")
        if suggestion:
            d = suggestion["delta_eta_vs_original"]
            msg += (f" Suggested plan: {'a new route' if suggestion['rerouted'] else 'the same route'} arriving in "
                    f"{suggestion['eta_p85'] / 24:.1f} days ({'+' if d >= 0 else '−'}{abs(d) / 24:.1f} days vs your plan).")
        payload = {"affected_hubs": hit, "announced_delay_h": added, "threat": worst,
                   "category": disruptions[hit[0]]["category"], "original_eta_p85": route["adjusted_eta"],
                   "replan": suggestion, "origin": run["origin"], "destination": run["destination"]}
        alert = self.storage.add_alert(_severity(worst), f"{scenario['name']} affects a route you’re watching", msg,
                                       run_id=run["id"], scenario_id=scenario_id, payload=payload)
        self.dispatcher.emit("alert.created", alert)
        return alert

    def on_scenario_activated(self, scenario_id: str) -> List[Dict[str, Any]]:
        alerts = []
        for summary in self.storage.list_runs(limit=500, watched_only=True):
            run = self.storage.get_run(summary["id"])
            alert = self.evaluate_run(run, scenario_id)
            if alert:
                alerts.append(alert)
        scenario = self.scenario_mgr.get(scenario_id)
        self.dispatcher.emit("scenario.activated", {"scenario": scenario, "alerts_raised": len(alerts)})
        return alerts


# ============================================================================ exports
def _route(run, index):
    recs = run["response"].get("recommendations", [])
    if not recs:
        raise ValueError("Run has no recommendations")
    return recs[max(0, min(index, len(recs) - 1))]


def export_csv(run: Dict[str, Any], index: int = 0) -> str:
    route = _route(run, index)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["# Supplychainer audit trail", f"run {run['id']}", f"{run['origin']} -> {run['destination']}",
                f"personas {'/'.join(route['personas'])}", f"scenario {run['scenario'] or 'none'}"])
    w.writerow(["# ETA p50/p85/p95 (h)", route["eta_band"]["p50"], route["eta_band"]["p85"], route["eta_band"]["p95"],
                "total cost (USD)", route["total_cost"], "peak threat", route["threat_level"]])
    w.writerow(["seq", "from_id", "from", "to_id", "to", "mode", "type", "distance_km", "transit_h",
                "buffer_p50_h", "buffer_p85_h", "buffer_p95_h", "event_delay_h", "leg_total_p85_h", "cost_usd",
                "threat", "threat_category", "intel_source", "model_profile", "reason"])
    for i, l in enumerate(route["legs"], 1):
        w.writerow([i, l["from"], l["from_name"], l["to"], l["to_name"], l["mode"], l["type"], l["distance_km"],
                    l["eta"], l["delay_band"]["p50"], l["delay_band"]["p85"], l["delay_band"]["p95"],
                    l["event_delay"], l["leg_total_p85"], l["cost"], l["threat"], l["threat_category"],
                    l["intel_source"], l.get("model_profile") or "", l["reason"]])
    t = route["audit_trace"]
    w.writerow([])
    w.writerow(["# audit_trace"] + [f"{k}.{kk}={vv}" for k, v in t.items() for kk, vv in v.items()])
    if route.get("drivers"):
        d = route["drivers"]
        w.writerow(["# shapley drivers", d["leg"], f"base {d['base_value_h']}h", f"model {d['model_output_h']}h"]
                   + [f"{c['feature']}={c['value']}:{c['hours']:+}h" for c in d["contributions"]])
    return buf.getvalue()


def export_tms(run: Dict[str, Any], index: int = 0, departure: Optional[datetime] = None,
               hubs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    route = _route(run, index)
    hubs = hubs or {}
    dep = departure or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=24)

    def place(hub_id, name, coords):
        h = hubs.get(hub_id, {})
        return {"hub_id": hub_id, "name": name, "country": h.get("country"), "type": h.get("type"),
                "lat": coords[0], "lon": coords[1]}

    legs, clock = [], dep
    # Per-leg durations are rounded to 0.1h; rescale so the schedule lands exactly on the route ETA.
    raw_total = sum(l["leg_total_p85"] for l in route["legs"]) or 1.0
    scale = route["eta_band"]["p85"] / raw_total
    for i, l in enumerate(route["legs"], 1):
        start = clock
        clock = start + timedelta(hours=l["leg_total_p85"] * scale)
        if i == len(route["legs"]):
            clock = dep + timedelta(hours=route["eta_band"]["p85"])
        legs.append({
            "sequence": i, "mode": l["mode"], "movement": "HANDLING" if l["type"] == "transfer" else "TRANSIT",
            "from": place(l["from"], l["from_name"], l["from_coords"]),
            "to": place(l["to"], l["to_name"], l["to_coords"]),
            "planned_departure": _iso(start), "planned_arrival_p85": _iso(clock),
            "transit_hours": l["eta"], "dwell_buffer_hours": l["delay_band"], "event_delay_hours": l["event_delay"],
            "distance_km": l["distance_km"], "cost": {"amount": l["cost"], "currency": "USD"},
            "risk": {"threat": l["threat"], "category": l["threat_category"], "source": l["intel_source"],
                     "note": l["reason"]},
        })
    first, last = route["legs"][0], route["legs"][-1]
    band = route["eta_band"]
    return {
        "schema": "supplychainer.shipment_plan.v1",
        "plan_id": f"{run['id']}-{index}",
        "generated_at": _iso(datetime.now(timezone.utc)),
        "shipment": {
            "origin": place(first["from"], first["from_name"], first["from_coords"]),
            "destination": place(last["to"], last["to_name"], last["to_coords"]),
            "cargo_type": run["request"].get("cargo_type"), "priority": run["request"].get("priority"),
        },
        "service_level": route["personas"],
        "planned_departure": _iso(dep),
        "eta": {q: _iso(dep + timedelta(hours=band[q])) for q in ("p50", "p85", "p95")},
        "total_cost": {"amount": route["total_cost"], "currency": "USD"},
        "risk": {"peak_threat": route["threat_level"], "exposed_disruptions": route.get("exposed_disruptions", []),
                 "scenario": run["scenario"]},
        "legs": legs,
    }


def export_pdf(run: Dict[str, Any], index: int = 0) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    route = _route(run, index)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title=f"Supplychainer route audit {run['id']}")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, alignment=0, spaceAfter=2,
                        textColor=colors.HexColor("#0f172a"))
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#334155"))
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4,
                        textColor=colors.HexColor("#1e3a8a"))

    created = datetime.fromtimestamp(run["created_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    band = route["eta_band"]
    story = [
        Paragraph(f"Route audit — {run['origin']} → {run['destination']}", h1),
        Paragraph(f"Run {run['id']} · generated {created} · scenario: {run['scenario'] or 'none (normal operations)'}"
                  f" · personas: {', '.join(route['personas'])}", small),
        Spacer(1, 6),
        Paragraph(route.get("explanation", ""), body),
    ]
    kpi = [["ETA p50", "ETA p85 (plan)", "ETA p95", "Total cost", "Peak threat", "Primary mode"],
           [f"{band['p50']} h", f"{band['p85']} h ({band['p85'] / 24:.1f} d)", f"{band['p95']} h",
            f"${route['total_cost']:,.0f}", f"{route['threat_level'] * 100:.0f}%", route["primary_mode"]]]
    t = Table(kpi, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                           ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                           ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"), ("PADDING", (0, 0), (-1, -1), 5)]))
    story += [Spacer(1, 8), t]

    if route.get("scenario_impact"):
        si = route["scenario_impact"]
        story.append(Paragraph(
            f"Scenario impact vs normal operations: {si['delta_eta']:+.0f} h, ${si['delta_cost']:+,.0f}"
            f" — {'rerouted' if si['rerouted'] else 'same path'}.", body))

    story.append(Paragraph("Leg-by-leg audit trail", h2))
    rows = [["#", "From → To", "Mode", "km", "Transit h", "Buffer p50/p85/p95 h", "Event h", "Cost $", "Threat", "Source"]]
    for i, l in enumerate(route["legs"], 1):
        rows.append([i, Paragraph(f"{l['from_name']} → {l['to_name']}", small), l["mode"], f"{l['distance_km']:,.0f}",
                     l["eta"], f"{l['delay_band']['p50']} / {l['delay_band']['p85']} / {l['delay_band']['p95']}",
                     l["event_delay"] or "", f"{l['cost']:,.0f}", f"{l['threat'] * 100:.0f}%", l["intel_source"]])
    lt = Table(rows, repeatRows=1, colWidths=[8 * mm, 88 * mm, 18 * mm, 16 * mm, 17 * mm, 36 * mm, 15 * mm,
                                             20 * mm, 15 * mm, 24 * mm])
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, l in enumerate(route["legs"], 1):
        if l["intel_source"] == "SCENARIO":
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fee2e2")))
        elif l["type"] == "transfer":
            style.append(("TEXTCOLOR", (0, i), (-1, i), colors.HexColor("#64748b")))
    lt.setStyle(TableStyle(style))
    story.append(lt)

    tr = route["audit_trace"]
    story.append(Paragraph("ETA & cost composition", h2))
    story.append(Paragraph(
        f"Transit {tr['eta']['transit']} h + handling {tr['eta']['transfer']} h + ML dwell buffer (p85) "
        f"{tr['eta']['ml_buffer_p85']} h + scenario {tr['eta']['scenario']} h + advisories {tr['eta']['advisory']} h. "
        f"Cost: transit ${tr['cost']['transit']:,.0f} + handling ${tr['cost']['transfer']:,.0f} + risk premium "
        f"${tr['cost']['scenario']:,.0f}.", body))

    if route.get("drivers"):
        d = route["drivers"]
        story.append(Paragraph(f"Why this delay? Shapley drivers for the critical leg ({d['leg']})", h2))
        story.append(Paragraph(f"Model baseline {d['base_value_h']} h → prediction {d['model_output_h']} h "
                               f"({d['method']}).", small))
        drows = [["Feature", "Value", "Contribution (h)"]] + [[c["feature"], str(c["value"]), f"{c['hours']:+.2f}"]
                                                             for c in d["contributions"]]
        dt = Table(drows, hAlign="LEFT")
        dt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1"))]))
        story.append(dt)

    story.append(Spacer(1, 8))
    story.append(Paragraph("ETA band sums per-leg quantiles (comonotonic assumption) and is therefore a conservative "
                           "envelope. Delay buffers come from GradientBoosting quantile models (p50/p85/p95) with "
                           "per-mode historical calibration.", small))
    doc.build(story)
    return buf.getvalue()
