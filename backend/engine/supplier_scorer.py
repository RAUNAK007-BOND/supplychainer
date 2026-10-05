import time
from typing import List, Dict, Any, Optional

from .world_registry import haversine


class SupplierScorer:
    """
    Supplychainer Supplier Intelligence Engine (live data).

    Suppliers are real facilities/companies from the SupplierDirectory (Open Supply Hub or Wikidata).
    Nothing about them is invented: every figure below is computed live for the buyer's delivery point.

      gateway      nearest real export port (world registry) to the supplier's site
      first mile   road leg site -> gateway (straight line x ROAD_FACTOR at effective truck speed)
      main leg     the routing engine's BALANCED p85 route gateway -> destination, with the
                   p50/p85/p95 band, the active disruption scenario and standing advisories
      weather      live Open-Meteo conditions at the gateway (gusts, waves, storms)

    Score = cost (freight estimate, normalised within the result set) + lead time + reliability
    (1 - lane risk, discounted by the width of the delay band).
    """
    WEIGHTS = {"cost": 0.3, "lead_time": 0.3, "reliability": 0.4}
    LEAD_TIME_HORIZON_DAYS = 60.0
    ROAD_FACTOR = 1.25          # road km per straight-line km
    ROAD_SPEED_KMH = 68.0       # matches the road profile used by the routing graph
    ROAD_COST_PER_KM = 1.2

    def __init__(self, directory, recommender, world, conditions):
        self.directory = directory
        self.recommender = recommender
        self.world = world
        self.conditions = conditions

    def categories(self) -> List[str]:
        return self.directory.categories()

    # ------------------------------------------------------------------ ranking
    def rank(self, category: str, destination: str, iso: Optional[str] = None,
             disruptions: Optional[Dict[str, Any]] = None, limit: int = 15) -> Dict[str, Any]:
        t0 = time.perf_counter()
        found = self.directory.find(category, iso=iso, limit=limit)
        facilities = found["facilities"]

        lanes, base_lanes = {}, {}

        def lane(origin_id):
            if origin_id not in lanes:
                lanes[origin_id] = self.recommender.quick_route(origin_id, destination, disruptions or {})
                base_lanes[origin_id] = (self.recommender.quick_route(origin_id, destination, {})
                                         if disruptions else lanes[origin_id])
            return lanes[origin_id]

        # 1. Where each supplier enters the network: the nearest core land hub in its country (the
        #    engine then picks the export port itself) or, failing that / if faster, the nearest seaport.
        plan, unreachable = [], []
        for f in facilities:
            options = []
            land = self.world.nearest_land_hub(f["lat"], f["lon"], f.get("iso"))
            if land:
                options.append((land[0], land[1], land[0]["id"]))
            port = self.world.nearest_port(f["lat"], f["lon"])
            if port:
                options.append((port[0], port[1], port[0].get("core_alias") or port[0]["id"]))
            scored = []
            for entry, km, origin_id in options:
                ln = lane(origin_id)
                if "error" not in ln:
                    scored.append((km * self.ROAD_FACTOR / self.ROAD_SPEED_KMH + ln["adjusted_eta"], entry, km, origin_id))
            if not scored:
                unreachable.append({"name": f["name"], "reason": "No route from this supplier's region."})
                continue
            _, entry, km, origin_id = min(scored, key=lambda s: s[0])
            plan.append((f, entry, km, origin_id))

        # 2. Live weather at each distinct gateway (the first port/airport the cargo leaves from)
        gateways = {}
        for _, entry, _, origin_id in plan:
            gw = self._gateway(entry, lanes[origin_id])
            gateways.setdefault(gw["id"], gw)
        weather = dict(zip(gateways, self.conditions.many(
            [{"lat": g["lat"], "lon": g["lon"], "marine": g.get("type") == "port"} for g in gateways.values()])))

        rows = []
        for f, entry, km, origin_id in plan:
            ln = lanes[origin_id]
            gw = self._gateway(entry, ln)
            rows.append(self._row(f, entry, km, gw, ln, base_lanes[origin_id], weather[gw["id"]]))

        if rows:
            self._score(rows)
            baseline = sorted(rows, key=lambda r: -r["_baseline_score"])
            base_rank = {r["id"]: i + 1 for i, r in enumerate(baseline)}
            rows.sort(key=lambda r: -r["decision_score"])
            for i, r in enumerate(rows):
                r["rank"] = i + 1
                r["baseline_rank"] = base_rank[r["id"]]
                r["rank_change"] = r["baseline_rank"] - r["rank"]
                r.pop("_baseline_score", None)

        return {"suppliers": rows, "unreachable": unreachable, "source": found["source"],
                "source_notes": found["notes"], "fetched_at": found["fetched_at"], "cached": found["cached"],
                "solve_ms": round((time.perf_counter() - t0) * 1000)}

    def _gateway(self, entry, lane) -> Dict[str, Any]:
        """The port or airport where the cargo leaves land: the start of the first sea/air leg."""
        hub = entry
        for leg in lane.get("legs", []):
            if leg["type"] == "transit" and leg["mode"] in ("SEA", "AIR"):
                hub = self.recommender.hubs.get(leg["from"]) or entry
                break
        return {"id": hub["id"], "name": hub["display_name"], "country": hub.get("country"), "type": hub.get("type"),
                "lat": hub["lat"], "lon": hub["lon"], "unlocode": hub.get("unlocode"),
                "harbor_size": hub.get("harbor_size")}

    def _row(self, f, entry, km, gw, lane, base, wx) -> Dict[str, Any]:
        road_km = km * self.ROAD_FACTOR
        first_mile_h = road_km / self.ROAD_SPEED_KMH
        first_mile_cost = road_km * self.ROAD_COST_PER_KM
        wx_delay = wx.get("delay_h", 0.0) if wx.get("ok") else 0.0

        lead_h = first_mile_h + lane["adjusted_eta"] + wx_delay
        base_lead_h = first_mile_h + base["adjusted_eta"] + wx_delay
        risk = max(lane["threat_level"], wx.get("threat", 0.0))
        base_risk = max(base["threat_level"], wx.get("threat", 0.0))
        band = lane["eta_band"]
        spread = (band["p95"] - band["p50"]) / max(band["p85"], 1.0)

        names = [self.recommender.hubs.get(h, {}).get("display_name", h) for h in lane["hubs"]]
        exposed = list(lane["exposed_disruptions"])
        return {
            "id": f["id"], "name": f["name"], "category": f["sector"], "source": f["source"], "url": f["url"],
            "site": f.get("site"), "country": self.world.doc["countries"].get(f.get("iso"), f.get("iso")),
            "iso": f.get("iso"), "city": f.get("city"), "lat": f["lat"], "lon": f["lon"],
            "gateway": gw,
            "entry": {"id": entry["id"], "name": entry["display_name"], "distance_km": round(km, 1)},
            "first_mile": {"km": round(road_km, 1), "hours": round(first_mile_h, 1), "cost": round(first_mile_cost, 2)},
            "route": {"hubs": lane["hubs"], "path": names, "primary_mode": lane["primary_mode"],
                      "eta_band": band, "legs": len([l for l in lane["legs"] if l["type"] == "transit"])},
            "weather": wx,
            "effective_lead_time": round(lead_h / 24, 1),
            "freight_cost": round(first_mile_cost + lane["total_cost"], 2),
            "exposed_to": exposed,
            "_risk": risk, "_spread": spread, "_base_lead_h": base_lead_h, "_base_risk": base_risk,
            "_base_cost": first_mile_cost + base["total_cost"],
            "audit_trace": {"penalties": {"lead_time_impact": round(max(0.0, lead_h - base_lead_h) / 24, 1),
                                          "weather_delay_h": wx_delay,
                                          "risk_inflation": round(max(0.0, risk - base_risk), 2)}},
        }

    def _reliability(self, risk, spread):
        return max(0.03, min(0.98, 0.98 * (1.0 - risk) * (1.0 - min(0.5, spread * 0.5))))

    def _score(self, rows):
        costs = [r["freight_cost"] for r in rows] + [r["_base_cost"] for r in rows]
        lo, hi = min(costs), max(costs)
        norm_cost = lambda c: 1.0 if hi == lo else 1.0 - (c - lo) / (hi - lo)
        lt_score = lambda h: max(0.0, 1.0 - (h / 24) / self.LEAD_TIME_HORIZON_DAYS)
        W = self.WEIGHTS
        for r in rows:
            cs, ls = norm_cost(r["freight_cost"]), lt_score(r["effective_lead_time"] * 24)
            rs = self._reliability(r["_risk"], r["_spread"])
            r["decision_score"] = round(cs * W["cost"] + ls * W["lead_time"] + rs * W["reliability"], 3)
            r["risk_score"] = round(1.0 - rs, 2)
            r["_baseline_score"] = (norm_cost(r["_base_cost"]) * W["cost"] + lt_score(r["_base_lead_h"]) * W["lead_time"]
                                    + self._reliability(r["_base_risk"], r["_spread"]) * W["reliability"])
            r["audit_trace"].update({
                "weights": W,
                "scores": {"cost": round(cs, 2), "lead_time": round(ls, 2), "reliability": round(rs, 2)},
                "effective_metrics": {"lead_time_days": r["effective_lead_time"],
                                      "stability_index": round(rs * 100)},
            })
            for k in ("_risk", "_spread", "_base_lead_h", "_base_risk", "_base_cost"):
                r.pop(k)

    # ------------------------------------------------------------------ procurement
    def get_procurement_advice(self, current_inventory: int, safety_stock: int, demand_forecast: int,
                               ranked: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Determines if a procurement escalation is required based on inventory shortage logic."""
        projected_stock = current_inventory - demand_forecast
        shortage = safety_stock - projected_stock

        status = "HEALTHY"
        recommendation = "Maintain current replenishment schedule."
        urgency = "LOW"

        if projected_stock <= 0:
            status = "CRITICAL_SHORTAGE"
            recommendation = "EMERGENCY REPLENISHMENT REQUIRED. Projected stockout in current cycle."
            urgency = "CRITICAL"
        elif projected_stock < safety_stock:
            status = "SAFETY_STOCK_VIOLATION"
            recommendation = "Expedite sourcing from high-reliability suppliers to restore safety buffers."
            urgency = "HIGH"

        advice = {
            "status": status,
            "shortage_quantity": max(0, shortage),
            "recommendation": recommendation,
            "urgency_level": urgency,
            "projected_inventory": projected_stock,
        }
        if ranked:
            top = ranked[0]
            advice["recommended_supplier"] = {"id": top["id"], "name": top["name"],
                                              "effective_lead_time": top["effective_lead_time"]}
            displaced = next((s for s in ranked if s["baseline_rank"] == 1), None)
            if displaced and displaced["id"] != top["id"]:
                advice["ranking_shift"] = (f"Disruption demoted {displaced['name']} (#1 under normal conditions, "
                                           f"now #{displaced['rank']}); {top['name']} is the new first choice.")
        return advice
