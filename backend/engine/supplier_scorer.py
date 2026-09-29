import json
import os
from typing import List, Dict, Any, Optional


class SupplierScorer:
    """
    Supplychainer Supplier Intelligence Engine.
    Provides deterministic, disruption-aware supplier ranking and procurement advice.
    """
    WEIGHTS = {"cost": 0.3, "lead_time": 0.3, "reliability": 0.4}
    LEAD_TIME_HORIZON_DAYS = 30.0
    RISK_INFLATION_PER_THREAT = 0.3

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.suppliers = self._load_suppliers()

    def _load_suppliers(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                return json.load(f)
        return []

    def categories(self) -> List[str]:
        return sorted({s["category"] for s in self.suppliers})

    def _score(self, s: Dict[str, Any], cost_range, active_disruptions: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Cost Score — normalised within the category. The previous `1 - cost/1000` went
        #    negative for every Raw Materials supplier (unit cost > $1,000).
        lo, hi = cost_range
        cost_score = 1.0 if hi == lo else 1.0 - (s['unit_cost'] - lo) / (hi - lo)

        # 2. Lead Time Score — each disrupted node on the supplier's lane adds its full outage.
        #    (The old code added 50% of it while its comment claimed 10%; a shipment behind a
        #    closed canal waits out the whole closure.)
        exposure = []
        penalty_days, risk_inflation = 0.0, 0.0
        for node, impact in (active_disruptions or {}).items():
            if node == s['location_hub'] or node in s.get('transit_choke_points', []):
                exposure.append(node)
                penalty_days += impact['delay'] / 24.0
                risk_inflation += impact['threat'] * self.RISK_INFLATION_PER_THREAT
        effective_lead_time = s['base_lead_time_days'] + penalty_days
        lead_time_score = max(0.0, 1.0 - (effective_lead_time / self.LEAD_TIME_HORIZON_DAYS))

        # 3. Reliability Score — historical reliability minus disruption risk inflation
        reliability_score = max(0.03, min(0.98, s['historical_reliability'] - risk_inflation))

        total_score = (cost_score * self.WEIGHTS["cost"] + lead_time_score * self.WEIGHTS["lead_time"]
                       + reliability_score * self.WEIGHTS["reliability"])
        return {
            **s,
            "effective_lead_time": round(effective_lead_time, 1),
            "decision_score": round(total_score, 3),
            "risk_score": round(1.0 - reliability_score, 2),
            "exposed_to": exposure,
            "audit_trace": {
                "weights": self.WEIGHTS,
                "scores": {"cost": round(cost_score, 2), "lead_time": round(lead_time_score, 2),
                           "reliability": round(reliability_score, 2)},
                "penalties": {"lead_time_impact": round(penalty_days, 1), "risk_inflation": round(risk_inflation, 2)},
                "effective_metrics": {"lead_time_days": round(effective_lead_time, 1),
                                      "stability_index": round(reliability_score * 100, 0)},
            },
        }

    def get_ranked_suppliers(self, category: str, active_disruptions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Ranks suppliers based on cost, lead time, reliability, and active disruption impact."""
        filtered = [s for s in self.suppliers if s['category'] == category]
        if not filtered:
            return []
        costs = [s['unit_cost'] for s in filtered]
        cost_range = (min(costs), max(costs))

        scored = [self._score(s, cost_range, active_disruptions) for s in filtered]
        baseline = {s["id"]: self._score(s, cost_range, {})["decision_score"] for s in filtered}
        baseline_rank = {sid: i + 1 for i, sid in enumerate(sorted(baseline, key=lambda k: -baseline[k]))}

        scored.sort(key=lambda x: x['decision_score'], reverse=True)
        for i, s in enumerate(scored):
            s['rank'] = i + 1
            s['baseline_rank'] = baseline_rank[s['id']]
            s['rank_change'] = baseline_rank[s['id']] - s['rank']
        return scored

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
