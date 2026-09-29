import json
from typing import Dict, List, Any, Optional

class ScenarioManager:
    """
    Supplychainer Scenario Trigger Engine.
    Injects deterministic disruptions for judge-facing demonstrations.
    """
    SCENARIOS = {
        "SUEZ_BLOCK": {
            "name": "Suez Canal Blockage",
            "category": "infrastructure",
            "description": "Critical maritime corridor obstructed by vessel grounding.",
            "affected_nodes": ["CHOKE-SUEZ"],
            "threat_level": 1.0,
            "delay_hours": 240, # 10 days
            "reason": "Vessel grounding in Canal Narrows. Canal authority estimates 10-day salvage window.",
            "mode": "sea"
        },
        "RED_SEA_CONFLICT": {
            "name": "Red Sea Escalation",
            "category": "geopolitical",
            "description": "Increased regional instability affecting Bab el-Mandeb.",
            "affected_nodes": ["CHOKE-BABEL"],
            "threat_level": 0.85,
            "delay_hours": 72,
            "reason": "Regional conflict escalation. Vessels rerouting via Cape of Good Hope for risk mitigation.",
            "mode": "sea"
        },
        "LA_PORT_STRIKE": {
            "name": "LA Port Strike",
            "category": "labor",
            "description": "Labor dispute causing terminal shutdowns in Los Angeles.",
            "affected_nodes": ["PORT-LOSANGELES", "PORT-LONGBEACH"],
            "threat_level": 0.9,
            "delay_hours": 120,
            "reason": "Terminal labor strike. Picket lines at all major berths. Throughput at 0%.",
            "mode": "sea"
        },
        "CHENNAI_FLOOD": {
            "name": "Chennai Monsoon Flooding",
            "category": "weather",
            "description": "Extreme weather disrupting South India logistics.",
            "affected_nodes": ["PORT-CHENNAI", "HUB-CHENNAI"],
            "threat_level": 0.75,
            "delay_hours": 48,
            "reason": "Severe urban flooding. Inland road access to Port and Logistics Park is underwater.",
            "mode": "road"
        },
        "DUBAI_AIR_CONGESTION": {
            "name": "Dubai Hub Surge",
            "category": "congestion",
            "description": "Massive cargo backlog at DXB/DWC.",
            "affected_nodes": ["AIR-DUBAI"],
            "threat_level": 0.65,
            "delay_hours": 24,
            "reason": "Regional cargo surge exceeding ground handling capacity. 48h clearance backlog.",
            "mode": "air"
        },
        "HORMUZ_CLOSURE": {
            "name": "Hormuz Strait Escalation",
            "category": "geopolitical",
            "description": "Strategic maritime choke point tension.",
            "affected_nodes": ["CHOKE-HORMUZ"],
            "threat_level": 1.0,
            "delay_hours": 168,
            "reason": "Strategic naval activity. Vessels holding position at Jebel Ali / Colombo.",
            "mode": "sea"
        }
    }

    # Standing corridor advisories: structural conditions that are always true, independent of any
    # scenario. Without this the Arctic Northern Sea Route (a seasonal, ice-class-only lane) was
    # chosen as the default year-round Asia-Europe container route.
    ADVISORIES = {
        "CHOKE-NSR": {
            "delay": 96, "threat": 0.6, "category": "weather",
            "reason": "Seasonal Arctic lane: ice-class hull and icebreaker escort required; navigable roughly Jul-Nov only.",
        },
    }

    def __init__(self):
        # The *live* scenario: an incident broadcast to the whole platform (drives alerts on
        # watched routes). A scenario passed to /api/recommend is a per-request what-if and no
        # longer mutates this shared state (previously /api/suppliers and /api/recommend both
        # overwrote it, so concurrent users silently changed each other's results).
        self.active_scenario_id = None

    def get(self, scenario_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if scenario_id and scenario_id in self.SCENARIOS:
            return {"id": scenario_id, **self.SCENARIOS[scenario_id]}
        return None

    def activate_scenario(self, scenario_id: Optional[str]):
        """Sets the platform-wide live scenario. Returns the scenario or None."""
        if scenario_id and scenario_id in self.SCENARIOS:
            self.active_scenario_id = scenario_id
            return self.SCENARIOS[scenario_id]
        self.active_scenario_id = None
        return None

    def get_disruptions(self, scenario_id: Optional[str]) -> Dict[str, Any]:
        """Stateless lookup of the per-node disruptions a scenario injects."""
        scenario = self.SCENARIOS.get(scenario_id) if scenario_id else None
        if not scenario:
            return {}
        return {node: {
                    "delay": scenario["delay_hours"],
                    "threat": scenario["threat_level"],
                    "reason": scenario["reason"],
                    "category": scenario["category"],
                    "mode": scenario["mode"],
                    "scenario_id": scenario_id,
                    "source": "SCENARIO_OVERRIDE"
                } for node in scenario["affected_nodes"]}

    def get_advisories(self) -> Dict[str, Any]:
        return {node: {**a, "source": "ADVISORY", "scenario_id": None} for node, a in self.ADVISORIES.items()}

    def get_active_disruptions(self) -> Dict[str, Any]:
        return self.get_disruptions(self.active_scenario_id)

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self.SCENARIOS.items()]
