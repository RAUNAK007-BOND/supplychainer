import networkx as nx
import math
import time
from typing import List, Dict, Any, Optional, Tuple
from .multimodal_network import (MODE_PROFILES, PRIORITY_MULTIPLIERS, TRANSFER_PROFILES, _travel_time,
                                 create_multimodal_network)
from .threat_intelligence import (ThreatIntelligencePredictor, ContrastiveNLPEngine, CARFFilter,
                                  classify_threat, DIRECT_MODEL_NODES)
from .news_ingestion import DynamicNewsIngestor
from .node_resolver import NodeResolver

PERSONAS = ["FASTEST", "BALANCED", "SAFEST"]
SAFEST_RISK_MULTIPLIER = 12.0
# Oversize/heavy cargo may still use trucks for first/last mile, but not for long-haul road legs.
OVERSIZE_ROAD_LIMIT_KM = 300
PREFERRED_MODE_PENALTY = 1.5
# Seasonal lanes (ice-class only, ~Jul-Nov) are excluded unless the caller opts in.
SEASONAL_HUBS = {"CHOKE-NSR"}


def _cargo_bans() -> Dict[str, set]:
    """cargo_type -> set of modes that may not carry it (from MODE_PROFILES cargo_restrictions)."""
    bans: Dict[str, set] = {}
    for mode, profile in MODE_PROFILES.items():
        for cargo in profile.get("cargo_restrictions", []):
            bans.setdefault(cargo, set()).add(mode)
    return bans


CARGO_BANS = _cargo_bans()


class RouteRecommender:
    """
    Supplychainer Unified Multimodal Optimization Engine.
    V9: Quantile-risk routing — the p85 model now drives the path choice, with a p50/p95 band.
    """

    def __init__(self, network, predictor, simulator, scenario_mgr, demo_mode=False):
        self.network = network # Legacy
        self.predictor = predictor
        self.simulator = simulator
        self.scenario_mgr = scenario_mgr
        self.demo_mode = demo_mode
        self.is_warmed_up = False
        self.warmup_failed = False

        self.nlp = ContrastiveNLPEngine(lazy_load=True)
        self.carf = CARFFilter()
        self.news_ingestor = DynamicNewsIngestor()
        self.resolver = NodeResolver()
        self.hubs = {h["id"]: h for h in self.resolver.hubs}
        self.advisories = scenario_mgr.get_advisories() if hasattr(scenario_mgr, "get_advisories") else {}
        self.world = None  # WorldRegistry: real ports/airports attached to the graph on demand
        self._baseline_intel: Dict[str, Tuple[float, str]] = {}
        self._general_graph = None

        print(f"[STARTUP] Initializing Split-Node Global Topology...")
        self.unified_graph = create_multimodal_network()
        self._bands_ready = False
        self._annotate_graph()

        if self.demo_mode:
            self.is_warmed_up = True

        print(f"[STARTUP] Unified Engine Ready.")

    # ------------------------------------------------------------------ setup
    def _annotate_graph(self):
        """Attaches the model-profile mapping and a neutral intel baseline to every transit edge."""
        for u, v, d in self.unified_graph.edges(data=True):
            d.setdefault("base_threat", 0.0)
            d.setdefault("base_news", "No live intelligence ingested for this corridor.")
            d.setdefault("threat_category", "none")
            if d["type"] == "transfer":
                continue
            mode = d["transport_mode"]
            hu = self.hubs.get(self.unified_graph.nodes[u]["physical_id"])
            hv = self.hubs.get(self.unified_graph.nodes[v]["physical_id"])
            d["model_nodes"] = (self.predictor.model_node_for(hu, mode, "origin"),
                                self.predictor.model_node_for(hv, mode, "destination"))
            # The model predicts terminal dwell. Sailing *through* an open strait is not a port
            # call, so only hubs and canals (which queue) carry a dwell buffer.
            d["dwell"] = not (hv and hv.get("type") == "choke_point" and hv["id"] not in DIRECT_MODEL_NODES)

    def run_background_warmup(self):
        if self.is_warmed_up: return
        print("[WARMUP] Calibrating global threat floor...")
        try:
            self.predictor.warmup()
            self.nlp.warmup()

            # Enrich unified graph with baseline intelligence (NLP -> CARF per mode)
            baseline = {}
            for mode, news in self.news_ingestor.fallback_news.items():
                score = self.nlp.get_semantic_score(news)
                verdict = self.carf.assess(score, news, mode)
                baseline[mode] = (verdict["threat"], news)
            self._baseline_intel = baseline
            for u, v, d in self.unified_graph.edges(data=True):
                mode = d.get("transport_mode", "road")
                if mode == "transfer": continue
                threat, news = baseline.get(mode, (0.0, "Normal conditions."))
                d["base_threat"], d["base_news"] = threat, news
                d["threat_category"] = classify_threat(news)["category"] if threat > 0 else "none"

            # Pre-compute the p50/p85/p95 band for every corridor in one vectorised pass.
            self._precompute_bands()
            self.is_warmed_up = True
            print("[WARMUP] Unified Calibration Complete.")
        except Exception as e:
            print(f"[WARMUP] Error during warmup: {e}")
            self.warmup_failed = True

    def _precompute_bands(self):
        self._bands_ready = True
        edges = [(u, v, d) for u, v, d in self.unified_graph.edges(data=True) if d["type"] == "transit"]
        keys = [self._ml_key(d, d["base_threat"]) for _, _, d in edges]
        for (_, _, d), band in zip(edges, self.predictor.predict_band_many(keys)):
            d["band"] = band
        self._general_graph = None  # graph copies hold their own edge dicts; rebuild from the fresh bands

    def _ml_key(self, d: Dict[str, Any], threat: float) -> tuple:
        o, dst = d["model_nodes"]
        return self.predictor.make_key(o, dst, d["transport_mode"], threat)

    # ------------------------------------------------------------------ leg physics
    def _leg(self, G, u, v, d, disruptions) -> Dict[str, Any]:
        """Everything the solver and the audit trail need to know about traversing u -> v."""
        pu, pv = G.nodes[u]["physical_id"], G.nodes[v]["physical_id"]
        leg = {
            "time": d["baseline_time"], "cost": d.get("cost", 0.0), "threat": d.get("base_threat", 0.0),
            "category": d.get("threat_category", "none"), "news": d.get("base_news", ""),
            "source": "BASELINE", "event_delay": 0.0, "band": {"p50": 0.0, "p85": 0.0, "p95": 0.0},
            "ml_key": None, "event_hub": None,
        }
        # Scenario disruptions override standing advisories at the same hub.
        event = disruptions.get(pv) or self.advisories.get(pv)
        if event:
            leg["threat"] = max(leg["threat"], event["threat"])
            leg["news"], leg["category"] = event["reason"], event["category"]
            leg["source"] = "ADVISORY" if event["source"] == "ADVISORY" else "SCENARIO"
            leg["event_hub"] = pv
            # A hub outage is paid once, when the cargo *enters* the hub. Moving between that hub's
            # own mode-nodes (transfer edges) must not add the outage again - previously a
            # road -> transfer -> sea path through a struck port paid the strike delay twice.
            if pu != pv:
                leg["event_delay"] = float(event["delay"])

        ed = leg["event_delay"]
        if d["type"] == "transit" and d.get("dwell", True):
            key = self._ml_key(d, leg["threat"])
            ml = self.predictor.predict_band_many([key])[0] if (event or "band" not in d) else d["band"]
            leg["ml_key"] = key
            # An announced outage is a floor on the model's quantile, not an addition to it:
            # both describe the same dwell at the same hub.
            leg["band"] = {q: max(ed, ml[q]) for q in ("p50", "p85", "p95")}
            leg["calibration"] = ml.get("calibration_reason")
        else:
            leg["band"] = {q: ed for q in ("p50", "p85", "p95")}
        return leg

    # ------------------------------------------------------------------ real-world endpoints
    def resolve(self, location: str) -> Dict[str, Any]:
        """Entry virtual node for a core hub / city, or for a real port or airport from the world registry.
        World records come back with `attach` set: they must be wired into a graph copy before solving."""
        if self.world is not None and self.world.is_world_id(location):
            rec = self.world.get(location)
            if not rec:
                return {"error": f"Unknown port or airport {location}"}
            if rec.get("core_alias"):
                return self.resolver.resolve_node_to_entry_point(rec["core_alias"])
            feeders = self.world.feeders(rec)
            if not feeders:
                return {"error": f"{rec['display_name']} has no connection to the routing network."}
            modes = {m for _, m, _ in feeders}
            entry = "road" if "road" in modes else rec["modes"][0]
            return {"id": f"{rec['id']}:{entry}", "attach": rec, "feeders": feeders}
        return self.resolver.resolve_node_to_entry_point(location)

    def _attach(self, G, rec: Dict[str, Any], feeders) -> None:
        """Adds a real port/airport to G with feeder legs to the nearest core hubs (both directions)."""
        rid = rec["id"]
        self.hubs.setdefault(rid, {**rec, "parent_city": rec.get("city") or rec["display_name"]})
        hub = self.hubs[rid]
        modes = sorted({m for _, m, _ in feeders})
        for mode in modes:
            G.add_node(f"{rid}:{mode}", physical_id=rid, display_name=rec["display_name"], type=rec["type"],
                       country=rec["country"], lat=rec["lat"], lon=rec["lon"], importance=rec["importance"],
                       mode=mode, parent_city=hub["parent_city"])
        for i, m1 in enumerate(modes):
            for m2 in modes[i + 1:]:
                p = TRANSFER_PROFILES["road_to_sea" if "sea" in (m1, m2) else "road_to_air"]
                for a, b in ((m1, m2), (m2, m1)):
                    G.add_edge(f"{rid}:{a}", f"{rid}:{b}", baseline_time=p["delay"], distance=0.1,
                               transport_mode="transfer", type="transfer", cost=p["cost"], risk=p["risk"],
                               base_threat=0.0, base_news="", threat_category="none")
        new_edges = []
        for core_id, mode, dist in feeders:
            core_node = f"{core_id}:{mode}"
            if not G.has_node(core_node):
                continue
            dist = max(dist, 1.0)
            threat, news = self._baseline_intel.get(mode, (0.0, "No live intelligence ingested for this corridor."))
            core = self.hubs.get(core_id)
            for a, b, ha, hb in ((f"{rid}:{mode}", core_node, hub, core), (core_node, f"{rid}:{mode}", core, hub)):
                G.add_edge(a, b, baseline_time=_travel_time(dist, mode), distance=round(dist, 1),
                           transport_mode=mode, type="transit", cost=dist * MODE_PROFILES[mode]["cost_per_km"],
                           base_threat=threat, base_news=news,
                           threat_category=classify_threat(news)["category"] if threat > 0 else "none",
                           model_nodes=(self.predictor.model_node_for(ha, mode, "origin"),
                                        self.predictor.model_node_for(hb, mode, "destination")),
                           dwell=not (hb and hb.get("type") == "choke_point" and hb["id"] not in DIRECT_MODEL_NODES))
                new_edges.append(G[a][b])
        if new_edges:
            for d, band in zip(new_edges, self.predictor.predict_band_many(
                    [self._ml_key(d, d["base_threat"]) for d in new_edges])):
                d["band"] = band

    def _base_graph(self, *resolved):
        attach = [r for r in resolved if r.get("attach")]
        if not attach:
            return self.unified_graph
        G = self.unified_graph.copy()
        for r in attach:
            self._attach(G, r["attach"], r["feeders"])
        return G

    def quick_route(self, source: str, destination: str, disruptions: Dict[str, Any] = None,
                    persona: str = "BALANCED") -> Dict[str, Any]:
        """One persona, general cargo, no explanation: the cheap path used to score many suppliers at once."""
        rs, rd = self.resolve(source), self.resolve(destination)
        if "error" in rs: return {"error": rs["error"]}
        if "error" in rd: return {"error": rd["error"]}
        if rs["id"] == rd["id"]:
            return {"eta_band": {"p50": 0.0, "p85": 0.0, "p95": 0.0}, "adjusted_eta": 0.0, "total_cost": 0.0,
                    "threat_level": 0.0, "hubs": [rs["id"].split(":")[0]], "legs": [], "exposed_disruptions": [],
                    "primary_mode": "ROAD"}
        if not self._bands_ready:
            self._precompute_bands()
        if self._general_graph is None:
            # Built once: general cargo with no preference never drops an edge, only seasonal lanes.
            self._general_graph = self._persona_graph([], "any", "STRICT", "general")
        G = self._general_graph
        if rs.get("attach") or rd.get("attach"):
            G = G.copy()
            for r in (rs, rd):
                if r.get("attach"):
                    self._attach(G, r["attach"], r["feeders"])
        try:
            path = self._solve(G, rs["id"], rd["id"], persona, disruptions or {}, "any", "STRICT", "normal")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"error": "No route found."}
        cand = self._compose(G, path, persona, disruptions or {})
        for l in cand["legs"]:
            l.pop("_ml_key", None)
        return cand

    # ------------------------------------------------------------------ solver
    def _persona_graph(self, avoid_hubs, transport_preference, routing_policy, cargo_type, allow_seasonal=False,
                       base=None):
        G = (base if base is not None else self.unified_graph).copy()
        excluded = set(avoid_hubs) | (set() if allow_seasonal else SEASONAL_HUBS)
        for hub_id in excluded:
            G.remove_nodes_from([n for n, d in G.nodes(data=True) if d.get("physical_id") == hub_id])

        banned = CARGO_BANS.get(cargo_type, set())
        drop = []
        for u, v, d in G.edges(data=True):
            mode = d["transport_mode"]
            if mode in banned:
                # Road is also the first/last-mile mode, so only long-haul trucking is excluded.
                if mode == "road" and d.get("distance", 0) <= OVERSIZE_ROAD_LIMIT_KM:
                    continue
                drop.append((u, v))
            elif transport_preference != "any" and routing_policy == "STRICT" \
                    and mode not in (transport_preference, "transfer", "road"):
                drop.append((u, v))
        G.remove_edges_from(drop)
        return G

    def _solve(self, G, s, t, persona, disruptions, transport_preference, routing_policy, priority):
        mult = PRIORITY_MULTIPLIERS.get(priority, 1.0)
        time_w, cost_w, risk_w = 0.3 / mult, 0.5 * mult, 0.2

        def weight(u, v, d):
            leg = self._leg(G, u, v, d, disruptions)
            t85 = leg["time"] + leg["band"]["p85"]
            if persona == "FASTEST":
                w = t85
            elif persona == "SAFEST":
                w = (leg["time"] + leg["band"]["p95"]) * (1.0 + leg["threat"] * SAFEST_RISK_MULTIPLIER)
            else:  # BALANCED (economic leaning)
                w = t85 * time_w + (leg["cost"] / 150.0) * cost_w + (leg["threat"] * 40.0) * risk_w
            # PREFERRED policy was accepted by the API but never applied: now a soft bias.
            if routing_policy == "PREFERRED" and transport_preference != "any" \
                    and d["transport_mode"] not in (transport_preference, "transfer", "road"):
                w *= PREFERRED_MODE_PENALTY
            return w

        return nx.dijkstra_path(G, s, t, weight=weight)

    def _compose(self, G, path, persona, disruptions) -> Dict[str, Any]:
        legs = []
        trace = {
            "eta": {"transit": 0.0, "transfer": 0.0, "ml_buffer_p85": 0.0, "scenario": 0.0, "advisory": 0.0},
            "cost": {"transit": 0.0, "transfer": 0.0, "scenario": 0.0},
            "risk": {"baseline": 0.0, "scenario": 0.0},
        }
        band_total = {"p50": 0.0, "p85": 0.0, "p95": 0.0}
        base_total, total_cost, max_threat = 0.0, 0.0, 0.0
        exposed, advisories_hit = set(), set()

        origin_pid = G.nodes[path[0]]["physical_id"]
        origin_disruption = disruptions.get(origin_pid)

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            d = G[u][v]
            leg = self._leg(G, u, v, d, disruptions)
            if i == 0 and origin_disruption:
                # Cargo departing a disrupted hub waits out the outage before the first leg.
                od = float(origin_disruption["delay"])
                leg["event_delay"] += od
                leg["band"] = {q: leg["band"][q] + od for q in leg["band"]}
                leg["threat"] = max(leg["threat"], origin_disruption["threat"])
                leg["source"], leg["news"] = "SCENARIO", origin_disruption["reason"]
                leg["category"] = origin_disruption["category"]
                exposed.add(origin_pid)
            if leg["event_hub"]:
                (exposed if leg["source"] == "SCENARIO" else advisories_hit).add(leg["event_hub"])

            pu, pv = G.nodes[u]["physical_id"], G.nodes[v]["physical_id"]
            hu, hv = self.hubs.get(pu, {}), self.hubs.get(pv, {})
            risk_premium = leg["cost"] * 0.1 if leg["source"] == "SCENARIO" else 0.0
            ml_part = leg["band"]["p85"] - leg["event_delay"]

            bucket = "transfer" if d["type"] == "transfer" else "transit"
            trace["eta"][bucket] += leg["time"]
            trace["eta"]["scenario" if leg["source"] == "SCENARIO" else "advisory"] += leg["event_delay"]
            trace["eta"]["ml_buffer_p85"] += max(0.0, ml_part)
            trace["cost"][bucket] += leg["cost"]
            trace["cost"]["scenario"] += risk_premium
            if leg["source"] == "SCENARIO":
                trace["risk"]["scenario"] = max(trace["risk"]["scenario"], leg["threat"])
            else:
                trace["risk"]["baseline"] = max(trace["risk"]["baseline"], leg["threat"])

            base_total += leg["time"]
            for q in band_total: band_total[q] += leg["band"][q]
            total_cost += leg["cost"] + risk_premium
            max_threat = max(max_threat, leg["threat"])

            legs.append({
                "from": pu, "from_name": hu.get("display_name", pu),
                "to": pv, "to_name": hv.get("display_name", pv),
                "to_type": hv.get("type"),
                "from_coords": [hu.get("lat"), hu.get("lon")], "to_coords": [hv.get("lat"), hv.get("lon")],
                "mode": d["transport_mode"].upper(), "type": d["type"],
                "distance_km": d.get("distance", 0.0),
                "eta": round(leg["time"], 1),
                "delay_band": {q: round(x, 1) for q, x in leg["band"].items()},
                "event_delay": round(leg["event_delay"], 1),
                "leg_total_p85": round(leg["time"] + leg["band"]["p85"], 1),
                "cost": round(leg["cost"] + risk_premium, 2),
                "threat": round(leg["threat"], 2),
                "threat_category": leg["category"],
                "reason": leg["news"],
                "intel_source": leg["source"],
                "model_profile": " → ".join(d["model_nodes"]) if leg["ml_key"] else None,
                "calibration": leg.get("calibration"),
                "_ml_key": leg["ml_key"],
            })

        eta_band = {q: round(base_total + band_total[q], 1) for q in band_total}
        for k in trace:
            trace[k] = {kk: round(vv, 2) for kk, vv in trace[k].items()}
        route_hubs = [legs[0]["from"]] + [l["to"] for l in legs] if legs else []
        return {
            "persona": persona,
            "personas": [persona],
            "primary_mode": self._primary_mode(legs),
            "mode_mix": self._mode_mix(legs),
            "legs": legs,
            "hubs": list(dict.fromkeys(route_hubs)),
            "baseline_eta": round(base_total, 1),
            "adjusted_eta": eta_band["p85"],
            "eta_band": eta_band,
            "total_cost": round(total_cost, 2),
            "threat_level": round(max_threat, 2),
            "exposed_disruptions": sorted(exposed),
            "advisories": sorted(advisories_hit),
            "audit_trace": trace,
        }

    @staticmethod
    def _mode_mix(legs):
        mix = {}
        for l in legs:
            if l["type"] == "transit":
                mix[l["mode"]] = round(mix.get(l["mode"], 0) + l["distance_km"], 1)
        return mix

    def _primary_mode(self, legs):
        mix = self._mode_mix(legs)
        if not mix: return "ROAD"
        top = max(mix, key=mix.get)
        return top if mix[top] >= 0.8 * sum(mix.values()) else "MULTIMODAL"

    # ------------------------------------------------------------------ public API
    def recommend(self, source: str, destination: str, transport_preference: str = "any",
                  routing_policy: str = "STRICT", cargo_type: str = "general",
                  priority: str = "normal", scenario: str = None,
                  overrides: dict = None, explain: bool = True) -> dict:

        t0 = time.perf_counter()
        overrides = overrides or {}
        avoid_hubs = overrides.get("avoid_chokepoints", [])
        cost_ceiling = overrides.get("cost_ceiling", 999999) or 999999
        max_delay = overrides.get("max_delay", 9999) or 9999  # days

        # 1. Resolve Entry/Exit (Virtual Nodes)
        res_s = self.resolve(source)
        res_d = self.resolve(destination)

        if "error" in res_s: return {"error": res_s["error"]}
        if "error" in res_d: return {"error": res_d["error"]}
        s_vnode, d_vnode = res_s["id"], res_d["id"]
        if s_vnode == d_vnode:
            return {"error": "Origin and destination resolve to the same hub."}
        base_graph = self._base_graph(res_s, res_d)

        if not self._bands_ready:
            self._precompute_bands()

        # 2. Scenario (per-request what-if; does not mutate the platform-wide live scenario)
        active_scenario = self.scenario_mgr.get(scenario)
        disruptions = self.scenario_mgr.get_disruptions(scenario)

        # 3. Persona Optimization
        candidates, rejected = [], []
        for persona in PERSONAS:
            try:
                G_p = self._persona_graph(avoid_hubs, transport_preference, routing_policy, cargo_type,
                                          allow_seasonal=bool(overrides.get("allow_seasonal_lanes")),
                                          base=base_graph)
                path = self._solve(G_p, s_vnode, d_vnode, persona, disruptions,
                                   transport_preference, routing_policy, priority)
                cand = self._compose(G_p, path, persona, disruptions)
                if cand["total_cost"] > cost_ceiling or cand["adjusted_eta"] > max_delay * 24:
                    rejected.append(persona)
                    continue
                cand["override_applied"] = bool(avoid_hubs or cost_ceiling < 999999 or max_delay < 9999)
                if disruptions:
                    cand["scenario_impact"] = self._scenario_impact(G_p, s_vnode, d_vnode, persona, cand,
                                                                    transport_preference, routing_policy, priority)
                candidates.append(cand)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            except Exception as e:
                print(f"[ROUTING ERROR] {persona}: {e}")

        if not candidates:
            reason = " (all candidates exceeded the cost/delay limits)" if rejected else ""
            return {"error": f"No valid multimodal route under current strategic constraints{reason}."}

        # 4. Merge personas that converged on the same physical path
        final: List[Dict[str, Any]] = []
        by_sig = {}
        for c in candidates:
            sig = tuple((l["to"], l["mode"]) for l in c["legs"])
            if sig in by_sig:
                by_sig[sig]["personas"].append(c["persona"])
            else:
                by_sig[sig] = c
                final.append(c)

        for c in final:
            c["explanation"] = self._explain_choice(c, final, disruptions)
            if explain:
                c["drivers"] = self._drivers(c)
            for l in c["legs"]:
                l.pop("_ml_key", None)

        return {
            "origin": source, "destination": destination,
            "origin_hub": res_s["id"].split(":")[0], "destination_hub": res_d["id"].split(":")[0],
            "active_scenario": active_scenario["name"] if active_scenario else None,
            "scenario": active_scenario,
            "request": {"transport_preference": transport_preference, "routing_policy": routing_policy,
                        "cargo_type": cargo_type, "priority": priority, "scenario": scenario,
                        "overrides": overrides},
            "engine": {"ml_model": "GradientBoosting quantile (p50/p85/p95)" if self.predictor.is_trained
                       else "deterministic priors (models not loaded)",
                       "nlp_ready": self.nlp.ready, "warmed_up": self.is_warmed_up,
                       "solve_ms": round((time.perf_counter() - t0) * 1000, 1),
                       "band_note": "Route band sums per-leg quantiles (comonotonic): a conservative envelope."},
            "recommendations": final[:3],
        }

    def _scenario_impact(self, G, s, t, persona, cand, pref, policy, priority) -> Dict[str, Any]:
        """What the disruption did to this persona's decision, vs. the same request with no scenario."""
        try:
            base = self._compose(G, self._solve(G, s, t, persona, {}, pref, policy, priority), persona, {})
        except nx.NetworkXNoPath:
            return {}
        rerouted = [l["to"] for l in base["legs"]] != [l["to"] for l in cand["legs"]]
        return {
            "rerouted": rerouted,
            "normal_eta": base["adjusted_eta"], "normal_cost": base["total_cost"],
            "delta_eta": round(cand["adjusted_eta"] - base["adjusted_eta"], 1),
            "delta_cost": round(cand["total_cost"] - base["total_cost"], 2),
            "normal_path": [self.hubs.get(h, {}).get("display_name", h) for h in base["hubs"]],
        }

    def _drivers(self, cand) -> Optional[Dict[str, Any]]:
        """Shapley attribution for the leg carrying the largest p85 delay buffer."""
        transit = [l for l in cand["legs"] if l.get("_ml_key")]
        if not transit:
            return None
        critical = max(transit, key=lambda l: l["delay_band"]["p85"] - l["event_delay"])
        exp = self.predictor.explain(critical["_ml_key"], "p85")
        if not exp:
            return None
        return {"leg": f'{critical["from_name"]} → {critical["to_name"]}', "mode": critical["mode"],
                "predicted_p85_h": critical["delay_band"]["p85"], **exp}

    def _explain_choice(self, c, all_c, disruptions) -> str:
        """Plain-language explanation computed from the actual alternatives (no invented figures)."""
        others = [o for o in all_c if o is not c]
        band = c["eta_band"]
        days = lambda h: f"{h / 24:.1f} days"
        span = lambda h: f"{h / 24:.1f} days" if h >= 48 else f"{h:.0f} hours"
        name = lambda h: self.hubs.get(h, {}).get("display_name", h)
        parts = []
        lead = {"FASTEST": "The quickest option", "SAFEST": "The most reliable option",
                "BALANCED": "The best-value option"}[c["persona"]]
        also = {"FASTEST": "quickest", "SAFEST": "most reliable", "BALANCED": "best value"}
        extra = [also[p] for p in c["personas"][1:]]
        parts.append(f"{lead}{' (and the ' + ' and '.join(extra) + ')' if extra else ''}: "
                     f"plan for {days(c['adjusted_eta'])}. It usually takes about {days(band['p50'])}, "
                     f"and rarely more than {days(band['p95'])}.")
        if others:
            cheapest = min(others, key=lambda o: o["total_cost"])
            fastest = min(others, key=lambda o: o["adjusted_eta"])
            if c["total_cost"] < cheapest["total_cost"]:
                saving = (1 - c["total_cost"] / cheapest["total_cost"]) * 100
                parts.append(f"It costs {saving:.0f}% less than the next-cheapest option "
                             f"(${c['total_cost']:,.0f} vs ${cheapest['total_cost']:,.0f}).")
            if c["adjusted_eta"] < fastest["adjusted_eta"]:
                parts.append(f"It arrives {span(fastest['adjusted_eta'] - c['adjusted_eta'])} sooner than the next-fastest option.")
            elif c["adjusted_eta"] > fastest["adjusted_eta"]:
                parts.append(f"It takes {span(c['adjusted_eta'] - fastest['adjusted_eta'])} longer than the fastest option"
                             + (f" but costs {(1 - c['total_cost'] / fastest['total_cost']) * 100:.0f}% less."
                                if fastest["total_cost"] > c["total_cost"] else "."))
        if disruptions:
            avoided = sorted(set(disruptions) - set(c["exposed_disruptions"]))
            if c["exposed_disruptions"]:
                parts.append(f"It still passes through the disruption at {', '.join(name(h) for h in c['exposed_disruptions'])}.")
            elif avoided:
                parts.append(f"It avoids the disruption at {', '.join(name(h) for h in avoided)}.")
        return " ".join(parts)
