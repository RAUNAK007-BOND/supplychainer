import networkx as nx
from typing import Dict, Any
from .ml_predictor import DelayPredictor
from .baseline import BaselineRouter

class RouteOptimizer:
    def __init__(self, network: nx.DiGraph, predictor: DelayPredictor, simulator_state: Any):
        self.network = network
        self.predictor = predictor
        self.simulator = simulator_state
        self.baseline_router = BaselineRouter(network)

    def get_optimized_route(self, source: str, destination: str, strategy: str = "aggressive", alpha: float = 1.5, beta: float = 0.5) -> Dict[str, Any]:
        """
        Calculates the optimized route based on ML predictions.
        strategy: 'aggressive', 'conservative', or 'confidence'
        """
        baseline_result = self.baseline_router.get_route(source, destination)
        if not baseline_result:
            return None
            
        def weight_aggressive(u, v, d):
            edge_key = (u, v)
            w = self.simulator.weather_states.get(edge_key, 0.0)
            t = self.simulator.traffic_states.get(edge_key, 0.0)
            b = self.simulator.backlog_states.get(v, 0.0)
            prob_delay = self.predictor.predict_delay_probability(w, t, b, d["baseline_time"])
            return d["baseline_time"] + (prob_delay * d["baseline_time"] * 0.5)

        def weight_confidence(u, v, d):
            edge_key = (u, v)
            w = self.simulator.weather_states.get(edge_key, 0.0)
            t = self.simulator.traffic_states.get(edge_key, 0.0)
            b = self.simulator.backlog_states.get(v, 0.0)
            prob_delay = self.predictor.predict_delay_probability(w, t, b, d["baseline_time"])
            
            risk_penalty = alpha * (prob_delay ** 2) * d["baseline_time"]
            unc_penalty = beta * prob_delay * (1.0 - prob_delay) * d["baseline_time"]
            
            return d["baseline_time"] + risk_penalty + unc_penalty

        try:
            # We explicitly mention Dijkstra here for pathfinding terminology rigor
            weight_func = weight_confidence if strategy == "confidence" else weight_aggressive
            opt_route = nx.shortest_path(self.network, source=source, target=destination, weight=weight_func)
            
            opt_time = 0.0
            opt_cost = 0.0
            avg_prob_delay = 0.0
            opt_penalty = 0.0
            
            for i in range(len(opt_route)-1):
                u = opt_route[i]
                v = opt_route[i+1]
                d = self.network.edges[u, v]
                
                opt_time += d["baseline_time"]
                opt_cost += d["baseline_cost"]
                
                edge_key = (u, v)
                w = self.simulator.weather_states.get(edge_key, 0.0)
                t = self.simulator.traffic_states.get(edge_key, 0.0)
                b = self.simulator.backlog_states.get(v, 0.0)
                
                p = self.predictor.predict_delay_probability(w, t, b, d["baseline_time"])
                avg_prob_delay += p
                
                if strategy == "confidence":
                    r_pen = alpha * (p ** 2) * d["baseline_time"]
                    u_pen = beta * p * (1.0 - p) * d["baseline_time"]
                    opt_penalty += (r_pen + u_pen)
                else:
                    opt_penalty += p * (d["baseline_time"] * 0.5)
                
            avg_prob_delay /= max(1, len(opt_route) - 1)
            
            is_rerouted = (opt_route != baseline_result["route"])
            justification = "Standard route optimal."
            
            # Predict baseline route penalty
            baseline_penalty = 0.0
            for i in range(len(baseline_result["route"])-1):
                u = baseline_result["route"][i]
                v = baseline_result["route"][i+1]
                d = self.network.edges[u, v]
                edge_key = (u, v)
                w = self.simulator.weather_states.get(edge_key, 0.0)
                t = self.simulator.traffic_states.get(edge_key, 0.0)
                b = self.simulator.backlog_states.get(v, 0.0)
                p = self.predictor.predict_delay_probability(w, t, b, d["baseline_time"])
                
                if strategy == "confidence":
                    r_pen = alpha * (p ** 2) * d["baseline_time"]
                    u_pen = beta * p * (1.0 - p) * d["baseline_time"]
                    baseline_penalty += (r_pen + u_pen)
                else:
                    baseline_penalty += p * (d["baseline_time"] * 0.5)

            if is_rerouted:
                if strategy == "conservative":
                    b_time = baseline_result["expected_time"]
                    b_total_expected = b_time + baseline_penalty
                    
                    max_detour = min(b_time * 0.15, 6.0)
                    time_added = opt_time - b_time
                    
                    opt_total_expected = opt_time + opt_penalty
                    expected_improvement = b_total_expected - opt_total_expected
                    min_improvement = max(2.0, b_total_expected * 0.08)
                    
                    predicted_delay_cost = baseline_penalty * 100 
                    detour_cost = opt_cost - baseline_result["expected_cost"]
                    
                    if time_added > max_detour:
                        is_rerouted = False
                        justification = f"Reroute aborted: Detour cap exceeded (+{time_added:.1f} hrs vs max {max_detour:.1f} hrs)."
                    elif expected_improvement < min_improvement:
                        is_rerouted = False
                        justification = f"Reroute aborted: Insufficient expected improvement ({expected_improvement:.1f} hrs vs min {min_improvement:.1f} hrs)."
                    elif predicted_delay_cost <= (detour_cost * 1.2):
                        is_rerouted = False
                        justification = f"Reroute aborted: Poor financial tradeoff (Delay Cost ${predicted_delay_cost:.0f} vs Detour ${detour_cost:.0f}*1.2)."

                if is_rerouted:
                    time_diff = baseline_result["expected_time"] - opt_time
                    cost_diff = opt_cost - baseline_result["expected_cost"]
                    algo = "Dijkstra Confidence-Scoring" if strategy == "confidence" else "Threshold-Based"
                    justification = (f"Rerouted via {algo}. "
                                     f"ETA trade-off: {time_diff:+.1f} hrs, "
                                     f"Cost trade-off: ${cost_diff:+.0f}.")

            if not is_rerouted:
                opt_route = baseline_result["route"]
                opt_time = baseline_result["expected_time"]
                opt_cost = baseline_result["expected_cost"]

            return {
                "baseline_route": baseline_result["route"],
                "optimized_route": opt_route,
                "is_rerouted": is_rerouted,
                "baseline_time": baseline_result["expected_time"],
                "baseline_cost": baseline_result["expected_cost"],
                "optimized_time": opt_time,
                "optimized_cost": opt_cost,
                "avg_risk_probability": avg_prob_delay,
                "justification": justification
            }
        except nx.NetworkXNoPath:
            return None
