import hashlib
import networkx as nx
from typing import Dict, Any
from ortools.graph.python import min_cost_flow
from .ml_predictor import DelayPredictor
from .baseline import BaselineRouter

def _get_volatility(region_str: str) -> float:
    return int(hashlib.md5(region_str.encode()).hexdigest(), 16) % 100 / 100.0

class OROptimizer:
    def __init__(self, network: nx.DiGraph, predictor: DelayPredictor, simulator_state: Any):
        self.network = network
        self.predictor = predictor
        self.simulator = simulator_state
        self.baseline_router = BaselineRouter(network)

    def get_optimized_route(self, source: str, destination: str) -> Dict[str, Any]:
        baseline_result = self.baseline_router.get_route(source, destination)
        if not baseline_result:
            return None

        # Build OR-Tools SimpleMinCostFlow model
        smcf = min_cost_flow.SimpleMinCostFlow()

        nodes_list = list(self.network.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        idx_to_node = {i: n for i, n in enumerate(nodes_list)}

        for u, v, data in self.network.edges(data=True):
            edge_key = (u, v)
            w = self.simulator.weather_states.get(edge_key, 0.0)
            t = self.simulator.traffic_states.get(edge_key, 0.0)
            b = self.simulator.backlog_states.get(v, 0.0)
            
            # Deterministic classical simulator state (current environment)
            speed_multiplier = 1.0 - (w * 0.3) - (t * 0.4) - (b * 0.2)
            speed_multiplier = max(0.1, speed_multiplier)
            expected_time = data["baseline_time"] / speed_multiplier
            
            # 1. Transport Cost: base route cost + $25/hr
            transport_cost = data["baseline_cost"] + (expected_time * 25.0)
            
            # 2. Expected Delay Cost (ML-guided risk)
            expected_delay = max(0.0, expected_time - data["baseline_time"])
            delay_penalty = expected_delay * 100.0
            
            delay_probability = self.predictor.predict_delay_probability(w, t, b, data["baseline_time"])
            expected_delay_cost = delay_probability * delay_penalty
            
            # 3. Route Stability Penalty
            region_str = data.get("region", "Unknown")
            historical_volatility_score = _get_volatility(region_str)
            coefficient = 200.0
            stability_penalty = historical_volatility_score * coefficient
            
            total_cost = transport_cost + expected_delay_cost + stability_penalty
            
            # OR-Tools requires integer costs
            int_cost = int(total_cost * 100)
            
            # Add arc: capacity 1 since we are routing a single truck
            smcf.add_arc_with_capacity_and_unit_cost(
                node_to_idx[u], node_to_idx[v], 1, int_cost
            )

        # Set node supplies: Source = +1, Dest = -1
        for node in nodes_list:
            if node == source:
                smcf.set_node_supply(node_to_idx[node], 1)
            elif node == destination:
                smcf.set_node_supply(node_to_idx[node], -1)
            else:
                smcf.set_node_supply(node_to_idx[node], 0)

        status = smcf.solve()

        if status == smcf.OPTIMAL:
            # Reconstruct route
            opt_route = []
            curr_node = node_to_idx[source]
            opt_route.append(source)
            
            while curr_node != node_to_idx[destination]:
                for i in range(smcf.num_arcs()):
                    if smcf.tail(i) == curr_node and smcf.flow(i) > 0:
                        curr_node = smcf.head(i)
                        opt_route.append(idx_to_node[curr_node])
                        break
                        
            opt_time = 0.0
            opt_cost = 0.0
            for i in range(len(opt_route)-1):
                u = opt_route[i]
                v = opt_route[i+1]
                d = self.network.edges[u, v]
                opt_time += d["baseline_time"]
                opt_cost += d["baseline_cost"]
                
            is_rerouted = (opt_route != baseline_result["route"])
            
            justification = "Standard route optimal."
            if is_rerouted:
                time_diff = baseline_result["expected_time"] - opt_time
                cost_diff = opt_cost - baseline_result["expected_cost"]
                justification = (f"Rerouted via OR-Tools Baseline. "
                                 f"ETA trade-off: {time_diff:+.1f} hrs, "
                                 f"Cost trade-off: ${cost_diff:+.0f}.")

            return {
                "baseline_route": baseline_result["route"],
                "optimized_route": opt_route,
                "is_rerouted": is_rerouted,
                "baseline_time": baseline_result["expected_time"],
                "baseline_cost": baseline_result["expected_cost"],
                "optimized_time": opt_time,
                "optimized_cost": opt_cost,
                "justification": justification
            }
        else:
            return None
