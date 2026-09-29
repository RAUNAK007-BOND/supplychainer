import sys
import random
from .graph_model import create_logistics_network
from .simulator import LogisticsSimulator
from .ml_predictor import DelayPredictor
from .optimizer import RouteOptimizer
from .baseline import BaselineRouter
from .or_baseline import OROptimizer

def run_benchmark():
    network = create_logistics_network()
    sim = LogisticsSimulator(network)
    predictor = DelayPredictor()
    baseline = BaselineRouter(network)
    
    print("Running Pre-Training Simulation (1500 ticks)...")
    nodes = list(network.nodes())
    for _ in range(1500):
        if random.random() < 0.5:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if src != dst:
                route = baseline.get_route(src, dst)
                if route:
                    sim.dispatch_truck(src, dst, route["route"])
        sim.step()
        
    df = sim.get_trip_dataframe()
    predictor.train(df)

    optimizer = RouteOptimizer(network, predictor, sim)
    or_optimizer = OROptimizer(network, predictor, sim)
    
    print("\nStarting Phase 3 Benchmarking Phase (500 Dispatches)...")
    
    benchmark_results = []
    pair_id_counter = 0
    dispatches = 0
    active_pairs = {}
    
    while dispatches < 500 or len(active_pairs) > 0:
        if dispatches < 500 and random.random() < 0.8:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if src != dst:
                dec_agg = optimizer.get_optimized_route(src, dst, strategy="aggressive")
                dec_con = optimizer.get_optimized_route(src, dst, strategy="conservative")
                dec_conf = optimizer.get_optimized_route(src, dst, strategy="confidence", alpha=0.5, beta=0.1)
                dec_or = or_optimizer.get_optimized_route(src, dst)
                
                if dec_agg and dec_con and dec_conf and dec_or:
                    pair_id = pair_id_counter
                    pair_id_counter += 1
                    
                    # Dispatch Truck A (Baseline)
                    sim.dispatch_truck(src, dst, dec_agg["baseline_route"])
                    trip_a_id = sim.active_trips[-1]["trip_id"]
                    
                    # Dispatch Truck B (Aggressive)
                    sim.dispatch_truck(src, dst, dec_agg["optimized_route"])
                    trip_b_id = sim.active_trips[-1]["trip_id"]
                    
                    # Dispatch Truck C (Conservative)
                    sim.dispatch_truck(src, dst, dec_con["optimized_route"])
                    trip_c_id = sim.active_trips[-1]["trip_id"]
                    
                    # Dispatch Truck D (Confidence)
                    sim.dispatch_truck(src, dst, dec_conf["optimized_route"])
                    trip_d_id = sim.active_trips[-1]["trip_id"]
                    
                    # Dispatch Truck E (OR-Tools Baseline)
                    sim.dispatch_truck(src, dst, dec_or["optimized_route"])
                    trip_e_id = sim.active_trips[-1]["trip_id"]
                    
                    active_pairs[pair_id] = {
                        "trip_a_id": trip_a_id,
                        "trip_b_id": trip_b_id,
                        "trip_c_id": trip_c_id,
                        "trip_d_id": trip_d_id,
                        "trip_e_id": trip_e_id,
                        "dec_agg": dec_agg,
                        "dec_con": dec_con,
                        "dec_conf": dec_conf,
                        "dec_or": dec_or,
                        "src": src,
                        "dst": dst
                    }
                    dispatches += 1
                    
        sim.step()
        
        # Check completed
        completed_ids = [t["trip_id"] for t in sim.completed_trips]
        
        pairs_to_remove = []
        for pid, pdata in active_pairs.items():
            if (pdata["trip_a_id"] in completed_ids and 
                pdata["trip_b_id"] in completed_ids and 
                pdata["trip_c_id"] in completed_ids and
                pdata["trip_d_id"] in completed_ids and
                pdata["trip_e_id"] in completed_ids):
                
                t_a = next(t for t in sim.completed_trips if t["trip_id"] == pdata["trip_a_id"])
                t_b = next(t for t in sim.completed_trips if t["trip_id"] == pdata["trip_b_id"])
                t_c = next(t for t in sim.completed_trips if t["trip_id"] == pdata["trip_c_id"])
                t_d = next(t for t in sim.completed_trips if t["trip_id"] == pdata["trip_d_id"])
                t_e = next(t for t in sim.completed_trips if t["trip_id"] == pdata["trip_e_id"])
                
                benchmark_results.append({
                    "src": pdata["src"],
                    "dst": pdata["dst"],
                    "dec_agg": pdata["dec_agg"],
                    "dec_con": pdata["dec_con"],
                    "dec_conf": pdata["dec_conf"],
                    "dec_or": pdata["dec_or"],
                    "time_a": t_a["actual_time_taken"],
                    "cost_a": pdata["dec_agg"]["baseline_cost"],
                    "time_b": t_b["actual_time_taken"],
                    "cost_b": pdata["dec_agg"]["optimized_cost"],
                    "time_c": t_c["actual_time_taken"],
                    "cost_c": pdata["dec_con"]["optimized_cost"],
                    "time_d": t_d["actual_time_taken"],
                    "cost_d": pdata["dec_conf"]["optimized_cost"],
                    "time_e": t_e["actual_time_taken"],
                    "cost_e": pdata["dec_or"]["optimized_cost"]
                })
                pairs_to_remove.append(pid)
                
        for pid in pairs_to_remove:
            del active_pairs[pid]

    # Calculate KPIs
    total_trips = len(benchmark_results)
    
    # We will analyze all cases where ANY strategy intervened
    intervention_cases = [r for r in benchmark_results if r["dec_agg"]["is_rerouted"] or r["dec_con"]["is_rerouted"] or r["dec_conf"]["is_rerouted"] or r["dec_or"]["is_rerouted"]]
    
    if not intervention_cases:
        print("No interventions triggered across 500 trips.")
        return

    def compute_stats(times_x, costs_x, strategy_key):
        times_a = [r["time_a"] for r in intervention_cases]
        costs_a = [r["cost_a"] for r in intervention_cases]
        
        avg_a = sum(times_a) / len(times_a)
        avg_x = sum(times_x) / len(times_x)
        delay_red = ((avg_a - avg_x) / avg_a) * 100 if avg_a > 0 else 0
        
        success = sum(1 for i, tx in enumerate(times_x) if tx < times_a[i] and intervention_cases[i][strategy_key]["is_rerouted"])
        attempts = sum(1 for r in intervention_cases if r[strategy_key]["is_rerouted"])
        success_rate = (success / attempts * 100) if attempts > 0 else 0
        
        worse_cases = sum(1 for i, tx in enumerate(times_x) if tx > times_a[i] and intervention_cases[i][strategy_key]["is_rerouted"])
        cost_inc = ((sum(costs_x) - sum(costs_a)) / sum(costs_a)) * 100 if sum(costs_a) > 0 else 0
        
        return attempts, delay_red, success_rate, worse_cases, cost_inc

    agg_stats = compute_stats([r["time_b"] for r in intervention_cases], [r["cost_b"] for r in intervention_cases], "dec_agg")
    con_stats = compute_stats([r["time_c"] for r in intervention_cases], [r["cost_c"] for r in intervention_cases], "dec_con")
    cnf_stats = compute_stats([r["time_d"] for r in intervention_cases], [r["cost_d"] for r in intervention_cases], "dec_conf")
    or_stats = compute_stats([r["time_e"] for r in intervention_cases], [r["cost_e"] for r in intervention_cases], "dec_or")
    
    print(f"\n--- 5-WAY PERFORMANCE EVALUATION (OVER {len(intervention_cases)} INTERVENTIONS) ---")
    
    print("\n[B] AGGRESSIVE OPTIMIZER")
    print(f"Reroutes: {agg_stats[0]}")
    print(f"Net Delay Reduction: {agg_stats[1]:.1f}%")
    print(f"Success Rate: {agg_stats[2]:.1f}%")
    print(f"Worse than baseline: {agg_stats[3]}")
    print(f"Avg Cost Increase: {agg_stats[4]:.1f}%")

    print("\n[C] CONSERVATIVE THRESHOLD OPTIMIZER")
    print(f"Reroutes: {con_stats[0]}")
    print(f"Net Delay Reduction: {con_stats[1]:.1f}%")
    print(f"Success Rate: {con_stats[2]:.1f}%")
    print(f"Worse than baseline: {con_stats[3]}")
    print(f"Avg Cost Increase: {con_stats[4]:.1f}%")

    print("\n[D] CONFIDENCE-WEIGHTED OPTIMIZER (DIJKSTRA)")
    print(f"Reroutes: {cnf_stats[0]}")
    print(f"Net Delay Reduction: {cnf_stats[1]:.1f}%")
    print(f"Success Rate: {cnf_stats[2]:.1f}%")
    print(f"Worse than baseline: {cnf_stats[3]}")
    print(f"Avg Cost Increase: {cnf_stats[4]:.1f}%")

    print("\n[E] OR-TOOLS BASELINE")
    print(f"Reroutes: {or_stats[0]}")
    print(f"Net Delay Reduction: {or_stats[1]:.1f}%")
    print(f"Success Rate: {or_stats[2]:.1f}%")
    print(f"Worse than baseline: {or_stats[3]}")
    print(f"Avg Cost Increase: {or_stats[4]:.1f}%")
    
    # Find a specific failure case for Confidence
    conf_failures = [r for r in intervention_cases if r["time_d"] > r["time_a"] and r["dec_conf"]["is_rerouted"]]
    if conf_failures:
        worst = max(conf_failures, key=lambda x: x["time_d"] - x["time_a"])
        print("\n--- FAILURE LOG: CONFIDENCE-WEIGHTED ---")
        print(f"Route: {worst['src']} -> {worst['dst']}")
        print(f"Justification: {worst['dec_conf']['justification']}")
        print(f"Baseline Time: {worst['time_a']:.1f} hrs")
        print(f"Confidence Time: {worst['time_d']:.1f} hrs")
        print(f"Difference: +{worst['time_d'] - worst['time_a']:.1f} hrs")

if __name__ == "__main__":
    run_benchmark()
