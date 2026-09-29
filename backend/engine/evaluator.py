from .graph_model import create_logistics_network
from .simulator import LogisticsSimulator
from .ml_predictor import DelayPredictor
from .optimizer import RouteOptimizer
from .baseline import BaselineRouter
import random

class Benchmarker:
    def __init__(self):
        self.network = create_logistics_network()
        self.simulator = LogisticsSimulator(self.network)
        self.predictor = DelayPredictor()
        self.baseline = BaselineRouter(self.network)
        
    def run_pre_training(self):
        """Run simulation randomly to gather data for ML."""
        print("Running pre-training simulation to generate emergent outcomes...")
        nodes = list(self.network.nodes())
        
        for _ in range(500): # 500 ticks
            # Dispatch random truck 30% of the time
            if random.random() < 0.3:
                src = random.choice(nodes)
                dst = random.choice(nodes)
                if src != dst:
                    route = self.baseline.get_route(src, dst)
                    if route:
                        self.simulator.dispatch_truck(src, dst, route["route"])
            self.simulator.step()
            
        df = self.simulator.get_trip_dataframe()
        print(f"Generated {len(df)} trips for training.")
        self.predictor.train(df)

    def evaluate_optimization(self):
        """Run benchmark comparing Baseline vs Optimizer."""
        print("\nStarting Evaluation Benchmark...")
        optimizer = RouteOptimizer(self.network, self.predictor, self.simulator)
        
        total_eval_trips = 0
        reroutes = 0
        
        nodes = list(self.network.nodes())
        for _ in range(200): # 200 ticks evaluation
            if random.random() < 0.5:
                src = random.choice(nodes)
                dst = random.choice(nodes)
                if src != dst:
                    decision = optimizer.get_optimized_route(src, dst)
                    if decision:
                        total_eval_trips += 1
                        if decision["is_rerouted"]:
                            reroutes += 1
                        self.simulator.dispatch_truck(src, dst, decision["optimized_route"])
            self.simulator.step()
            
        df = self.simulator.get_trip_dataframe()
        kpis = {
            "Total Evaluation Trips": total_eval_trips,
            "Reroute Frequency %": (reroutes / max(1, total_eval_trips)) * 100,
            "Prediction Model Trained": self.predictor.is_trained
        }
        return kpis

if __name__ == "__main__":
    benchmarker = Benchmarker()
    benchmarker.run_pre_training()
    kpis = benchmarker.evaluate_optimization()
    print("\n=== Benchmark KPIs ===")
    for k, v in kpis.items():
        print(f"{k}: {v}")
