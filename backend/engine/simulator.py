import random
import pandas as pd
from typing import List, Dict
import networkx as nx
from .graph_model import create_logistics_network
from .weather_integration import WeatherProvider, DeterministicWeatherProvider

class LogisticsSimulator:
    def __init__(self, network: nx.DiGraph, weather_provider: 'WeatherProvider' = None):
        self.network = network
        self.weather_provider = weather_provider or DeterministicWeatherProvider()
        self.time_tick = 0  # 1 tick = 1 hour
        
        # Environmental state
        self.weather_states = {}  # edge -> weather severity (0.0 to 1.0)
        self.traffic_states = {}  # edge -> traffic severity (0.0 to 1.0)
        self.backlog_states = {}  # node -> backlog severity (0.0 to 1.0)
        
        # Active trips: list of dicts
        self.active_trips = []
        
        # Completed trips log (for ML training)
        self.completed_trips = []

    def _update_environment(self):
        """Update causal environmental factors based on time of day and regions."""
        hour_of_day = self.time_tick % 24
        is_rush_hour = hour_of_day in [7, 8, 9, 16, 17, 18]
        
        # Update Node Backlogs
        for node, data in self.network.nodes(data=True):
            # Nodes with higher capacity handle backlog better, but still fluctuate
            base_backlog = random.uniform(0.0, 0.2)
            # Simulated incoming rush increases backlog
            if is_rush_hour:
                base_backlog += random.uniform(0.1, 0.3)
            self.backlog_states[node] = min(1.0, base_backlog)

        # Update Edge Weather
        self.weather_states = self.weather_provider.get_weather_states(self.network, self.weather_states)

        # Update Traffic
        for u, v, data in self.network.edges(data=True):
            edge_key = (u, v)

            # Traffic logic (causal by time of day)
            traffic_severity = random.uniform(0.0, 0.2)
            if is_rush_hour:
                traffic_severity += random.uniform(0.4, 0.8)
            self.traffic_states[edge_key] = min(1.0, traffic_severity)

    def dispatch_truck(self, source: str, destination: str, route: List[str]):
        """Dispatches a truck along a specified route."""
        # Calculate expected baseline ETA
        expected_eta = 0.0
        for i in range(len(route)-1):
            u = route[i]
            v = route[i+1]
            expected_eta += self.network.edges[u, v]["baseline_time"]
            
        trip = {
            "trip_id": len(self.active_trips) + len(self.completed_trips) + 1,
            "source": source,
            "destination": destination,
            "route": route,
            "current_node_idx": 0,
            "progress_on_edge": 0.0, # hours traveled on current edge
            "expected_eta": expected_eta,
            "actual_time_taken": 0.0,
            "features_encountered": {
                "avg_weather": 0.0,
                "avg_traffic": 0.0,
                "avg_backlog": 0.0,
                "ticks_recorded": 0
            }
        }
        self.active_trips.append(trip)

    def step(self):
        """Advance simulation by 1 hour (tick)."""
        self._update_environment()
        
        completed_this_tick = []
        for trip in self.active_trips:
            # Advance trip
            trip["actual_time_taken"] += 1.0
            
            u = trip["route"][trip["current_node_idx"]]
            if trip["current_node_idx"] == len(trip["route"]) - 1:
                # Already at destination
                completed_this_tick.append(trip)
                continue
                
            v = trip["route"][trip["current_node_idx"] + 1]
            edge_key = (u, v)
            
            # Experience environment
            w_severity = self.weather_states.get(edge_key, 0.0)
            t_severity = self.traffic_states.get(edge_key, 0.0)
            b_severity = self.backlog_states.get(v, 0.0) # Backlog at target node
            
            # Record features encountered for ML training later
            trip["features_encountered"]["avg_weather"] += w_severity
            trip["features_encountered"]["avg_traffic"] += t_severity
            trip["features_encountered"]["avg_backlog"] += b_severity
            trip["features_encountered"]["ticks_recorded"] += 1
            
            # Calculate speed penalty. Max penalty means 1 hour of driving only covers 0.2 hours of baseline distance.
            # Base speed = 1.0. With severe disruptions, speed drops.
            speed_multiplier = 1.0 - (w_severity * 0.3) - (t_severity * 0.4) - (b_severity * 0.2)
            speed_multiplier = max(0.1, speed_multiplier) # Never fully stopped
            
            effective_progress = 1.0 * speed_multiplier
            trip["progress_on_edge"] += effective_progress
            
            edge_baseline_time = self.network.edges[u, v]["baseline_time"]
            if trip["progress_on_edge"] >= edge_baseline_time:
                # Reached next node
                trip["current_node_idx"] += 1
                trip["progress_on_edge"] = 0.0
                
                if trip["current_node_idx"] == len(trip["route"]) - 1:
                    completed_this_tick.append(trip)

        # Cleanup completed
        for trip in completed_this_tick:
            # Calculate averages for ML features
            ticks = trip["features_encountered"]["ticks_recorded"]
            if ticks > 0:
                trip["features_encountered"]["avg_weather"] /= ticks
                trip["features_encountered"]["avg_traffic"] /= ticks
                trip["features_encountered"]["avg_backlog"] /= ticks
                
            # Assign label: delayed if actual > expected + 20% + 1 hour grace period
            delay_threshold = (trip["expected_eta"] * 1.2) + 1.0 
            trip["is_delayed"] = 1 if trip["actual_time_taken"] > delay_threshold else 0
            
            self.completed_trips.append(trip)
            self.active_trips.remove(trip)
            
        self.time_tick += 1

    def get_trip_dataframe(self):
        """Returns completed trips as a pandas DataFrame for ML training."""
        data = []
        for trip in self.completed_trips:
            data.append({
                "trip_id": trip["trip_id"],
                "source": trip["source"],
                "destination": trip["destination"],
                "expected_eta": trip["expected_eta"],
                "actual_time_taken": trip["actual_time_taken"],
                "avg_weather": trip["features_encountered"]["avg_weather"],
                "avg_traffic": trip["features_encountered"]["avg_traffic"],
                "avg_backlog": trip["features_encountered"]["avg_backlog"],
                "is_delayed": trip["is_delayed"]
            })
        return pd.DataFrame(data)

if __name__ == "__main__":
    net = create_logistics_network()
    sim = LogisticsSimulator(net)
    
    # Simple test
    sim.dispatch_truck("Seattle", "Los Angeles", ["Seattle", "Portland", "San Francisco", "Los Angeles"])
    while len(sim.active_trips) > 0:
        sim.step()
    
    df = sim.get_trip_dataframe()
    print(df)
