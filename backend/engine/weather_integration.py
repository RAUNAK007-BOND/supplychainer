from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import random
import networkx as nx

WEATHER_SEVERITY_MAPPING = {
    "clear": 0.0,
    "clouds": 0.1,
    "mist": 0.2,
    "drizzle": 0.3,
    "fog": 0.4,
    "rain": 0.5,
    "snow": 0.6,
    "heavy_rain": 0.7,
    "thunderstorm": 0.9,
    "extreme": 1.0
}

import requests

WEATHER_SEVERITY_MAPPING = {
    "clear": 0.0,
    "clouds": 0.1,
    "mist": 0.2,
    "drizzle": 0.3,
    "fog": 0.4,
    "rain": 0.5,
    "snow": 0.6,
    "heavy_rain": 0.7,
    "thunderstorm": 0.9,
    "extreme": 1.0
}

# Mapping Open-Meteo WMO codes to severity
WMO_SEVERITY_MAPPING = {
    0: 0.0,    # Clear sky
    1: 0.1, 2: 0.2, 3: 0.3, # Mainly clear, partly cloudy, overcast
    45: 0.4, 48: 0.5, # Fog
    51: 0.3, 53: 0.4, 55: 0.5, # Drizzle
    61: 0.5, 63: 0.7, 65: 0.9, # Rain
    71: 0.6, 73: 0.8, 75: 1.0, # Snow
    80: 0.5, 81: 0.7, 82: 0.9, # Rain showers
    95: 0.9, 96: 1.0, 99: 1.0, # Thunderstorm
}

class WeatherProvider(ABC):
    @property
    @abstractmethod
    def source_confidence(self) -> float:
        """Returns the confidence level of this provider (0.0 to 1.0)."""
        pass

    @abstractmethod
    def get_weather_states(self, network: nx.DiGraph, current_states: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        """Returns updated weather states for edges."""
        pass

class DeterministicWeatherProvider(WeatherProvider):
    @property
    def source_confidence(self) -> float:
        return 0.3  # Deterministic fallback

    def get_weather_states(self, network: nx.DiGraph, current_states: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        new_states = current_states.copy()
        
        for u, v, data in network.edges(data=True):
            edge_key = (u, v)
            region = data.get("region", "Unknown")
            
            # Weather logic (causal by region)
            weather_risk = 0.05
            if region == "Pacific_Northwest":
                weather_risk = 0.3  # Rain
            elif region == "Southeast":
                weather_risk = 0.2  # Storms
            elif region == "Midwest":
                weather_risk = 0.15 # Snow/Wind
            
            # 10% chance of an event happening based on region risk
            if random.random() < weather_risk:
                new_states[edge_key] = random.uniform(0.3, 1.0)
            else:
                new_states[edge_key] = max(0.0, new_states.get(edge_key, 0.0) - 0.1) # Decay
                
        return new_states

class APIWeatherProvider(WeatherProvider):
    @property
    def source_confidence(self) -> float:
        return 0.95  # Live Open-Meteo data

    def get_weather_severity(self, lat: float, lon: float) -> float:
        """Fetches live weather from Open-Meteo and returns severity score."""
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            response = requests.get(url, timeout=5)
            data = response.json()
            if "current_weather" in data:
                code = data["current_weather"]["weathercode"]
                return WMO_SEVERITY_MAPPING.get(code, 0.0)
        except Exception as e:
            print(f"Weather API error: {e}. Falling back to deterministic model.")
            # Fallback to a random but region-consistent value
            return random.uniform(0.1, 0.4)

    def get_weather_states(self, network: nx.DiGraph, current_states: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        # For the internal graph nodes, we use the city coordinates
        from .live_routing import BENCHMARK_CITY_COORDS
        new_states = current_states.copy()
        
        # To avoid rate limits, we'll only fetch for unique regions/hubs or a subset
        unique_nodes = list(network.nodes())
        node_weather = {}
        
        # For the demo, we'll pick a few major hubs to fetch real data for
        sample_nodes = random.sample(unique_nodes, min(5, len(unique_nodes)))
        for node in sample_nodes:
            coords = BENCHMARK_CITY_COORDS.get(node)
            if coords:
                node_weather[node] = self.get_weather_severity(coords[0], coords[1])

        for u, v, data in network.edges(data=True):
            edge_key = (u, v)
            # Use node weather if available, else decay
            severity = node_weather.get(v, node_weather.get(u))
            if severity is not None:
                new_states[edge_key] = severity
            else:
                new_states[edge_key] = max(0.0, new_states.get(edge_key, 0.0) - 0.1)
                
        return new_states
