import requests
import time
from typing import Dict, Tuple, Optional, List

# In-memory geocode cache
GEOCODE_CACHE = {}

# Coordinates for benchmark cities to allow hub detection and fallback
BENCHMARK_CITY_COORDS = {
    "Seattle": (47.6062, -122.3321),
    "Portland": (45.5152, -122.6784),
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "Salt Lake City": (40.7608, -111.8910),
    "Denver": (39.7392, -104.9903),
    "Phoenix": (33.4484, -112.0740),
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Chicago": (41.8781, -87.6298),
    "St. Louis": (38.6270, -90.1994),
    "Atlanta": (33.7490, -84.3880),
    "Miami": (25.7617, -80.1918),
    "New York": (40.7128, -74.0060),
    "Boston": (42.3601, -71.0589),
    # Global Hubs
    "Mumbai": (18.9750, 72.8258),
    "Kochi": (9.9312, 76.2673),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Singapore": (1.3521, 103.8198),
    "Shanghai": (31.2304, 121.4737),
    "Rotterdam": (51.9225, 4.4792),
    "Dubai": (25.2048, 55.2708),
    "Suez Canal": (30.5852, 32.2654)
}

def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Geocodes a location name using Nominatim.
    Uses in-memory cache and includes timeout protection.
    """
    if location_name in BENCHMARK_CITY_COORDS:
        return BENCHMARK_CITY_COORDS[location_name]
        
    if location_name in GEOCODE_CACHE:
        return GEOCODE_CACHE[location_name]

    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
        headers = {'User-Agent': 'SmartSupplyChainDemo/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        if data:
            coords = (float(data[0]["lat"]), float(data[0]["lon"]))
            GEOCODE_CACHE[location_name] = coords
            return coords
    except Exception as e:
        print(f"Geocoding error for {location_name}: {e}")
        
    return None

def get_route_between(source_coords: Tuple[float, float], destination_coords: Tuple[float, float]) -> Optional[Dict]:
    """
    Fetches road routing data using OSRM public API.
    """
    try:
        # OSRM format: lon,lat;lon,lat
        url = f"http://router.project-osrm.org/route/v1/driving/{source_coords[1]},{source_coords[0]};{destination_coords[1]},{destination_coords[0]}?overview=false"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            return {
                "distance_km": route["distance"] / 1000.0,
                "duration_hours": route["duration"] / 3600.0,
                "route_source": "live_api"
            }
    except Exception as e:
        print(f"Routing error: {e}. Falling back to Haversine estimate.")
        dist = calculate_haversine_distance(source_coords, destination_coords)
        return {
            "distance_km": dist * 1.3, # Road distance is ~30% longer than air
            "duration_hours": (dist * 1.3) / 75.0, # Average 75km/h
            "route_source": "fallback_haversine"
        }

def calculate_haversine_distance(coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
    """Calculates the great-circle distance between two points in km."""
    import math
    lat1, lon1 = coords1
    lat2, lon2 = coords2
    R = 6371  # Earth radius in km
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c
