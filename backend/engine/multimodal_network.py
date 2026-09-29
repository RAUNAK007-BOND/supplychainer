import networkx as nx
import json
import os
import math
from typing import List, Dict, Any

MODE_PROFILES = {
    "road": {"speed": 80, "cost_per_km": 1.2, "cargo_restrictions": ["oversize_heavy"]},
    "rail": {"speed": 60, "cost_per_km": 0.5, "cargo_restrictions": []},
    "air": {"speed": 800, "cost_per_km": 15.0, "cargo_restrictions": ["hazardous_waste"]},
    "sea": {"speed": 35, "cost_per_km": 0.15, "cargo_restrictions": ["perishable_urgent"]}
}

PRIORITY_MULTIPLIERS = {
    "low": 1.2,
    "normal": 1.0,
    "urgent": 0.7
}

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _travel_time(dist, mode):
    # Speeds in km/h based on freight benchmarks
    speed = MODE_PROFILES.get(mode, {}).get("speed", 50)
    
    # Road friction factor: Real trucking time includes standard stops/traffic
    if mode == "road":
        # Effective speed is lower than max highway speed due to freight regulations
        effective_speed = speed * 0.85 # 68 km/h effective
        return dist / effective_speed
    
    return dist / speed

# Country pairs with no land freight link between them (separated by sea or by a third country).
# Road/rail corridors between them — e.g. Iran<->Saudi Arabia rail across the Persian Gulf, or the
# <200km road auto-wire bridging the Strait of Gibraltar — are physically impossible.
NO_LAND_LINK = {frozenset(p) for p in [
    ("China", "South Korea"), ("China", "Taiwan"), ("Russia", "South Korea"), ("Djibouti", "Sudan"),
    ("Egypt", "Greece"), ("Iran", "Saudi Arabia"), ("Saudi Arabia", "Sudan"), ("Israel", "Turkey"),
    ("Morocco", "Spain"), ("Russia", "Turkey"), ("Azerbaijan", "Kazakhstan"), ("Georgia", "Kazakhstan"),
    ("Greece", "Italy"), ("Georgia", "Iran"), ("Thailand", "Vietnam"), ("Singapore", "Indonesia"),
    ("Malaysia", "Indonesia"), ("UAE", "Iran"), ("Oman", "Iran"), ("Qatar", "Iran"), ("Denmark", "Sweden"),
]}
LAND_MODES = ("road", "rail")


def land_link_allowed(h1, h2, mode):
    return mode not in LAND_MODES or frozenset((h1["country"], h2["country"])) not in NO_LAND_LINK


def load_canonical_hubs():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'canonical_hubs.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

TRANSFER_PROFILES = {
    "port_to_rail": {"delay": 8.0, "cost": 180, "risk": 0.05},
    "road_to_air": {"delay": 6.0, "cost": 150, "risk": 0.02},
    "road_to_sea": {"delay": 14.0, "cost": 250, "risk": 0.08},
    "road_to_rail": {"delay": 4.0, "cost": 100, "risk": 0.03},
    "rail_to_road": {"delay": 4.0, "cost": 80, "risk": 0.02},
    "default": {"delay": 4.0, "cost": 100, "risk": 0.03}
}

def create_multimodal_network():
    """
    Supplychainer Unified Multimodal Optimization Graph.
    V3: Node Splitting Edition (The Forensic Fix).
    """
    G = nx.DiGraph()
    hubs = load_canonical_hubs()
    hub_lookup = {h["id"]: h for h in hubs}

    # 1. Add Mode-Specific Virtual Nodes
    # Each hub H with modes M gets nodes H:m1, H:m2...
    for hub in hubs:
        hub_id = hub["id"]
        for mode in hub["modes"]:
            v_node = f"{hub_id}:{mode}"
            G.add_node(v_node,
                physical_id=hub_id,
                display_name=hub["display_name"], 
                type=hub["type"],
                country=hub["country"], 
                lat=hub["lat"], 
                lon=hub["lon"],
                importance=hub.get("importance", 5),
                mode=mode, 
                parent_city=hub.get("parent_city"))

    # 2. Add Intra-Hub Transfer Edges (The Friction Layer)
    for hub in hubs:
        hub_id = hub["id"]
        modes = hub["modes"]
        for i, m1 in enumerate(modes):
            for m2 in modes[i+1:]:
                u, v = f"{hub_id}:{m1}", f"{hub_id}:{m2}"
                
                # Determine transfer profile
                t_type = hub["type"]
                profile_key = "default"
                if t_type == "port" and (m1 == "rail" or m2 == "rail"): profile_key = "port_to_rail"
                elif t_type == "airport" and (m1 == "road" or m2 == "road"): profile_key = "road_to_air"
                elif t_type == "port" and (m1 == "road" or m2 == "road"): profile_key = "road_to_sea"
                
                profile = TRANSFER_PROFILES.get(profile_key, TRANSFER_PROFILES["default"])
                
                G.add_edge(u, v, baseline_time=profile["delay"], distance=0.1, 
                           transport_mode="transfer", type="transfer", 
                           cost=profile["cost"], risk=profile["risk"])
                G.add_edge(v, u, baseline_time=profile["delay"], distance=0.1, 
                           transport_mode="transfer", type="transfer", 
                           cost=profile["cost"], risk=profile["risk"])

    # 3. Add Strategic Intra-Mode Transit Edges
    # Corridors are physical sea lanes / air routes / tracks and are traversable both ways. The
    # registry lists ~700 connections on one side only, so adding u->v alone made routes
    # direction-dependent and left nodes like the Cape of Good Hope unreachable.
    for hub in hubs:
        u_base = hub["id"]
        for conn in hub.get("connections", []):
            v_base = conn["to"]
            mode = conn["mode"]
            
            u_vnode = f"{u_base}:{mode}"
            v_vnode = f"{v_base}:{mode}"
            
            if G.has_node(u_vnode) and G.has_node(v_vnode):
                h1, h2 = hub, hub_lookup[v_base]
                if not land_link_allowed(h1, h2, mode):
                    continue
                dist = _haversine(h1["lat"], h1["lon"], h2["lat"], h2["lon"])
                t = _travel_time(dist, mode)
                cost = dist * MODE_PROFILES[mode]["cost_per_km"]
                
                for a, b in ((u_vnode, v_vnode), (v_vnode, u_vnode)):
                    G.add_edge(a, b, baseline_time=t, distance=round(dist, 1),
                               transport_mode=mode, type="transit", cost=cost)

    # 4. Local Road Auto-wire (<200km)
    for i, h1 in enumerate(hubs):
        if "road" not in h1["modes"]: continue
        for h2 in hubs[i+1:]:
            if "road" not in h2["modes"]: continue
            d = _haversine(h1["lat"], h1["lon"], h2["lat"], h2["lon"])
            if d < 200 and land_link_allowed(h1, h2, "road"):
                u, v = f"{h1['id']}:road", f"{h2['id']}:road"
                if G.has_node(u) and G.has_node(v) and not G.has_edge(u, v):
                    t = _travel_time(d, "road")
                    cost = d * MODE_PROFILES["road"]["cost_per_km"]
                    G.add_edge(u, v, baseline_time=t, distance=round(d, 1), 
                               transport_mode="road", type="transit", cost=cost)
                    G.add_edge(v, u, baseline_time=t, distance=round(d, 1), 
                               transport_mode="road", type="transit", cost=cost)

    print(f"Split-Node Multimodal Network: {G.number_of_nodes()} virtual nodes, {G.number_of_edges()} edges")
    return G

def get_city_capabilities(G):
    city_data = {}
    for node, data in G.nodes(data=True):
        city = data.get("parent_city", data.get("display_name"))
        if city not in city_data:
            city_data[city] = {"id": data.get("physical_id"), "display_name": city, "country": data.get("country"), 
                               "has_port": False, "has_airport": False, "has_rail": False, "nodes": []}
        
        city_data[city]["nodes"].append(node)
        mode = data.get("mode")
        if mode == "sea": city_data[city]["has_port"] = True
        if mode == "air": city_data[city]["has_airport"] = True
        if mode == "rail": city_data[city]["has_rail"] = True

    return sorted(list(city_data.values()), key=lambda c: c["display_name"])
