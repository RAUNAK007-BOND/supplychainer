import networkx as nx

def create_logistics_network():
    """
    Creates a baseline directed graph representing the truck logistics network.
    Each node represents a city/warehouse.
    Each edge represents a route with a baseline transit time (hours) and cost.
    Edges also have a 'region' property which makes them susceptible to specific causal disruptions.
    """
    G = nx.DiGraph()

    # Define nodes with region data (for causal simulation like weather zones)
    nodes = [
        ("Seattle", {"region": "Pacific_Northwest", "warehouse_capacity": 100}),
        ("Portland", {"region": "Pacific_Northwest", "warehouse_capacity": 80}),
        ("San Francisco", {"region": "West_Coast", "warehouse_capacity": 150}),
        ("Los Angeles", {"region": "West_Coast", "warehouse_capacity": 200}),
        ("Salt Lake City", {"region": "Mountain", "warehouse_capacity": 90}),
        ("Denver", {"region": "Mountain", "warehouse_capacity": 110}),
        ("Phoenix", {"region": "Southwest", "warehouse_capacity": 120}),
        ("Dallas", {"region": "South", "warehouse_capacity": 180}),
        ("Houston", {"region": "South", "warehouse_capacity": 160}),
        ("Chicago", {"region": "Midwest", "warehouse_capacity": 250}),
        ("St. Louis", {"region": "Midwest", "warehouse_capacity": 100}),
        ("Atlanta", {"region": "Southeast", "warehouse_capacity": 140}),
        ("Miami", {"region": "Southeast", "warehouse_capacity": 90}),
        ("New York", {"region": "Northeast", "warehouse_capacity": 220}),
        ("Boston", {"region": "Northeast", "warehouse_capacity": 100}),
    ]
    G.add_nodes_from(nodes)

    # Define edges (baseline_time in hours, baseline_cost in dollars, region)
    # The 'region' on the edge dictates what weather/traffic it experiences
    edges = [
        ("Seattle", "Portland", {"baseline_time": 3.0, "baseline_cost": 300, "region": "Pacific_Northwest"}),
        ("Portland", "San Francisco", {"baseline_time": 10.0, "baseline_cost": 900, "region": "West_Coast"}),
        ("San Francisco", "Los Angeles", {"baseline_time": 6.0, "baseline_cost": 500, "region": "West_Coast"}),
        ("Los Angeles", "Phoenix", {"baseline_time": 5.5, "baseline_cost": 450, "region": "Southwest"}),
        ("Portland", "Salt Lake City", {"baseline_time": 11.5, "baseline_cost": 1000, "region": "Mountain"}),
        ("San Francisco", "Salt Lake City", {"baseline_time": 11.0, "baseline_cost": 950, "region": "Mountain"}),
        ("Salt Lake City", "Denver", {"baseline_time": 8.0, "baseline_cost": 700, "region": "Mountain"}),
        ("Los Angeles", "Salt Lake City", {"baseline_time": 10.5, "baseline_cost": 900, "region": "Mountain"}),
        ("Phoenix", "Dallas", {"baseline_time": 14.5, "baseline_cost": 1200, "region": "South"}),
        ("Denver", "Dallas", {"baseline_time": 12.0, "baseline_cost": 1000, "region": "South"}),
        ("Denver", "Chicago", {"baseline_time": 14.5, "baseline_cost": 1300, "region": "Midwest"}),
        ("Dallas", "Houston", {"baseline_time": 3.5, "baseline_cost": 350, "region": "South"}),
        ("Dallas", "St. Louis", {"baseline_time": 9.5, "baseline_cost": 850, "region": "Midwest"}),
        ("Houston", "Atlanta", {"baseline_time": 11.5, "baseline_cost": 1000, "region": "Southeast"}),
        ("St. Louis", "Chicago", {"baseline_time": 4.5, "baseline_cost": 400, "region": "Midwest"}),
        ("St. Louis", "Atlanta", {"baseline_time": 8.5, "baseline_cost": 750, "region": "Midwest_Southeast"}),
        ("Chicago", "New York", {"baseline_time": 12.0, "baseline_cost": 1100, "region": "Northeast"}),
        ("Atlanta", "New York", {"baseline_time": 13.0, "baseline_cost": 1150, "region": "East_Coast"}),
        ("Atlanta", "Miami", {"baseline_time": 9.5, "baseline_cost": 850, "region": "Southeast"}),
        ("New York", "Boston", {"baseline_time": 4.0, "baseline_cost": 350, "region": "Northeast"}),
    ]
    
    # Since it's a truck network, roads go both ways (symmetric costs for simplicity)
    bidirectional_edges = []
    for u, v, data in edges:
        bidirectional_edges.append((u, v, data))
        bidirectional_edges.append((v, u, data.copy()))
        
    G.add_edges_from(bidirectional_edges)
    
    return G

if __name__ == "__main__":
    network = create_logistics_network()
    print(f"Created Logistics Network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges.")
