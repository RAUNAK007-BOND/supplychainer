import networkx as nx

class BaselineRouter:
    def __init__(self, network: nx.DiGraph):
        self.network = network

    def get_route(self, source: str, destination: str):
        """
        Calculates the naive shortest path based solely on baseline_time.
        This represents standard routing without dynamic risk awareness.
        """
        try:
            route = nx.shortest_path(self.network, source=source, target=destination, weight="baseline_time")
            
            expected_time = 0.0
            expected_cost = 0.0
            
            for i in range(len(route)-1):
                u = route[i]
                v = route[i+1]
                edge_data = self.network.edges[u,v]
                expected_time += edge_data["baseline_time"]
                expected_cost += edge_data["baseline_cost"]
                
            return {
                "route": route,
                "expected_time": expected_time,
                "expected_cost": expected_cost
            }
        except nx.NetworkXNoPath:
            return None

if __name__ == "__main__":
    from .graph_model import create_logistics_network
    net = create_logistics_network()
    router = BaselineRouter(net)
    res = router.get_route("Seattle", "Miami")
    print(res)
