"""
ChronoEye Infinity - Phase 11: Standard A* Router
Classic A* shortest path router with spatial Euclidean distance heuristic function h(n).
"""

import heapq
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_schema import EdgeType
from app.optimization.route_schema import OptimizationRoute


class AStarRouter:
    """
    Standard A* Shortest Path Router with Euclidean Heuristic.
    """

    def _heuristic(self, node_a: str, node_b: str, builder: SpatioTemporalGraphBuilder) -> float:
        """
        Calculates admissible spatial heuristic h(n) in estimated seconds.
        """
        na = builder.graph.nodes.get(node_a, {}).get("data", {})
        nb = builder.graph.nodes.get(node_b, {}).get("data", {})

        lat1, lon1 = na.get("latitude", 13.0827), na.get("longitude", 80.2707)
        lat2, lon2 = nb.get("latitude", 13.0827), nb.get("longitude", 80.2707)

        # Haversine distance in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist_m = 6371000.0 * c

        # Estimated minimum travel time assuming free flow (60 km/h = 16.67 m/s)
        return dist_m / 16.67

    def find_route(
        self,
        builder: SpatioTemporalGraphBuilder,
        origin: str,
        destination: str,
    ) -> OptimizationRoute:
        """
        Finds shortest path using standard A* algorithm.
        f(n) = g(n) + h(n)
        """
        start_t = time.perf_counter()

        if origin == destination:
            return OptimizationRoute(
                origin_junction=origin,
                destination_junction=destination,
                path_junctions=[origin],
                algorithm_name="StandardAStar",
            )

        # Priority Queue: (f_score, g_cost, current_node, path_nodes, path_roads, total_dist)
        h_start = self._heuristic(origin, destination, builder)
        pq: List[Tuple[float, float, str, List[str], List[str], float]] = [
            (h_start, 0.0, origin, [origin], [], 0.0)
        ]
        visited = set()

        while pq:
            f, g, u, path_nodes, path_roads, dist = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == destination:
                latency = round((time.perf_counter() - start_t) * 1000.0, 3)
                return OptimizationRoute(
                    origin_junction=origin,
                    destination_junction=destination,
                    path_junctions=path_nodes,
                    path_roads=path_roads,
                    total_distance_km=round(dist, 2),
                    total_travel_time_seconds=round(g, 2),
                    computation_latency_ms=latency,
                    algorithm_name="StandardAStar",
                )

            for _, v, k, d in builder.graph.edges(u, data=True, keys=True):
                edge_type = d.get("edge_type")
                if edge_type == EdgeType.ROAD_CONNECTS_JUNCTION.value or d.get("road_id"):
                    road_id = d.get("road_id", f"ROAD_{u}_{v}")
                    length_m = d.get("length_meters", 500.0)
                    t_time = d.get("travel_time", 30.0)

                    if v not in visited:
                        new_g = g + t_time
                        h_val = self._heuristic(v, destination, builder)
                        heapq.heappush(
                            pq,
                            (
                                new_g + h_val,
                                new_g,
                                v,
                                path_nodes + [v],
                                path_roads + [road_id],
                                dist + (length_m / 1000.0),
                            ),
                        )

        latency = round((time.perf_counter() - start_t) * 1000.0, 3)
        return OptimizationRoute(
            origin_junction=origin,
            destination_junction=destination,
            path_junctions=[],
            computation_latency_ms=latency,
            algorithm_name="StandardAStar",
        )
