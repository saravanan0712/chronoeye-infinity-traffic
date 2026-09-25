"""
ChronoEye Infinity - Phase 11: Standard Dijkstra Router
Classic Dijkstra shortest path router operating on static current edge travel times.
"""

import heapq
import time
from typing import List, Dict, Tuple, Optional, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_schema import NodeType, EdgeType
from app.optimization.route_schema import OptimizationRoute, RouteSegment


class DijkstraRouter:
    """
    Standard Dijkstra Shortest Path Router.
    """

    def find_route(
        self,
        builder: SpatioTemporalGraphBuilder,
        origin: str,
        destination: str,
    ) -> OptimizationRoute:
        """
        Finds shortest path using standard Dijkstra algorithm.
        """
        start_t = time.perf_counter()

        if origin == destination:
            return OptimizationRoute(
                origin_junction=origin,
                destination_junction=destination,
                path_junctions=[origin],
                algorithm_name="StandardDijkstra",
            )

        # Graph adjacency traversal over JUNCTION and ROAD nodes
        # Priority Queue: (cost, current_node, path_nodes, path_roads, total_dist)
        pq: List[Tuple[float, str, List[str], List[str], float]] = [(0.0, origin, [origin], [], 0.0)]
        visited = set()

        while pq:
            cost, u, path_nodes, path_roads, dist = heapq.heappop(pq)

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
                    total_travel_time_seconds=round(cost, 2),
                    computation_latency_ms=latency,
                    algorithm_name="StandardDijkstra",
                )

            # Neighbor traversal
            for _, v, k, d in builder.graph.edges(u, data=True, keys=True):
                edge_type = d.get("edge_type")
                if edge_type == EdgeType.ROAD_CONNECTS_JUNCTION.value or edge_type == EdgeType.VEHICLE_TRANSITIONS_TO_ROAD.value or d.get("road_id"):
                    road_id = d.get("road_id", f"ROAD_{u}_{v}")
                    length_m = d.get("length_meters", 500.0)
                    t_time = d.get("travel_time", 30.0)

                    if v not in visited:
                        heapq.heappush(
                            pq,
                            (
                                cost + t_time,
                                v,
                                path_nodes + [v],
                                path_roads + [road_id],
                                dist + (length_m / 1000.0),
                            ),
                        )

        # Fallback if no route found
        latency = round((time.perf_counter() - start_t) * 1000.0, 3)
        return OptimizationRoute(
            origin_junction=origin,
            destination_junction=destination,
            path_junctions=[],
            computation_latency_ms=latency,
            algorithm_name="StandardDijkstra",
        )
