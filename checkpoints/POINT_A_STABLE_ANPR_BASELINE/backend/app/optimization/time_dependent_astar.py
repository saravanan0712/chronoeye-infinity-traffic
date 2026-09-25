"""
ChronoEye Infinity - Phase 11: Time-Dependent A* Router
Time-dependent A* router where edge travel times depend on exact arrival timestamp at downstream nodes.
"""

import heapq
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_schema import EdgeType
from app.optimization.route_schema import OptimizationRoute


class TimeDependentAStarRouter:
    """
    Time-Dependent A* Router.
    Calculates dynamic travel time along path based on exact timestamp of arrival at each downstream node.
    """

    def _heuristic(self, node_a: str, node_b: str, builder: SpatioTemporalGraphBuilder) -> float:
        """Calculates spatial heuristic h(n)."""
        na = builder.graph.nodes.get(node_a, {}).get("data", {})
        nb = builder.graph.nodes.get(node_b, {}).get("data", {})

        lat1, lon1 = na.get("latitude", 13.0827), na.get("longitude", 80.2707)
        lat2, lon2 = nb.get("latitude", 13.0827), nb.get("longitude", 80.2707)

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return (6371000.0 * c) / 16.67

    def find_route(
        self,
        builder: SpatioTemporalGraphBuilder,
        origin: str,
        destination: str,
        departure_timestamp: float = 0.0,
    ) -> OptimizationRoute:
        """
        Finds time-dependent shortest path considering dynamic departure timestamp t_dep = t_start + g(u).
        """
        start_t = time.perf_counter()

        if origin == destination:
            return OptimizationRoute(
                origin_junction=origin,
                destination_junction=destination,
                path_junctions=[origin],
                algorithm_name="TimeDependentAStar",
            )

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
                    algorithm_name="TimeDependentAStar",
                )

            current_timestamp = departure_timestamp + g

            for _, v, k, d in builder.graph.edges(u, data=True, keys=True):
                edge_type = d.get("edge_type")
                if edge_type == EdgeType.ROAD_CONNECTS_JUNCTION.value or d.get("road_id"):
                    road_id = d.get("road_id", f"ROAD_{u}_{v}")
                    length_m = d.get("length_meters", 500.0)

                    # Dynamic time-dependent travel time evaluation
                    t_base = d.get("travel_time", 30.0)

                    # If road has dynamic temporal profile function
                    if "temporal_profile" in d and callable(d["temporal_profile"]):
                        t_time = d["temporal_profile"](current_timestamp)
                    else:
                        t_time = t_base

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
            algorithm_name="TimeDependentAStar",
        )
