"""
ChronoEye Infinity - Phase 11: ChronoEye Predictive A* Router
Predictive A* router leveraging Phase 8/9 predicted travel times, predicted congestion,
uncertainty penalties, and incident blockages to recommend optimal spatio-temporal routes.
"""

import heapq
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder
from app.graph.graph_schema import EdgeType
from app.optimization.route_schema import OptimizationRoute, RouteSegment
from app.optimization.graph_cost import SpatioTemporalEdgeCostCalculator


class ChronoEyePredictiveAStarRouter:
    """
    ChronoEye Dynamic Predictive A* Router.
    """

    def __init__(self, cost_calculator: Optional[SpatioTemporalEdgeCostCalculator] = None):
        self.cost_calculator = cost_calculator or SpatioTemporalEdgeCostCalculator()

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
        forecasting_engine: Optional[Any] = None,
        uncertainty_estimator: Optional[Any] = None,
    ) -> OptimizationRoute:
        """
        Finds optimal predictive route using ST-GNN predicted travel time, congestion,
        uncertainty penalties, and incident clearance status.
        """
        start_t = time.perf_counter()

        if origin == destination:
            return OptimizationRoute(
                origin_junction=origin,
                destination_junction=destination,
                path_junctions=[origin],
                algorithm_name="ChronoEyePredictiveAStar",
            )

        h_start = self._heuristic(origin, destination, builder)
        # Priority Queue: (f_score, g_cost, current_node, path_nodes, path_roads, total_dist, accum_travel_time)
        pq: List[Tuple[float, float, str, List[str], List[str], float, float]] = [
            (h_start, 0.0, origin, [origin], [], 0.0, 0.0)
        ]
        visited = set()

        while pq:
            f, g, u, path_nodes, path_roads, dist, travel_time_acc = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == destination:
                latency = round((time.perf_counter() - start_t) * 1000.0, 3)
                lower_conf = round(travel_time_acc * 0.9, 2)
                upper_conf = round(travel_time_acc * 1.15, 2)
                return OptimizationRoute(
                    origin_junction=origin,
                    destination_junction=destination,
                    path_junctions=path_nodes,
                    path_roads=path_roads,
                    total_distance_km=round(dist, 2),
                    total_travel_time_seconds=round(travel_time_acc, 2),
                    total_predicted_delay_seconds=round(max(0.0, travel_time_acc - (dist * 60.0)), 2),
                    computation_latency_ms=latency,
                    algorithm_name="ChronoEyePredictiveAStar",
                    confidence_lower_seconds=lower_conf,
                    confidence_upper_seconds=upper_conf,
                )

            current_timestamp = departure_timestamp + travel_time_acc

            for _, v, k, d in builder.graph.edges(u, data=True, keys=True):
                edge_type = d.get("edge_type")
                if edge_type == EdgeType.ROAD_CONNECTS_JUNCTION.value or d.get("road_id"):
                    road_id = d.get("road_id", f"ROAD_{u}_{v}")
                    length_m = d.get("length_meters", 500.0)
                    base_t_time = d.get("travel_time", 30.0)
                    cong_score = d.get("congestion_score", 0.0)
                    unc_std = d.get("uncertainty_std", 0.0)
                    has_inc = d.get("has_incident", False)

                    # Dynamic predicted travel time evaluation
                    if "temporal_profile" in d and callable(d["temporal_profile"]):
                        pred_t_time = d["temporal_profile"](current_timestamp)
                    else:
                        pred_t_time = base_t_time

                    seg = RouteSegment(
                        road_id=road_id,
                        from_junction=u,
                        to_junction=v,
                        length_meters=length_m,
                        travel_time_seconds=pred_t_time,
                        congestion_score=cong_score,
                        uncertainty_std=unc_std,
                        has_incident=has_inc,
                    )

                    step_cost = self.cost_calculator.calculate_cost(seg, predicted_travel_time=pred_t_time)

                    if v not in visited:
                        new_g = g + step_cost
                        new_travel_acc = travel_time_acc + pred_t_time
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
                                new_travel_acc,
                            ),
                        )

        latency = round((time.perf_counter() - start_t) * 1000.0, 3)
        return OptimizationRoute(
            origin_junction=origin,
            destination_junction=destination,
            path_junctions=[],
            computation_latency_ms=latency,
            algorithm_name="ChronoEyePredictiveAStar",
        )
