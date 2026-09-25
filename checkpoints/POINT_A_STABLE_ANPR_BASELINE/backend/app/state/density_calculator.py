"""
ChronoEye Infinity - Phase 7: Traffic Density & Occupancy Calculator
Calculates traffic density k (vehicles/km) and spatial occupancy (0.0 to 1.0) for road segments.
"""

class DensityCalculator:
    """
    Traffic Density & Spatial Occupancy Calculator.
    """

    @staticmethod
    def calculate_density(vehicle_count: int, length_meters: float = 500.0, *, road_length_meters: float = None) -> float:
        """
        Calculates traffic density k = N / length_km (vehicles / km).
        road_length_meters is a backward-compatible alias for length_meters.
        """
        if road_length_meters is not None:
            length_meters = road_length_meters
        if vehicle_count <= 0 or length_meters <= 0:
            return 0.0

        length_km = max(0.05, length_meters / 1000.0)
        density = vehicle_count / length_km
        return round(float(density), 2)


    @staticmethod
    def calculate_occupancy(
        vehicle_count: int,
        length_meters: float = 500.0,
        avg_vehicle_length_m: float = 5.0,
        num_lanes: int = 1,
        *,
        road_length_meters: float = None,
    ) -> float:
        """
        Calculates spatial occupancy ratio (0.0 to 1.0).
        Occupancy = (N * avg_vehicle_length) / (length_meters * num_lanes).
        """
        if road_length_meters is not None:
            length_meters = road_length_meters
            
        if vehicle_count <= 0 or length_meters <= 0 or num_lanes <= 0:
            return 0.0

        total_road_capacity_length = length_meters * float(num_lanes)
        total_vehicle_length = float(vehicle_count) * avg_vehicle_length_m

        occupancy = total_vehicle_length / max(1.0, total_road_capacity_length)
        return round(min(1.0, max(0.0, float(occupancy))), 3)
