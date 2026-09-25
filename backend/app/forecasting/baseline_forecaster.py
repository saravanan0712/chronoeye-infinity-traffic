"""
ChronoEye Infinity - Phase 8: Level 1 Persistence / Moving Average Forecaster
Simple baseline forecaster that uses recent historical observations to project future traffic states.
"""

from typing import List, Optional
from app.state.traffic_state_schema import NetworkTrafficSnapshot
from app.forecasting.forecasting_schema import (
    ForecastHorizon,
    SegmentForecast,
    ForecastingModelType,
)


class PersistenceForecaster:
    """
    Level 1 Baseline Persistence & Moving Average Forecaster.
    """

    HORIZON_SECONDS = {
        ForecastHorizon.PLUS_5MIN: 300.0,
        ForecastHorizon.PLUS_10MIN: 600.0,
        ForecastHorizon.PLUS_15MIN: 900.0,
        ForecastHorizon.PLUS_30MIN: 1800.0,
    }

    def __init__(self, moving_avg_window: int = 3):
        self.moving_avg_window = moving_avg_window

    def predict(
        self,
        history: List[NetworkTrafficSnapshot],
        horizon: ForecastHorizon,
        segment_id: str = "ROAD_R_AB",
    ) -> SegmentForecast:
        """
        Predicts future traffic conditions using moving average of recent snapshots.
        """
        if not history:
            return SegmentForecast(
                segment_id=segment_id,
                target_timestamp=self.HORIZON_SECONDS[horizon],
                horizon=horizon,
                predicted_flow=0.0,
                predicted_queue=0,
                predicted_density=0.0,
                predicted_speed=60.0,
                predicted_travel_time=30.0,
            )

        current_t = history[-1].timestamp
        target_t = current_t + self.HORIZON_SECONDS[horizon]
        recent_snaps = history[-self.moving_avg_window:]

        flows, queues, densities, speeds, travel_times = [], [], [], [], []
        for snap in recent_snaps:
            seg = snap.segment_states.get(segment_id)
            if seg:
                flows.append(seg.flow_rate)
                queues.append(float(seg.queue_length))
                densities.append(seg.density)
                speeds.append(seg.average_speed)
                travel_times.append(seg.travel_time)

        p_flow = round(sum(flows) / float(len(flows)), 2) if flows else 0.0
        p_queue = int(round(sum(queues) / float(len(queues)))) if queues else 0
        p_density = round(sum(densities) / float(len(densities)), 2) if densities else 0.0
        p_speed = round(sum(speeds) / float(len(speeds)), 2) if speeds else 60.0
        p_tt = round(sum(travel_times) / float(len(travel_times)), 2) if travel_times else 30.0

        return SegmentForecast(
            segment_id=segment_id,
            target_timestamp=target_t,
            horizon=horizon,
            predicted_flow=max(0.0, p_flow),
            predicted_queue=max(0, p_queue),
            predicted_density=max(0.0, p_density),
            predicted_speed=max(0.0, p_speed),
            predicted_travel_time=max(1.0, p_tt),
        )
