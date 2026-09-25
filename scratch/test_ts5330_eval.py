import sys
import os
sys.path.insert(0, os.path.abspath("backend"))

from app.perception.plate_fusion import TemporalPlateFusionEngine
from app.schemas.plate import PlateObservation, PlateValidationStatus
from app.perception.target_plate_evaluator import TargetPlateEvaluator

engine = TemporalPlateFusionEngine()

# Frame 80
obs1 = PlateObservation(
    track_id="TRK_106",
    camera_id="CAM_A",
    frame_id=80,
    timestamp=2.67,
    raw_text="PTS5330",
    normalized_text="TS5330",
    ocr_confidence=0.955,
    validation_status=PlateValidationStatus.FORMAT_MISMATCH,
    validation_confidence=0.40,
    quality_score=0.5,
    preprocessing_variant="STACKED_SPLIT",
    source="ALPR_PIPELINE"
)
fused1 = engine.process_observation("TRK_106", "CAM_A", obs1, track_frame_count=15)
print("After Frame 80:", fused1.best_plate_number, fused1.status, fused1.confirmed)

# Frame 85
obs2 = PlateObservation(
    track_id="TRK_106",
    camera_id="CAM_A",
    frame_id=85,
    timestamp=2.83,
    raw_text="PTS5330",
    normalized_text="TS5330",
    ocr_confidence=0.955,
    validation_status=PlateValidationStatus.FORMAT_MISMATCH,
    validation_confidence=0.40,
    quality_score=0.5,
    preprocessing_variant="STACKED_SPLIT",
    source="ALPR_PIPELINE"
)
fused2 = engine.process_observation("TRK_106", "CAM_A", obs2, track_frame_count=15)
print("After Frame 85:", fused2.best_plate_number, fused2.status, fused2.confirmed)

evaluator = TargetPlateEvaluator()
res = evaluator.evaluate("TS5330", engine)
print("TargetPlateEvaluator TS5330 status:", res.status, "plate:", res.recognized_plate)
