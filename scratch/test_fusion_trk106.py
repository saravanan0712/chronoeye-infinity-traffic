from app.schemas.plate import PlateObservation, PlateValidationStatus
from app.perception.plate_fusion import TemporalPlateFusionEngine

engine = TemporalPlateFusionEngine()

obs1 = PlateObservation(
    track_id="TRK_106",
    camera_id="CAM_A",
    frame_id=80,
    timestamp=2.67,
    raw_text="TS5330",
    normalized_text="TS5330",
    ocr_confidence=1.000,
    validation_status=PlateValidationStatus.FORMAT_MISMATCH,
    validation_confidence=0.40,
    quality_score=0.5,
    preprocessing_variant="STACKED_SPLIT"
)

fused1 = engine.process_observation("TRK_106", "CAM_A", obs1)
print(f"After Frame 80: best='{fused1.best_plate_number}', pending='{fused1.pending_plate_number}', status='{fused1.status}'")

obs2 = PlateObservation(
    track_id="TRK_106",
    camera_id="CAM_A",
    frame_id=85,
    timestamp=2.83,
    raw_text="PTS5330",
    normalized_text="PTS5330",
    ocr_confidence=0.955,
    validation_status=PlateValidationStatus.FORMAT_MISMATCH,
    validation_confidence=0.40,
    quality_score=0.5,
    preprocessing_variant="STACKED_SPLIT"
)

fused2 = engine.process_observation("TRK_106", "CAM_A", obs2)
print(f"After Frame 85: best='{fused2.best_plate_number}', pending='{fused2.pending_plate_number}', status='{fused2.status}'")
