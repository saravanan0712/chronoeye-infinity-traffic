from app.schemas.plate import PlateObservation, PlateValidationStatus
from app.perception.plate_fusion import TemporalPlateFusionEngine, FusionConfig

engine = TemporalPlateFusionEngine()

obs1 = PlateObservation(
    track_id="TRK_102",
    camera_id="CAM_A",
    frame_id=1,
    timestamp=0.04,
    raw_text="PSX8525",
    normalized_text="PSX8525",
    ocr_confidence=0.854,
    validation_status=PlateValidationStatus.FORMAT_MISMATCH,
    validation_confidence=0.40,
    quality_score=0.5,
    preprocessing_variant="STACKED_SPLIT"
)

fused1 = engine.process_observation("TRK_102", "CAM_A", obs1)
print(f"After Frame 1: best='{fused1.best_plate_number}', pending='{fused1.pending_plate_number}', status='{fused1.status}'")

obs2 = PlateObservation(
    track_id="TRK_102",
    camera_id="CAM_A",
    frame_id=8,
    timestamp=0.32,
    raw_text="SX8525",
    normalized_text="SX8525",
    ocr_confidence=0.994,
    validation_status=PlateValidationStatus.FORMAT_MISMATCH,
    validation_confidence=0.40,
    quality_score=0.5,
    preprocessing_variant="STACKED_SPLIT"
)

fused2 = engine.process_observation("TRK_102", "CAM_A", obs2)
print(f"After Frame 8: best='{fused2.best_plate_number}', pending='{fused2.pending_plate_number}', status='{fused2.status}'")
