from app.perception.ocr_normalizer import generate_normalized_plate_candidates
from app.perception.plate_fusion import TemporalPlateFusionEngine
from app.perception.ocr_engine import IndianPlateValidator
from app.schemas.plate import PlateObservation, PlateValidationStatus

engine = TemporalPlateFusionEngine()

# Test candidate generation
print("Candidates for PTS5330:", generate_normalized_plate_candidates("PTS5330", "PTS5330"))
print("Candidates for FJZ8479:", generate_normalized_plate_candidates("FJZ8479", "FJZ8479"))
print("Candidates for PIND7823:", generate_normalized_plate_candidates("PIND7823", "PIND7823"))
print("Candidates for X0147:", generate_normalized_plate_candidates("X0147", "X0147"))
print("Candidates for MH12AB1234:", generate_normalized_plate_candidates("MH12AB1234", "MH12AB1234"))

# Simulate frame 80 for TRK_106
raw = "PTS5330"
norm = "PTS5330"
conf = 0.955
cands = generate_normalized_plate_candidates(raw, norm)
print("\nSimulating Frame 80 with candidates:", cands)
for c in cands:
    v_status, v_conf = IndianPlateValidator.validate_format(c)
    obs = PlateObservation(
        track_id="TRK_106",
        camera_id="CAM_A",
        frame_id=80,
        timestamp=2.67,
        raw_text=raw,
        normalized_text=c,
        ocr_confidence=conf,
        validation_status=v_status,
        validation_confidence=v_conf,
        quality_score=0.5,
        preprocessing_variant="ORIGINAL" if c == norm else "ORIGINAL+PREFIX_NORM",
        source="ALPR_PIPELINE"
    )
    fused = engine.process_observation("TRK_106", "CAM_A", obs, track_frame_count=15)
print(f"After Frame 80: best='{fused.best_plate_number}', pending='{fused.pending_plate_number}', status='{fused.status}', confirmed={fused.confirmed}")

# Simulate Frame 85
print("\nSimulating Frame 85 with candidates:", cands)
for c in cands:
    v_status, v_conf = IndianPlateValidator.validate_format(c)
    obs = PlateObservation(
        track_id="TRK_106",
        camera_id="CAM_A",
        frame_id=85,
        timestamp=2.83,
        raw_text=raw,
        normalized_text=c,
        ocr_confidence=conf,
        validation_status=v_status,
        validation_confidence=v_conf,
        quality_score=0.5,
        preprocessing_variant="ORIGINAL" if c == norm else "ORIGINAL+PREFIX_NORM",
        source="ALPR_PIPELINE"
    )
    fused = engine.process_observation("TRK_106", "CAM_A", obs, track_frame_count=15)
print(f"After Frame 85: best='{fused.best_plate_number}', pending='{fused.pending_plate_number}', status='{fused.status}', confirmed={fused.confirmed}")
