"""
Lightweight test for the serialization fix in run_full_video_audit.py.
Verifies that fused.observation_count and fused.raw_observations_audit
work correctly — WITHOUT processing any video frames.
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath("backend"))

from app.schemas.plate import FusedPlateIdentity

# Simulate exactly what plate_fusion.py builds as audit_trail
audit_trail = [
    {
        "frame_id": 42,
        "raw_text": "MH12AB1234",
        "normalized_text": "MH12AB1234",
        "voted_text": "MH12AB1234",
        "ocr_confidence": 0.91,
        "quality_score": 0.85,
        "variant": "ORIGINAL",
        "score": 0.78,
        "validation_status": "VALID",
    },
    {
        "frame_id": 43,
        "raw_text": "MH12AB1235",
        "normalized_text": "MH12AB1235",
        "voted_text": "MH12AB1234",
        "ocr_confidence": 0.88,
        "quality_score": 0.80,
        "variant": "CLAHE",
        "score": 0.72,
        "validation_status": "VALID",
    },
]

fused = FusedPlateIdentity(
    track_id="TRK_104",
    camera_id="CAM_A_EAST",
    best_plate_number="MH12AB1234",
    overall_confidence=0.91,
    observation_count=2,
    confirmed=True,
    status="CONFIRMED",
    pending_plate_number=None,
    evidence_frames=[42, 43],
    supporting_observations_count=2,
    weighted_evidence_score=0.78,
    character_agreement_ratio=1.0,
    raw_observations_audit=audit_trail,
)

# ---- EXACT serialization from the FIXED run_full_video_audit.py ----
res = {
    "track_id": "TRK_104",
    "fused_status": fused.status if fused else "NONE",
    "confirmed": fused.confirmed if fused else False,
    "best_plate": fused.best_plate_number if fused else None,
    "overall_conf": fused.overall_confidence if fused else 0.0,
    "observations_count": fused.observation_count if fused else 0,
    "observations": [
        {
            "frame_id": o["frame_id"],
            "raw_text": o["raw_text"],
            "normalized_text": o["normalized_text"],
            "ocr_confidence": o["ocr_confidence"],
            "overall_confidence": o.get("score", 0.0),
            "validation_status": o.get("validation_status", ""),
        }
        for o in (fused.raw_observations_audit if fused else [])
    ],
}

# JSON round-trip (exactly what the script does)
output = json.dumps(res, indent=2)
parsed = json.loads(output)

# Assertions
assert parsed["observations_count"] == 2, f"Expected 2, got {parsed['observations_count']}"
assert len(parsed["observations"]) == 2, f"Expected 2 obs, got {len(parsed['observations'])}"
assert parsed["observations"][0]["frame_id"] == 42
assert parsed["observations"][0]["normalized_text"] == "MH12AB1234"
assert parsed["observations"][0]["ocr_confidence"] == 0.91
assert parsed["observations"][0]["validation_status"] == "VALID"
assert parsed["confirmed"] is True
assert parsed["best_plate"] == "MH12AB1234"

print("=" * 60)
print("LIGHTWEIGHT SERIALIZATION FIX TEST")
print("=" * 60)
print(f"  fused type             : {type(fused).__name__}")
print(f"  observations_count     : {res['observations_count']}")
print(f"  observations entries   : {len(res['observations'])}")
for o in res["observations"]:
    print(f"    frame={o['frame_id']} text='{o['normalized_text']}' conf={o['ocr_confidence']} status={o['validation_status']}")
print()
print("JSON round-trip: OK")
print("All assertions passed.")
print("TEST PASSED")
