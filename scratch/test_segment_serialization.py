import sys, os
sys.path.insert(0, os.path.abspath('backend'))

from app.schemas.reid import JourneySegment, VehicleJourney
import json

s = JourneySegment(
    camera_id="CAM_A_EAST",
    track_id="TRK_101",
    timestamp=100.0,
    plate_number="TN09AB1234",
    plate_confidence=0.94,
    plate_status="CONFIRMED",
)

dump = s.model_dump()
print("JourneySegment model_dump:")
print(json.dumps(dump, indent=2))

assert dump["plate_number"] == "TN09AB1234"
assert dump["plate_confidence"] == 0.94
assert dump["plate_status"] == "CONFIRMED"
print("API serialization test PASSED successfully!")
