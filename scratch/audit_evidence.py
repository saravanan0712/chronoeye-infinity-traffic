import json
from collections import Counter

with open("data/diagnostic_crops/valid_observations.jsonl", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total lines in valid_observations.jsonl: {len(lines)}")

real_video = []
synthetic = []

for idx, line in enumerate(lines, start=1):
    try:
        data = json.loads(line)
        norm_text = data.get("normalized_text", "")
        raw_text = data.get("raw_text", "")
        track_id = data.get("track_id", "")
        crop_path = data.get("crop_path", "")
        
        # Check source indicators
        # Synthetic tests in backend/tests usually use KA01MH9999, TN09AB1234, TN10AB1234, AP09QP7744, or dummy crop paths like /tmp or test
        is_synth = False
        if any(dummy in norm_text for dummy in ["KA01", "TN09", "TN10", "AP09", "MH12", "DL01"]):
            is_synth = True
        elif "test" in str(crop_path).lower() or "mock" in str(crop_path).lower() or crop_path is None or crop_path == "":
            is_synth = True
        elif norm_text in ["KW527", "PSX8525", "SX8525", "8525", "TS5330", "PTS5330", "5330"]:
            is_synth = False
        else:
            is_synth = True
            
        if is_synth:
            synthetic.append((idx, track_id, raw_text, norm_text, crop_path))
        else:
            real_video.append((idx, track_id, raw_text, norm_text, data))
    except Exception as e:
        print(f"Error parsing line {idx}: {e}")

print(f"\nSynthetic / Test observations count: {len(synthetic)}")
synth_plates = Counter([s[3] for s in synthetic])
print(f"Synthetic plates breakdown: {dict(synth_plates)}")

print(f"\nReal video observations count: {len(real_video)}")
real_plates = Counter([r[3] for r in real_video])
print(f"Real video plates breakdown: {dict(real_plates)}")

print("\nSample real video records:")
for idx, track_id, raw, norm, d in real_video[:10]:
    print(f"Line {idx}: Track={track_id}, Raw={raw}, Norm={norm}, Frame={d.get('frame_number')}, Conf={d.get('overall_confidence')}, ValStatus={d.get('validator_status')}, Fusion={d.get('fusion_status')}, Crop={d.get('crop_path')}")
