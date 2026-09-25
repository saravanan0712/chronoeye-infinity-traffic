import json

required_obs_fields = [
    "frame_number", "track_id", "raw_text", "normalized_text",
    "ocr_confidence", "validator_status", "validation_confidence",
    "overall_confidence", "fusion_status", "crop_path"
]

required_conf_fields = [
    "frame_number", "track_id", "confirmed_plate_text",
    "confirmation_confidence", "crop_path"
]

with open("data/diagnostic_crops/valid_observations.jsonl", "r") as f:
    lines = [l.strip() for l in f if l.strip()]

total = len(lines)
real_obs = []
real_conf = []
synth_obs = []

for i, line in enumerate(lines, 1):
    data = json.loads(line)
    ev_type = data.get("event", "")
    norm = data.get("normalized_text", "")
    conf_plate = data.get("confirmed_plate_text", "")
    crop = data.get("crop_path", "")
    
    is_real = False
    if norm in ["KW527", "PSX8525", "SX8525", "TS5330", "PTS5330"] or conf_plate in ["KW527", "SX8525", "TS5330"]:
        if "outputs/anpr_debug" in crop:
            is_real = True
            
    if is_real:
        if ev_type == "VALID_OBSERVATION":
            real_obs.append((i, data))
        elif ev_type == "PLATE_CONFIRMED":
            real_conf.append((i, data))
    else:
        synth_obs.append((i, data))

print(f"Total lines: {total}")
print(f"Real Video VALID_OBSERVATION records: {len(real_obs)}")
print(f"Real Video PLATE_CONFIRMED records: {len(real_conf)}")
print(f"Synthetic / Unit-test records: {len(synth_obs)}")

# Verify fields on real observations
missing_obs_fields = []
for idx, d in real_obs:
    for rf in required_obs_fields:
        if rf not in d:
            missing_obs_fields.append((idx, rf))

print(f"Real observations missing fields: {len(missing_obs_fields)}")

# Verify fields on confirmation events
missing_conf_fields = []
for idx, d in real_conf:
    for rf in required_conf_fields:
        if rf not in d:
            missing_conf_fields.append((idx, rf))

print(f"Confirmation events missing fields: {len(missing_conf_fields)}")

# Unique plate occurrences in real video
print("\nReal Video Observations Summary:")
plates_seen = {}
for idx, d in real_obs:
    p = d["normalized_text"]
    plates_seen[p] = plates_seen.get(p, 0) + 1
print(plates_seen)

print("\nReal Video Plate Confirmations Summary:")
conf_seen = {}
for idx, d in real_conf:
    p = d["confirmed_plate_text"]
    conf_seen[p] = conf_seen.get(p, 0) + 1
print(conf_seen)
