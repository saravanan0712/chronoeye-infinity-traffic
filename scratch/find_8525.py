import json

with open("data/diagnostic_crops/valid_observations.jsonl", "r") as f:
    for idx, line in enumerate(f, start=1):
        line_str = line.strip()
        if not line_str:
            continue
        data = json.loads(line_str)
        norm = data.get("normalized_text", "")
        best = data.get("confirmed_plate_text", "")
        event = data.get("event", "")
        track_id = data.get("track_id", "")
        frame = data.get("frame_number", "")
        if "8525" in norm or "8525" in best:
            print(f"Line {idx}: Event={event} Track={track_id} Frame={frame} Norm='{norm}' Best='{best}' Data={data}")
