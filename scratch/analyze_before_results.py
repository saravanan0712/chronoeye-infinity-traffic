import json

with open("outputs/full_video_audit/full_video_audit_results.json") as f:
    data = json.load(f)

total_tracks = len(data)
confirmed = [d for d in data if d["confirmed"]]
pending = [d for d in data if not d["confirmed"] and (d["best_plate"] or d["pending_plate"])]
unresolved = [d for d in data if not d["confirmed"] and not d["best_plate"] and not d["pending_plate"]]

total_obs = sum(d["observations_count"] for d in data)

print(f"Total Tracks: {total_tracks}")
print(f"Confirmed ({len(confirmed)}):")
for d in confirmed:
    print(f"  {d['track_id']}: {d['best_plate']} (obs={d['observations_count']}, conf={d['overall_conf']})")

print(f"\nPending ({len(pending)}):")
for d in pending:
    plate = d['pending_plate'] or d['best_plate']
    print(f"  {d['track_id']}: {plate} (obs={d['observations_count']}, conf={d['overall_conf']})")

print(f"\nUnresolved ({len(unresolved)}):")
for d in unresolved:
    print(f"  {d['track_id']}: max_size={d['max_size']}, frames={d['frame_count']}")

print(f"\nTotal valid observations across all tracks: {total_obs}")
