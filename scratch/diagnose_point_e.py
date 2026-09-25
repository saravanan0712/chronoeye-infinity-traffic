import cv2, json, os, sys

json_path = r"outputs/full_video_audit_pointd/full_video_audit_results_pointd.json"
with open(json_path, "r") as f:
    results = json.load(f)

res_by_id = {r["track_id"]: r for r in results}

target_tracks = ["TRK_104", "TRK_114", "TRK_116", "TRK_117", "TRK_125", "TRK_127", "TRK_109", "TRK_111", "TRK_112"]

for tid in target_tracks:
    r = res_by_id.get(tid)
    if not r:
        continue
    print("=" * 60)
    print(f"TRACK {tid}: status={r['fused_status']}, confirmed={r['confirmed']}")
    print(f"  Best Plate:    {r['best_plate']}")
    print(f"  Pending Plate: {r['pending_plate']}")
    print(f"  Confidence:    {r['overall_conf']:.3f}")
    print(f"  Frame Range:   {r['start_frame']} - {r['end_frame']} ({r['frame_count']} frames)")
    print(f"  Max Vehicle:   {r['max_size'][0]}x{r['max_size'][1]} at best frame {r['best_frame']}")
    print(f"  Observations ({r['observations_count']}):")
    for o in r["observations"]:
        print(f"    - Frame {o['frame_id']:03d}: raw='{o['raw_text']}' norm='{o['normalized_text']}' ocr_conf={o['ocr_confidence']:.3f} overall={o['overall_confidence']:.3f} val_status={o['validation_status']}")
