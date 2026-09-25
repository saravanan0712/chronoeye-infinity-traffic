import json

with open("outputs/full_video_audit/full_video_audit_results.json") as f:
    before = json.load(f)

with open("outputs/full_video_audit_pointd/full_video_audit_results_pointd.json") as f:
    after = json.load(f)

print("=== COMPARISON BETWEEN BEFORE AND POINT D ===")
for b, a in zip(before, after):
    tid = b["track_id"]
    b_plate = b.get("best_plate") or b.get("pending_plate")
    a_plate = a.get("best_plate") or a.get("pending_plate")
    b_conf = b.get("confirmed")
    a_conf = a.get("confirmed")
    if b_plate != a_plate or b_conf != a_conf:
        print(f"Track {tid:7s}: Before={b_plate} (conf={b_conf}) -> After={a_plate} (conf={a_conf})")

b_confirmed = [d for d in before if d["confirmed"]]
a_confirmed = [d for d in after if d["confirmed"]]
b_pending = [d for d in before if not d["confirmed"] and (d["best_plate"] or d["pending_plate"])]
a_pending = [d for d in after if not d["confirmed"] and (d["best_plate"] or d["pending_plate"])]
b_unresolved = [d for d in before if not d["confirmed"] and not d["best_plate"] and not d["pending_plate"]]
a_unresolved = [d for d in after if not d["confirmed"] and not d["best_plate"] and not d["pending_plate"]]

print(f"\nBefore Confirmed ({len(b_confirmed)}): {[d['best_plate'] for d in b_confirmed]}")
print(f"After Confirmed  ({len(a_confirmed)}): {[d['best_plate'] for d in a_confirmed]}")
print(f"Before Pending   ({len(b_pending)}): {[d['best_plate'] or d['pending_plate'] for d in b_pending]}")
print(f"After Pending    ({len(a_pending)}): {[d['best_plate'] or d['pending_plate'] for d in a_pending]}")
print(f"Before Unresolved: {len(b_unresolved)}, After Unresolved: {len(a_unresolved)}")
