import cv2
import os
import glob

base_dir = r"E:\chronoeye\outputs\target_track_diagnosis"
tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

for t in tracks:
    t_path = os.path.join(base_dir, t)
    files = glob.glob(os.path.join(t_path, "*.png"))
    print(f"\n================ TRACK: {t} ================")
    print(f"Total crop files: {len(files)}")
    veh_files = sorted([f for f in files if "_veh_" in f])
    cand_files = sorted([f for f in files if "_cand_" in f])
    print(f"Veh files ({len(veh_files)}): {[os.path.basename(f) for f in veh_files[:10]]}")
    print(f"Cand files ({len(cand_files)}): {[os.path.basename(f) for f in cand_files[:10]]}")
