import cv2
import os
import glob

base_dir = r"E:\chronoeye\outputs\target_track_diagnosis"
tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

for t in tracks:
    t_path = os.path.join(base_dir, t)
    veh_files = sorted(glob.glob(os.path.join(t_path, "*veh*.png")))
    cand_files = sorted(glob.glob(os.path.join(t_path, "*cand*.png")))
    print(f"\n================ TRACK: {t} ================")
    print(f"Vehicle files ({len(veh_files)}): {[os.path.basename(f) for f in veh_files]}")
    print(f"Candidate files ({len(cand_files)}): {[os.path.basename(f) for f in cand_files]}")
