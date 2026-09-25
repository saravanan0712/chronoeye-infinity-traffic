import cv2
import numpy as np
import os
import glob

base_dir = r"E:\chronoeye\outputs\target_track_deep_diagnosis"
target_tracks = ["TRK_104", "TRK_105", "TRK_107", "TRK_109", "TRK_110", "TRK_111"]

for tid in target_tracks:
    tdir = os.path.join(base_dir, tid)
    veh_imgs = sorted(glob.glob(os.path.join(tdir, "*_veh_*.png")))
    cand_imgs = sorted(glob.glob(os.path.join(tdir, "*_cand_*.png")))
    print(f"\n=======================================================")
    print(f"TRACK {tid}:")
    print(f"  Veh crops ({len(veh_imgs)}): {[os.path.basename(f) for f in veh_imgs]}")
    print(f"  Cand crops ({len(cand_imgs)}): {[os.path.basename(f) for f in cand_imgs]}")
    
    # Check dimensions and intensity of cand crops
    for c in cand_imgs:
        img = cv2.imread(c)
        if img is not None:
            h, w = img.shape[:2]
            mean_val = float(np.mean(img))
            std_val = float(np.std(img))
            print(f"    Cand {os.path.basename(c)}: {w}x{h}, mean={mean_val:.1f}, std={std_val:.1f}")
