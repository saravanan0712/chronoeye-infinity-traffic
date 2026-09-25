import cv2
import numpy as np
import os
import glob
import json

base_dir = r"E:\chronoeye\outputs\full_video_audit"
veh_crops = sorted(glob.glob(os.path.join(base_dir, "*_best_*.png")))

print(f"Total vehicle crops saved: {len(veh_crops)}")
for vf in veh_crops:
    fname = os.path.basename(vf)
    print(f"Saved: {fname}")
