import cv2
import numpy as np
import os
import glob
import json

base_dir = r"E:\chronoeye\outputs\full_video_audit"
files = sorted(glob.glob(os.path.join(base_dir, "*_best_*.png")))

print(f"Total best track crop files: {len(files)}")
for f in files:
    fname = os.path.basename(f)
    parts = fname.split("_")
    tid = parts[0] + "_" + parts[1]
    bf = parts[3]
    sz = parts[4].replace(".png", "")
    print(f"Track: {tid:7s} | Best Frame: {bf} | Max Size: {sz}")
