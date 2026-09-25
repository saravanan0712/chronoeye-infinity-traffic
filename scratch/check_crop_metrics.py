import cv2
import numpy as np
import os
import json

with open("outputs/failure_audit_crops/audit_inspection_data.json") as f:
    data = json.load(f)

crop_dir = "outputs/failure_audit_crops"

print("================ FAILURE CROP VISUAL INSPECTION SUMMARY ================")
for tid, info in sorted(data.items(), key=lambda x: int(x[0].split("_")[1])):
    bf = info["best_frame"]
    status = info["status"]
    best_p = info["best_plate"]
    pend_p = info["pending_plate"]
    
    # find crop
    cfile = None
    for fn in os.listdir(crop_dir):
        if fn.startswith(f"{tid}_f{bf}_"):
            cfile = os.path.join(crop_dir, fn)
            break
            
    if not cfile or not os.path.exists(cfile):
        print(f"Track {tid:7s}: No crop found.")
        continue
        
    img = cv2.imread(cfile)
    h, w = img.shape[:2]
    
    # Check brightness, blur (laplacian variance), contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_val = np.mean(gray)
    min_val, max_val = np.min(gray), np.max(gray)
    dyn_range = max_val - min_val
    
    print(f"Track {tid:7s} | Status: {status:9s} | Plate: {best_p or pend_p or 'None':10s} | Frame: {bf:03d} | Size: {w:3d}x{h:2d} | Blur: {lap_var:6.1f} | Mean: {mean_val:5.1f} | DynRange: {dyn_range:3d}")
