import cv2
import numpy as np
import os

video_path = r"data\videos\traffic_video_modified.mp4"
cap = cv2.VideoCapture(video_path)

out_dir = r"E:\chronoeye\outputs\target_track_audit_visual\TRK_104_deep"
os.makedirs(out_dir, exist_ok=True)

for fid in range(1, 30):
    ret, frame = cap.read()
    if not ret:
        break
    if fid in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        # Save vehicle and plate crops
        # In frame 7: vehicle was at (901, 666, 1248, 899)
        # In frame 3: (954, 669, 1254, 893)
        # Let's save the exact bumper region
        cv2.imwrite(os.path.join(out_dir, f"frame_{fid:02d}_full.png"), frame)

cap.release()
print("TRK_104 frames dumped.")
