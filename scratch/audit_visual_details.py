import cv2
import numpy as np
import os

video_path = r"data\videos\traffic_video_modified.mp4"
cap = cv2.VideoCapture(video_path)

out_dir = r"E:\chronoeye\outputs\target_track_audit_visual"
os.makedirs(out_dir, exist_ok=True)

# Frames of interest for each track:
# TRK_104: frames 3, 5, 7, 8, 10
# TRK_105: frames 20, 21, 22
# TRK_107: frames 23, 30, 36, 40
# TRK_109: frames 92, 95, 98, 99
# TRK_110: frames 93, 96, 98, 100
# TRK_111: frames 98, 99, 100

frames_to_dump = [3, 5, 7, 8, 10, 20, 21, 22, 23, 30, 36, 40, 92, 93, 95, 96, 98, 99, 100]

saved = {}
for f in range(1, 101):
    ret, frame = cap.read()
    if not ret:
        break
    if f in frames_to_dump:
        saved[f] = frame.copy()

cap.release()

print("Analyzing high-resolution visual details...")
# Let's crop exact regions and save with high-res zoom
# TRK_104: at frame 7 bbox=(901,666,1248,899)
f7 = saved[7]
t104_f7 = f7[666:899, 901:1248]
cv2.imwrite(os.path.join(out_dir, "TRK_104_f7_veh.png"), t104_f7)
# Zoom into bumper / rear
t104_plate_f7 = f7[780:860, 1020:1140]
cv2.imwrite(os.path.join(out_dir, "TRK_104_f7_plate_zoom.png"), t104_plate_f7)

# TRK_105: at frame 20 bbox=(0,622,328,1004) - Auto-rickshaw on far left
f20 = saved[20]
t105_f20 = f20[622:1004, 0:328]
cv2.imwrite(os.path.join(out_dir, "TRK_105_f20_veh.png"), t105_f20)

# TRK_107: at frame 40 bbox=(1603,700,1700,831) - far right vehicle
f40 = saved[40]
t107_f40 = f40[700:831, 1603:1700]
cv2.imwrite(os.path.join(out_dir, "TRK_107_f40_veh.png"), t107_f40)

# TRK_109: at frame 98 bbox=(1364,615,1565,911)
f98 = saved[98]
t109_f98 = f98[615:911, 1364:1565]
cv2.imwrite(os.path.join(out_dir, "TRK_109_f98_veh.png"), t109_f98)

# TRK_110: at frame 100 bbox=(281,606,609,883)
f100 = saved[100]
t110_f100 = f100[606:883, 281:609]
cv2.imwrite(os.path.join(out_dir, "TRK_110_f100_veh.png"), t110_f100)

# TRK_111: at frame 99 bbox=(1654,678,1756,783)
f99 = saved[99]
t111_f99 = f99[678:783, 1654:1756]
cv2.imwrite(os.path.join(out_dir, "TRK_111_f99_veh.png"), t111_f99)

print("Visual audit crops dumped to outputs/target_track_audit_visual.")
