import cv2
import os

video_path = r"E:\chronoeye\data\videos\traffic_video_modified.mp4"
out_dir = r"E:\chronoeye\outputs\diagnostic_inspection"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total video frames: {total_frames}")

# Key frames to inspect for tracks:
# TRK_101: frames 1, 6
# TRK_102: frames 1, 8
# TRK_103: frames 1, 7
# TRK_104: frames 8, 13, 23, 38
# TRK_105: frames 20, 25
# TRK_106: frames 23, 43, 68, 73, 80
# TRK_107: frames 28, 33, 43
# TRK_108: frames 31, 36, 41, 46, 56
# TRK_109: frames 90, 95, 100
# TRK_110: frames 93, 98
# TRK_111: frame 98

frames_to_save = {1, 6, 8, 13, 20, 23, 25, 31, 33, 36, 38, 41, 46, 56, 68, 73, 80, 90, 93, 98, 100}

frame_idx = 0
while cap.isOpened() and frame_idx <= 100:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx in frames_to_save:
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:04d}.jpg"), frame)

cap.release()
print(f"Saved key frames to {out_dir}")
