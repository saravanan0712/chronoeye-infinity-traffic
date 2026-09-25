import os
import cv2

video_path = "outputs/chronoeye_annotated.mp4"
input_path = "data/videos/traffic_video_modified.mp4"

print("=" * 60)
print("VERIFYING ANNOTATED OUTPUT VIDEO")
print("=" * 60)

# 1. Exists
exists = os.path.exists(video_path)
print(f"1. Exists:           {exists} ({video_path})")
assert exists, "Output video does not exist!"

# 2. Non-empty
file_size = os.path.getsize(video_path)
print(f"2. File Size:        {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
assert file_size > 0, "Output video file is empty!"

# 3. OpenCV can reopen
cap = cv2.VideoCapture(video_path)
opened = cap.isOpened()
print(f"3. OpenCV Reopened:  {opened}")
assert opened, "OpenCV failed to open output video!"

# 4. Frame count > 0
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"4. Frame Count:      {frames} frames (at {fps:.2f} FPS)")
assert frames > 0, f"Frame count is {frames}, expected > 0!"

# 5. Dimensions match input video
in_cap = cv2.VideoCapture(input_path)
in_w = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_h = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
in_cap.release()

print(f"5. Dimensions:       {width}x{height} (Input Video: {in_w}x{in_h})")
assert width == in_w and height == in_h, f"Dimensions mismatch! Expected {in_w}x{in_h}, got {width}x{height}"

print("\nALL 5 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
