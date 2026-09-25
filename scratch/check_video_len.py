import cv2

video_path = r"data\videos\traffic_video_modified.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

print(f"Video Info:")
print(f"  Path: {video_path}")
print(f"  Total frames: {total_frames}")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Duration: {total_frames / max(1.0, fps):.2f}s")
