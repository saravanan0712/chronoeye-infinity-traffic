from app.perception.frame_source import VideoFrameSource
src = VideoFrameSource('data/videos/traffic_video_modified.mp4')
for i in range(100):
    success, frame, meta = src.read_frame()
    if not success:
        print(f"Failed at frame {i}")
        break
print(f"Read {i+1} frames successfully")
