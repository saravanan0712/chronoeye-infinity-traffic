import sys
import os
import traceback

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

try:
    import run_video_detection
    args = run_video_detection.parse_args()
    args.source = "data/videos/traffic_video_modified.mp4"
    args.anpr = True
    args.max_frames = 100
    run_video_detection.run_pipeline(args)
except SystemExit as e:
    print(f"SystemExit: {e}")
except Exception as e:
    print(f"Exception caught: {e}")
    traceback.print_exc()
