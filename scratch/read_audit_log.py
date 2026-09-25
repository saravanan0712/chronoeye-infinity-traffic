import json
import os

# Let's see the log of run_full_video_audit (task-313) or parse the output
log_path = r"C:\Users\Vijayakumari K\.gemini\antigravity-ide\brain\255eba75-0b89-406f-87c1-8ee1cfaf3335\.system_generated\tasks\task-313.log"
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for l in lines:
    if "Frame 715/715" in l or "Completed 715 frames" in l or "Confirmed (" in l:
        print(l.strip())
