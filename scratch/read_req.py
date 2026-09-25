import json, sys

with open(r"C:\Users\Vijayakumari K\.gemini\antigravity-ide\brain\88970f11-55a8-439c-a628-916bcfd59c9a\.system_generated\logs\transcript_full.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            print("="*40)
            print(f"STEP {data.get('step_index')}")
            content = data.get("content", "")
            sys.stdout.buffer.write(content.encode("utf-8"))
            sys.stdout.buffer.write(b"\n")
