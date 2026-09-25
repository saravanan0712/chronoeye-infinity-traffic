import os
import subprocess

# Let's see how many tests each file has
test_dir = "backend/tests"
files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]

print(f"Total test files in {test_dir}: {len(files)}")
total_pytest = 0
for f in sorted(files):
    path = os.path.join(test_dir, f)
    res = subprocess.run(["E:\\chronoeye\\chronoeye\\Scripts\\pytest.exe", "--collect-only", "-q", path], capture_output=True, text=True)
    lines = [l for l in res.stdout.strip().split("\n") if l and not l.startswith("warning") and not "warning" in l.lower() and not "test session starts" in l]
    # last line usually is "X tests collected in Ys"
    count = 0
    for l in lines:
        if "test collected" in l or "tests collected" in l:
            try:
                count = int(l.split()[0])
            except:
                pass
    print(f"{f}: {count}")
    total_pytest += count

print(f"\nSum of all collected tests: {total_pytest}")
