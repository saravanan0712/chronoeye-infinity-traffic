import os
import shutil
import hashlib
import json
import subprocess

checkpoint_dir = r"E:\chronoeye\checkpoints\POINT_A_STABLE_ANPR_BASELINE"
os.makedirs(checkpoint_dir, exist_ok=True)

# Define source base
src_base = r"E:\chronoeye"

# Paths to copy
items_to_copy = [
    os.path.join(src_base, "backend"),
    os.path.join(src_base, "run_video_detection.py"),
    os.path.join(src_base, "run_chronoeye.py"),
    os.path.join(src_base, "run_chronoeye.bat"),
    os.path.join(src_base, "pytest.ini"),
]

def ignore_patterns(path, names):
    ignored = set()
    for name in names:
        if name in ["__pycache__", ".pytest_cache", "license_plate_detector.pt", "yolov8n.pt"]:
            ignored.add(name)
        if name.endswith(".pyc"):
            ignored.add(name)
    return ignored

print(f"Creating checkpoint in: {checkpoint_dir}")

for item in items_to_copy:
    rel = os.path.relpath(item, src_base)
    dest = os.path.join(checkpoint_dir, rel)
    if os.path.isdir(item):
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(item, dest, ignore=ignore_patterns)
        print(f"  Copied dir: {rel} -> {dest}")
    elif os.path.isfile(item):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(item, dest)
        print(f"  Copied file: {rel} -> {dest}")

# Generate cryptographic SHA-256 manifest
manifest = {
    "checkpoint_tag": "POINT_A_STABLE_ANPR_BASELINE",
    "timestamp": "2026-09-08T10:30:00+05:30",
    "description": "Point A — Stable ANPR Baseline with verified single-line (KW527) and stacked plate support (SX8525, TS5330)",
    "files": {}
}

verified_count = 0
for root, dirs, files in os.walk(checkpoint_dir):
    if ".git" in root:
        continue
    for f in sorted(files):
        cp_path = os.path.join(root, f)
        rel_path = os.path.relpath(cp_path, checkpoint_dir)
        src_path = os.path.join(src_base, rel_path)
        
        # Hash checkpoint file
        h_cp = hashlib.sha256()
        with open(cp_path, "rb") as fp:
            while chunk := fp.read(65536):
                h_cp.update(chunk)
        cp_hash = h_cp.hexdigest()
        
        # Verify byte-for-byte with source file
        if os.path.exists(src_path):
            h_src = hashlib.sha256()
            with open(src_path, "rb") as fp:
                while chunk := fp.read(65536):
                    h_src.update(chunk)
            src_hash = h_src.hexdigest()
            assert cp_hash == src_hash, f"Hash mismatch for {rel_path}!"
            verified_count += 1
        
        manifest["files"][rel_path] = {
            "sha256": cp_hash,
            "size_bytes": os.path.getsize(cp_path)
        }

manifest_path = os.path.join(checkpoint_dir, "MANIFEST.json")
with open(manifest_path, "w", encoding="utf-8") as fp:
    json.dump(manifest, fp, indent=2)

print(f"Manifest written with {len(manifest['files'])} files. Verified byte-for-byte: {verified_count} files.")

# Initialize Git repository inside checkpoint directory
subprocess.run(["git", "init"], cwd=checkpoint_dir, check=True)
subprocess.run(["git", "config", "user.name", "ChronoEye ANPR"], cwd=checkpoint_dir, check=True)
subprocess.run(["git", "config", "user.email", "anpr@chronoeye.local"], cwd=checkpoint_dir, check=True)
subprocess.run(["git", "add", "."], cwd=checkpoint_dir, check=True)
subprocess.run(["git", "commit", "-m", "POINT_A_STABLE_ANPR_BASELINE: Stable baseline with single-line (KW527) and stacked plate support (SX8525, TS5330)"], cwd=checkpoint_dir, check=True)
subprocess.run(["git", "tag", "POINT_A_STABLE_ANPR_BASELINE"], cwd=checkpoint_dir, check=True)

# Retrieve commit hash
res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkpoint_dir, capture_output=True, text=True, check=True)
commit_hash = res.stdout.strip()
print(f"Git checkpoint committed with hash: {commit_hash}")

res_tag = subprocess.run(["git", "tag", "-l"], cwd=checkpoint_dir, capture_output=True, text=True, check=True)
print(f"Git tags: {res_tag.stdout.strip()}")
