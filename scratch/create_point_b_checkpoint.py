import os
import shutil
import hashlib
import subprocess
import json
from datetime import datetime

POINT_A_DIR = r"E:\chronoeye\checkpoints\POINT_A_STABLE_ANPR_BASELINE"
POINT_B_DIR = r"E:\chronoeye\checkpoints\POINT_B_STACKED_RECOGNITION_IMPROVED"
WORKSPACE_DIR = r"E:\chronoeye"

def run_git(cmd, cwd):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed in {cwd}: {cmd}\nStdout: {result.stdout}\nStderr: {result.stderr}")
    return result.stdout.strip()

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def create_checkpoint():
    print("Step 1: Verifying Point A integrity...")
    if not os.path.exists(POINT_A_DIR):
        raise FileNotFoundError(f"Point A directory does not exist: {POINT_A_DIR}")
    
    pt_a_commit = run_git('git log -1 --format="%H"', POINT_A_DIR).strip('"')
    print(f"Point A Commit Hash: {pt_a_commit}")
    if pt_a_commit != "5e7290f030fea412ea07a4cc0862571e964a0a4f":
        raise ValueError(f"Point A commit mismatch! Expected 5e7290f030fea412ea07a4cc0862571e964a0a4f, got {pt_a_commit}")

    pt_a_tag = run_git('git tag -l "POINT_A_STABLE_ANPR_BASELINE"', POINT_A_DIR)
    print(f"Point A Tag: {pt_a_tag}")

    print("\nStep 2: Preparing Point B directory...")
    if os.path.exists(POINT_B_DIR):
        print(f"Removing existing Point B directory: {POINT_B_DIR}")
        shutil.rmtree(POINT_B_DIR)

    print(f"Copying Point A repository to Point B: {POINT_B_DIR}...")
    shutil.copytree(POINT_A_DIR, POINT_B_DIR)

    print("\nStep 3: Updating Point B with working tree files...")
    files_to_copy = [
        os.path.join("backend", "app", "perception", "plate_preprocessor.py"),
        os.path.join("backend", "tests", "test_stacked_plate_recognition.py"),
    ]

    for rel_path in files_to_copy:
        src = os.path.join(WORKSPACE_DIR, rel_path)
        dst = os.path.join(POINT_B_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {rel_path} -> {compute_sha256(dst)[:16]}")

    print("\nStep 4: Committing Point B changes in Git...")
    run_git("git add backend/app/perception/plate_preprocessor.py backend/tests/test_stacked_plate_recognition.py", POINT_B_DIR)
    
    commit_msg = "POINT_B_STACKED_RECOGNITION_IMPROVED: Upper stacked-plate horizontal margin trimming and focused regression tests"
    run_git(f'git commit -m "{commit_msg}"', POINT_B_DIR)

    pt_b_commit = run_git('git log -1 --format="%H"', POINT_B_DIR).strip('"')
    print(f"Point B Commit Hash: {pt_b_commit}")

    print("\nStep 5: Creating Point B Git Tag...")
    run_git('git tag -a "POINT_B_STACKED_RECOGNITION_IMPROVED" -m "POINT_B_STACKED_RECOGNITION_IMPROVED: Upper stacked-plate horizontal margin trimming"', POINT_B_DIR)

    pt_b_tag = run_git('git tag -l "POINT_B_STACKED_RECOGNITION_IMPROVED"', POINT_B_DIR)
    print(f"Point B Tag: {pt_b_tag}")

    print("\nStep 6: Generating MANIFEST.json for Point B...")
    manifest = {
        "checkpoint_tag": "POINT_B_STACKED_RECOGNITION_IMPROVED",
        "commit_hash": pt_b_commit,
        "parent_commit": pt_a_commit,
        "timestamp": datetime.now().isoformat(),
        "description": "Point B — Stacked plate recognition improved with upper margin-inset trimming, recovering SX8525 and TS5330",
        "files": {}
    }

    for root, dirs, files in os.walk(POINT_B_DIR):
        if ".git" in root.split(os.sep):
            continue
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, POINT_B_DIR)
            if rel_path == "MANIFEST.json":
                continue
            manifest["files"][rel_path] = {
                "sha256": compute_sha256(full_path),
                "size_bytes": os.path.getsize(full_path)
            }

    manifest_path = os.path.join(POINT_B_DIR, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written with {len(manifest['files'])} files tracked.")

    print("\nStep 7: Verifying Point A remains immutable and intact...")
    pt_a_commit_final = run_git('git log -1 --format="%H"', POINT_A_DIR).strip('"')
    if pt_a_commit_final != "5e7290f030fea412ea07a4cc0862571e964a0a4f":
        raise ValueError("Point A was mutated!")
    print(f"Point A verified intact: {pt_a_commit_final}")

    print("\nStep 8: Verifying Point B Git history...")
    git_history = run_git("git log -n 2 --oneline", POINT_B_DIR)
    print("Point B Log:")
    print(git_history)

    print("\nSUCCESS: Point B checkpoint established.")
    return pt_b_commit, pt_b_tag

if __name__ == "__main__":
    create_checkpoint()
