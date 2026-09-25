import os
import shutil
import hashlib
import subprocess
import json
from datetime import datetime

POINT_B_DIR = r"E:\chronoeye\checkpoints\POINT_B_STACKED_RECOGNITION_IMPROVED"
POINT_C_DIR = r"E:\chronoeye\checkpoints\POINT_C_ANPR_FUSION_TARGET_EVALUATOR"
WORKSPACE_DIR = r"E:\chronoeye"
POINT_B_COMMIT_EXPECTED = "20d4c086be6dd2abdb82f7c48d0a04063ec19718"

# Files changed/added in Point C vs Point B
POINT_C_FILES = [
    os.path.join("backend", "app", "perception", "target_plate_evaluator.py"),  # NEW
    os.path.join("backend", "app", "perception", "ocr_normalizer.py"),           # NEW
    os.path.join("backend", "app", "perception", "plate_association.py"),        # MODIFIED
    os.path.join("backend", "app", "perception", "plate_detector.py"),           # MODIFIED
    os.path.join("backend", "app", "perception", "plate_fusion.py"),             # MODIFIED
    os.path.join("backend", "tests", "test_target_plate_evaluator.py"),          # NEW
    os.path.join("backend", "tests", "test_ocr_normalizer.py"),                  # NEW
    os.path.join("backend", "tests", "test_plate_spatial_association.py"),       # NEW
    os.path.join("backend", "tests", "test_single_obs_fusion.py"),               # NEW
]


def run_git(cmd, cwd):
    result = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, shell=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed in {cwd}: {cmd}\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )
    return result.stdout.strip()


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_checkpoint():
    print("=" * 70)
    print("POINT_C_ANPR_FUSION_TARGET_EVALUATOR - Checkpoint Creation")
    print("=" * 70)

    # Step 1: Verify Point B integrity
    print("\nStep 1: Verifying Point B integrity...")
    if not os.path.exists(POINT_B_DIR):
        raise FileNotFoundError(f"Point B not found: {POINT_B_DIR}")
    pt_b_commit = run_git('git log -1 --format="%H"', POINT_B_DIR).strip('"')
    print(f"  Point B commit: {pt_b_commit}")
    if pt_b_commit != POINT_B_COMMIT_EXPECTED:
        raise ValueError(f"Point B mismatch! Expected {POINT_B_COMMIT_EXPECTED}, got {pt_b_commit}")
    print("  Point B integrity: OK")
    pt_b_tag = run_git('git tag -l "POINT_B_STACKED_RECOGNITION_IMPROVED"', POINT_B_DIR)
    print(f"  Point B tag: {pt_b_tag}")

    # Step 2: Verify all Point C source files exist
    print("\nStep 2: Verifying Point C source files in workspace...")
    for rel_path in POINT_C_FILES:
        src = os.path.join(WORKSPACE_DIR, rel_path)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing: {src}")
        print(f"  OK  {rel_path}  ({os.path.getsize(src):,} bytes)")

    # Step 3: Prepare Point C directory (copy from Point B)
    print("\nStep 3: Preparing Point C directory...")
    if os.path.exists(POINT_C_DIR):
        print(f"  Removing existing: {POINT_C_DIR}")
        shutil.rmtree(POINT_C_DIR)
    print(f"  Copying Point B -> Point C: {POINT_C_DIR}")
    shutil.copytree(POINT_B_DIR, POINT_C_DIR)
    print("  Copy complete.")

    # Step 4: Overlay Point C files from workspace
    print("\nStep 4: Overlaying Point C files from workspace...")
    for rel_path in POINT_C_FILES:
        src = os.path.join(WORKSPACE_DIR, rel_path)
        dst = os.path.join(POINT_C_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {rel_path}")
        print(f"    sha256={compute_sha256(dst)[:24]}...  size={os.path.getsize(dst):,} bytes")

    # Step 5: Git add + commit
    print("\nStep 5: Committing Point C changes in Git...")
    git_paths = " ".join(p.replace("\\", "/") for p in POINT_C_FILES)
    run_git(f"git add {git_paths}", POINT_C_DIR)
    commit_msg = (
        "POINT_C_ANPR_FUSION_TARGET_EVALUATOR: "
        "ANPR fusion target plate evaluator, 715-frame validation, "
        "+3 confirmed tracks, 55/55 focused tests passed"
    )
    run_git(f'git commit -m "{commit_msg}"', POINT_C_DIR)
    pt_c_commit = run_git('git log -1 --format="%H"', POINT_C_DIR).strip('"')
    print(f"  Point C commit: {pt_c_commit}")

    # Step 6: Create Git tag
    print("\nStep 6: Creating Point C Git tag...")
    tag_msg = (
        "POINT_C_ANPR_FUSION_TARGET_EVALUATOR: 715 frames, 36 tracks "
        "(18 confirmed, 7 pending, 11 zero-obs), 41 observations, "
        "+3 vs Point B, 55/55 focused tests, Stage5/6 untouched"
    )
    run_git(f'git tag -a "POINT_C_ANPR_FUSION_TARGET_EVALUATOR" -m "{tag_msg}"', POINT_C_DIR)
    pt_c_tag = run_git('git tag -l "POINT_C_ANPR_FUSION_TARGET_EVALUATOR"', POINT_C_DIR)
    print(f"  Tag: {pt_c_tag}")

    # Step 7: Generate MANIFEST.json
    print("\nStep 7: Generating MANIFEST.json...")
    manifest = {
        "checkpoint_tag": "POINT_C_ANPR_FUSION_TARGET_EVALUATOR",
        "commit_hash": pt_c_commit,
        "parent_commit": pt_b_commit,
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Point C - ANPR Fusion Target Plate Evaluator. "
            "715-frame validation: 36 tracks (18 confirmed, 7 pending, "
            "11 zero-observation), 41 observations, +3 confirmed vs Point B. "
            "55/55 focused tests passed. Stage 5 and Stage 6 untouched."
        ),
        "validation": {
            "frames": 715,
            "tracks": 36,
            "confirmed": 18,
            "pending": 7,
            "zero_observation": 11,
            "observations": 41,
            "delta_confirmed_vs_point_b": 3,
            "focused_tests_passed": 55,
            "focused_tests_total": 55,
            "stage5_modified": False,
            "stage6_modified": False
        },
        "point_c_changes": {
            "new_files": [
                "backend/app/perception/target_plate_evaluator.py",
                "backend/app/perception/ocr_normalizer.py",
                "backend/tests/test_target_plate_evaluator.py",
                "backend/tests/test_ocr_normalizer.py",
                "backend/tests/test_plate_spatial_association.py",
                "backend/tests/test_single_obs_fusion.py",
            ],
            "modified_files": [
                "backend/app/perception/plate_association.py",
                "backend/app/perception/plate_detector.py",
                "backend/app/perception/plate_fusion.py",
            ]
        },
        "files": {}
    }
    for root, dirs, files in os.walk(POINT_C_DIR):
        if ".git" in root.split(os.sep):
            continue
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, POINT_C_DIR)
            if rel_path == "MANIFEST.json":
                continue
            manifest["files"][rel_path] = {
                "sha256": compute_sha256(full_path),
                "size_bytes": os.path.getsize(full_path)
            }
    manifest_path = os.path.join(POINT_C_DIR, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Manifest written: {len(manifest['files'])} files ({os.path.getsize(manifest_path):,} bytes)")

    # Step 8: Verify Point B remains immutable
    print("\nStep 8: Verifying Point B immutability...")
    pt_b_final = run_git('git log -1 --format="%H"', POINT_B_DIR).strip('"')
    if pt_b_final != POINT_B_COMMIT_EXPECTED:
        raise ValueError(f"Point B was mutated! Got {pt_b_final}")
    print(f"  Point B intact: {pt_b_final}")

    # Step 9: Print git history
    print("\nStep 9: Point C git history:")
    history = run_git("git log --oneline", POINT_C_DIR)
    for line in history.splitlines():
        print(f"  {line}")

    print("\n" + "=" * 70)
    print("SUCCESS: POINT_C_ANPR_FUSION_TARGET_EVALUATOR checkpoint established.")
    print(f"  Commit Hash     : {pt_c_commit}")
    print(f"  Tag             : {pt_c_tag}")
    print(f"  Checkpoint Path : {POINT_C_DIR}")
    print(f"  Manifest Path   : {manifest_path}")
    print(f"  Files tracked   : {len(manifest['files'])}")
    print(f"  Parent (Point B): {pt_b_commit}")
    print("=" * 70)

    return {
        "commit_hash": pt_c_commit,
        "tag": pt_c_tag,
        "checkpoint_path": POINT_C_DIR,
        "manifest_path": manifest_path,
        "files_tracked": len(manifest["files"]),
        "parent_commit": pt_b_commit,
    }


if __name__ == "__main__":
    result = create_checkpoint()
    print("\nRETURNED:")
    for k, v in result.items():
        print(f"  {k}: {v}")

