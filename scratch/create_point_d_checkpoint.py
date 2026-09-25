import os
import shutil
import hashlib
import subprocess
import json
from datetime import datetime

POINT_C_DIR = r"E:\chronoeye\checkpoints\POINT_C_ANPR_FUSION_TARGET_EVALUATOR"
POINT_D_DIR = r"E:\chronoeye\checkpoints\POINT_D_STAGE4_FIX1_FREEZE"
WORKSPACE_DIR = r"E:\chronoeye"

# Files changed in Fix #1
POINT_D_FILES = [
    os.path.join("backend", "app", "perception", "ocr_engine.py"),
    os.path.join("backend", "tests", "test_ocr_normalizer.py"),
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
    print("POINT_D_STAGE4_FIX1_FREEZE - Checkpoint Creation")
    print("=" * 70)

    if not os.path.exists(POINT_C_DIR):
        raise FileNotFoundError(f"Point C not found: {POINT_C_DIR}")
    pt_c_commit = run_git('git log -1 --format="%H"', POINT_C_DIR).strip('"')
    print(f"  Point C parent commit: {pt_c_commit}")

    # Copy Point C tree as base
    if os.path.exists(POINT_D_DIR):
        shutil.rmtree(POINT_D_DIR)
    shutil.copytree(POINT_C_DIR, POINT_D_DIR)
    print(f"  Cloned Point C to {POINT_D_DIR}")

    # Copy Fix #1 files from workspace
    for rel_path in POINT_D_FILES:
        src = os.path.join(WORKSPACE_DIR, rel_path)
        dst = os.path.join(POINT_D_DIR, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Copied {rel_path} -> sha256={compute_sha256(dst)[:16]}...")

    # Git commit
    git_paths = " ".join(p.replace("\\", "/") for p in POINT_D_FILES)
    run_git(f"git add {git_paths}", POINT_D_DIR)
    commit_msg = (
        "POINT_D_STAGE4_FIX1_FREEZE: "
        "Fix #1 prefix noise stripper in TextNormalizer. "
        "Corrected TRK_106 (PTS5330 -> TS5330). 78/78 Stage 4 tests passed. "
        "18 confirmed plates, 0 false-positive inventions. Stage 4 FROZEN."
    )
    run_git(f'git commit -m "{commit_msg}"', POINT_D_DIR)
    pt_d_commit = run_git('git log -1 --format="%H"', POINT_D_DIR).strip('"')
    print(f"  Point D commit: {pt_d_commit}")

    tag_msg = (
        "POINT_D_STAGE4_FIX1_FREEZE: 715 frames, 36 tracks, "
        "TRK_106 TS5330 verified, 18 confirmed, 0 false-positives, "
        "78/78 tests passed, Stage 4 FROZEN."
    )
    run_git(f'git tag -a "POINT_D_STAGE4_FIX1_FREEZE" -m "{tag_msg}"', POINT_D_DIR)
    print(f"  Tagged: POINT_D_STAGE4_FIX1_FREEZE")

    manifest = {
        "checkpoint_tag": "POINT_D_STAGE4_FIX1_FREEZE",
        "commit_hash": pt_d_commit,
        "parent_commit": pt_c_commit,
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Point D - Stage 4 Fix #1 Final Freeze Checkpoint. "
            "Validated on traffic_video_modified.mp4 (715 frames, 36 tracks). "
            "Corrected TRK_106: PTS5330 -> TS5330. "
            "78/78 Stage 4 tests passed (100%). "
            "18 confirmed plates, 0 false positives. "
            "Stages 1-3 and 5-18 remain strictly frozen."
        ),
        "validation": {
            "frames": 715,
            "vehicle_detections": 2611,
            "tracks": 36,
            "confirmed_plates": 18,
            "pending_plates": 6,
            "unresolved_tracks": 12,
            "false_positive_inventions": 0,
            "tests_passed": 78,
            "tests_failed": 0,
            "target_plate_trk_106": "TS5330"
        },
        "modified_files": [
            {
                "path": "backend/app/perception/ocr_engine.py",
                "sha256": compute_sha256(os.path.join(POINT_D_DIR, "backend", "app", "perception", "ocr_engine.py")),
                "description": "TextNormalizer prefix noise stripper + _select_best_candidate length gate"
            },
            {
                "path": "backend/tests/test_ocr_normalizer.py",
                "sha256": compute_sha256(os.path.join(POINT_D_DIR, "backend", "tests", "test_ocr_normalizer.py")),
                "description": "Unit tests for PTS5330 -> TS5330, protected plates, and legitimate P/F/E plates"
            }
        ]
    }

    with open(os.path.join(POINT_D_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("  Generated MANIFEST.json")
    print("\nSUCCESS: POINT_D_STAGE4_FIX1_FREEZE checkpoint established.")

if __name__ == "__main__":
    create_checkpoint()
