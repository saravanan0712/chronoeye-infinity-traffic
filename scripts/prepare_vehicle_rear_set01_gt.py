"""
ChronoEye Infinity — Vehicle-Rear Set01 Ground-Truth Extraction & Normalization
Parses Camera1 and Camera2 Set01 annotations from vehicles.xml or JSON metadata,
identifies cross-camera vehicle identities, and creates structured GT artifacts.

STRICT PROTOCOL: Read-only GT parser. Does not modify ChronoEye production code.
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

# Add project root to sys.path to access normalization utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from app.perception.ocr_engine import TextNormalizer
except ImportError:
    class TextNormalizer:
        @classmethod
        def normalize(cls, raw: str) -> str:
            import re
            if not raw:
                return ""
            return re.sub(r"[^A-Z0-9]", "", str(raw).upper().strip())


def normalize_plate_string(plate_text: Optional[str]) -> str:
    """Normalizes plate string for consistent ground-truth comparison."""
    if not plate_text:
        return ""
    try:
        return TextNormalizer.normalize(str(plate_text).strip())
    except Exception:
        import re
        return re.sub(r"[^A-Z0-9]", "", str(plate_text).upper().strip())


def parse_vehicle_rear_xml(xml_path: str, camera_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses Vehicle-Rear vehicles.xml ground-truth file for a single camera.
    Returns mapping: vehicle_gt_id -> vehicle_data_dict.
    """
    vehicles: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(xml_path):
        return vehicles

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[GT Parser ERROR] Failed to parse XML file {xml_path}: {e}")
        return vehicles

    # Format A: <vehicles><vehicle id="..."> ... <frame number="..."> ...
    for v_elem in root.findall(".//vehicle"):
        v_id = v_elem.get("id") or v_elem.findtext("id") or ""
        raw_plate = v_elem.findtext("plate") or v_elem.findtext("license_plate") or v_elem.get("plate") or ""
        v_type = v_elem.findtext("type") or v_elem.findtext("vehicle_type") or v_elem.get("type") or "car"
        make = v_elem.findtext("make") or v_elem.findtext("brand") or ""
        model = v_elem.findtext("model") or ""
        color = v_elem.findtext("color") or v_elem.findtext("colour") or ""
        year = v_elem.findtext("year") or ""
        motorcycle = v_elem.findtext("motorcycle") in {"1", "true", "True"}
        quality = v_elem.findtext("quality") or "good"
        discard = v_elem.findtext("discard") in {"1", "true", "True"}

        norm_plate = normalize_plate_string(raw_plate)
        canonical_id = norm_plate if norm_plate else f"{camera_id}_V{v_id}"

        frame_observations: List[Dict[str, Any]] = []

        # Find frames under vehicle
        for f_elem in v_elem.findall(".//frame"):
            try:
                frame_num_str = f_elem.get("number") or f_elem.get("id") or f_elem.findtext("number") or "0"
                frame_num = int(frame_num_str)
            except ValueError:
                continue

            # Plate bbox
            p_elem = f_elem.find("plate") or f_elem.find("plate_bbox")
            plate_bbox = None
            if p_elem is not None:
                try:
                    px = float(p_elem.get("x") or p_elem.get("x1") or 0)
                    py = float(p_elem.get("y") or p_elem.get("y1") or 0)
                    pw = float(p_elem.get("width") or p_elem.get("w") or 0)
                    ph = float(p_elem.get("height") or p_elem.get("h") or 0)
                    plate_bbox = [px, py, pw, ph]
                except (ValueError, TypeError):
                    plate_bbox = None

            # Vehicle bbox
            veh_elem = f_elem.find("vehicle") or f_elem.find("box") or f_elem.find("bbox")
            veh_bbox = None
            if veh_elem is not None:
                try:
                    vx = float(veh_elem.get("x") or veh_elem.get("x1") or 0)
                    vy = float(veh_elem.get("y") or veh_elem.get("y1") or 0)
                    vw = float(veh_elem.get("width") or veh_elem.get("w") or 0)
                    vh = float(veh_elem.get("height") or veh_elem.get("h") or 0)
                    veh_bbox = [vx, vy, vw, vh]
                except (ValueError, TypeError):
                    veh_bbox = None

            frame_observations.append({
                "frame": frame_num,
                "plate_bbox": plate_bbox,
                "vehicle_bbox": veh_bbox,
            })

        frame_observations.sort(key=lambda x: x["frame"])

        vehicles[canonical_id] = {
            "vehicle_gt_id": canonical_id,
            "raw_id": v_id,
            "camera_id": camera_id,
            "raw_plate": raw_plate,
            "normalized_plate": norm_plate,
            "vehicle_type": v_type,
            "make": make,
            "model": model,
            "color": color,
            "year": year,
            "motorcycle": motorcycle,
            "quality": quality,
            "discard": discard,
            "observations": frame_observations,
            "observation_count": len(frame_observations),
            "min_frame": frame_observations[0]["frame"] if frame_observations else 0,
            "max_frame": frame_observations[-1]["frame"] if frame_observations else 0,
        }

    return vehicles


def parse_vehicle_rear_json(json_path: str, camera_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses alternative JSON format ground-truth metadata (e.g. dataset_3.json).
    """
    vehicles: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(json_path):
        return vehicles

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[GT Parser ERROR] Failed to parse JSON file {json_path}: {e}")
        return vehicles

    # Support dict of vehicles or list of annotations
    items = data if isinstance(data, list) else data.get("vehicles", data.get("annotations", []))
    if isinstance(items, dict):
        items = list(items.values())

    for idx, item in enumerate(items):
        v_id = str(item.get("id", item.get("vehicle_id", idx)))
        raw_plate = item.get("plate", item.get("license_plate", ""))
        norm_plate = normalize_plate_string(raw_plate)
        canonical_id = norm_plate if norm_plate else f"{camera_id}_V{v_id}"

        cam = item.get("camera_id", camera_id)
        if cam != camera_id and camera_id not in str(cam):
            continue

        raw_frames = item.get("frames", item.get("observations", []))
        observations = []
        for rf in raw_frames:
            f_num = int(rf.get("frame", rf.get("frame_id", 0)))
            p_box = rf.get("plate_bbox", rf.get("plate", None))
            v_box = rf.get("vehicle_bbox", rf.get("bbox", None))
            observations.append({
                "frame": f_num,
                "plate_bbox": p_box,
                "vehicle_bbox": v_box,
            })
        observations.sort(key=lambda x: x["frame"])

        vehicles[canonical_id] = {
            "vehicle_gt_id": canonical_id,
            "raw_id": v_id,
            "camera_id": camera_id,
            "raw_plate": raw_plate,
            "normalized_plate": norm_plate,
            "vehicle_type": item.get("type", item.get("vehicle_type", "car")),
            "make": item.get("make", ""),
            "model": item.get("model", ""),
            "color": item.get("color", ""),
            "year": item.get("year", ""),
            "motorcycle": bool(item.get("motorcycle", False)),
            "quality": item.get("quality", "good"),
            "discard": bool(item.get("discard", False)),
            "observations": observations,
            "observation_count": len(observations),
            "min_frame": observations[0]["frame"] if observations else 0,
            "max_frame": observations[-1]["frame"] if observations else 0,
        }

    return vehicles


def extract_cross_camera_ground_truth(
    cam1_vehicles: Dict[str, Dict[str, Any]],
    cam2_vehicles: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identifies common vehicles appearing across Camera 1 and Camera 2.
    Returns: (cross_camera_vehicles, single_camera_vehicles).
    """
    cross_camera: List[Dict[str, Any]] = []
    single_camera: List[Dict[str, Any]] = []

    # Map by normalized plate if available, else by vehicle_gt_id
    cam1_by_plate = {v["normalized_plate"]: v for v in cam1_vehicles.values() if v["normalized_plate"]}
    cam2_by_plate = {v["normalized_plate"]: v for v in cam2_vehicles.values() if v["normalized_plate"]}

    common_plates = set(cam1_by_plate.keys()) & set(cam2_by_plate.keys())

    # Build cross-camera matched records
    for plate in sorted(common_plates):
        v1 = cam1_by_plate[plate]
        v2 = cam2_by_plate[plate]

        # Ignore if both discarded
        if v1.get("discard", False) and v2.get("discard", False):
            continue

        record = {
            "gt_identity": plate,
            "normalized_plate": plate,
            "raw_plate_c1": v1.get("raw_plate", plate),
            "raw_plate_c2": v2.get("raw_plate", plate),
            "vehicle_type": v1.get("vehicle_type") or v2.get("vehicle_type") or "car",
            "make": v1.get("make") or v2.get("make") or "",
            "model": v1.get("model") or v2.get("model") or "",
            "color": v1.get("color") or v2.get("color") or "",
            "year": v1.get("year") or v2.get("year") or "",
            "camera1_summary": {
                "raw_id": v1.get("raw_id", ""),
                "min_frame": v1.get("min_frame", 0),
                "max_frame": v1.get("max_frame", 0),
                "observation_count": v1.get("observation_count", 0),
            },
            "camera2_summary": {
                "raw_id": v2.get("raw_id", ""),
                "min_frame": v2.get("min_frame", 0),
                "max_frame": v2.get("max_frame", 0),
                "observation_count": v2.get("observation_count", 0),
            },
            "camera1_observations": v1.get("observations", []),
            "camera2_observations": v2.get("observations", []),
        }
        cross_camera.append(record)

    # Collect single-camera vehicles for negative sampling
    for plate, v1 in cam1_by_plate.items():
        if plate not in common_plates:
            single_camera.append(v1)
    for plate, v2 in cam2_by_plate.items():
        if plate not in common_plates:
            single_camera.append(v2)

    return cross_camera, single_camera


def prepare_set01_ground_truth(
    cam1_path: str,
    cam2_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Main entry point for ground-truth extraction and artifact creation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse Camera 1
    if cam1_path.endswith(".json"):
        cam1_data = parse_vehicle_rear_json(cam1_path, "Camera1")
    else:
        cam1_data = parse_vehicle_rear_xml(cam1_path, "Camera1")

    # Parse Camera 2
    if cam2_path.endswith(".json"):
        cam2_data = parse_vehicle_rear_json(cam2_path, "Camera2")
    else:
        cam2_data = parse_vehicle_rear_xml(cam2_path, "Camera2")

    cross_cam, single_cam = extract_cross_camera_ground_truth(cam1_data, cam2_data)

    gt_artifact = {
        "dataset_name": "Vehicle-Rear Set01",
        "camera1_total_vehicles": len(cam1_data),
        "camera2_total_vehicles": len(cam2_data),
        "cross_camera_vehicles_count": len(cross_cam),
        "single_camera_vehicles_count": len(single_cam),
        "camera1_vehicles": cam1_data,
        "camera2_vehicles": cam2_data,
    }

    cross_artifact = {
        "dataset_name": "Vehicle-Rear Set01 Cross-Camera Ground Truth",
        "cross_camera_count": len(cross_cam),
        "cross_camera_vehicles": cross_cam,
        "single_camera_vehicles": [
            {
                "vehicle_gt_id": v["vehicle_gt_id"],
                "camera_id": v["camera_id"],
                "normalized_plate": v["normalized_plate"],
                "observation_count": v["observation_count"],
                "min_frame": v["min_frame"],
                "max_frame": v["max_frame"],
            }
            for v in single_cam
        ],
    }

    gt_json_path = os.path.join(output_dir, "set01_ground_truth.json")
    cross_json_path = os.path.join(output_dir, "set01_cross_camera_gt.json")

    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_artifact, f, indent=2)

    with open(cross_json_path, "w", encoding="utf-8") as f:
        json.dump(cross_artifact, f, indent=2)

    print(f"[GT Extraction Success] Parsed Camera 1 ({len(cam1_data)} vehicles) and Camera 2 ({len(cam2_data)} vehicles).")
    print(f"  Cross-Camera Matched Vehicles: {len(cross_cam)}")
    print(f"  Single-Camera Vehicles:        {len(single_cam)}")
    print(f"  Saved: {gt_json_path}")
    print(f"  Saved: {cross_json_path}")

    return cross_artifact


def main():
    parser = argparse.ArgumentParser(description="Vehicle-Rear Set01 Ground Truth Preparation")
    parser.add_argument("--config", type=str, default="experiments/vehicle_rear_set01/config.json", help="Path to config.json")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root directory of Vehicle-Rear extracted dataset")
    parser.add_argument("--cam1-xml", type=str, default=None, help="Path to Camera 1 vehicles.xml")
    parser.add_argument("--cam2-xml", type=str, default=None, help="Path to Camera 2 vehicles.xml")
    parser.add_argument("--output-dir", type=str, default="experiments/vehicle_rear_set01/ground_truth", help="Output directory for GT JSONs")
    args = parser.parse_args()

    cam1_xml = args.cam1_xml
    cam2_xml = args.cam2_xml
    output_dir = args.output_dir

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            ds_cfg = cfg.get("dataset", {})
            root = args.dataset_root or ds_cfg.get("dataset_root", "")
            if not cam1_xml:
                rel_c1 = ds_cfg.get("camera1_gt_xml", "")
                cam1_xml = os.path.join(root, rel_c1) if root and rel_c1 else rel_c1
            if not cam2_xml:
                rel_c2 = ds_cfg.get("camera2_gt_xml", "")
                cam2_xml = os.path.join(root, rel_c2) if root and rel_c2 else rel_c2
            out_cfg = cfg.get("output_paths", {})
            if out_cfg.get("ground_truth_dir"):
                output_dir = out_cfg["ground_truth_dir"]

    if not cam1_xml or not cam2_xml:
        print("[ERROR] Please provide --cam1-xml and --cam2-xml paths or a valid config.json.")
        sys.exit(1)

    prepare_set01_ground_truth(cam1_xml, cam2_xml, output_dir)


if __name__ == "__main__":
    main()
