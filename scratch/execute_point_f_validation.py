"""
ChronoEye Infinity - Module 2 Real & Multi-Scenario Multi-Camera Validation Harness
Executes complete 7-signal cross-camera Re-ID, journey reconstruction, lifecycle management,
unobserved gap handling, and Module 3 temporal graph handoff.
"""

import os
import sys
import json
import time

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState, Direction
from app.schemas.plate import VehicleIdentityEvidence, FusedPlateIdentity
from app.schemas.reid import (
    ReIDConfig,
    MatchDecision,
    ReIDScoreBreakdown,
    JourneySegment,
    VehicleJourney,
)
from app.perception.camera_topology import CityCameraTopology, CameraNode
from app.perception.appearance import AppearanceEmbeddingExtractor
from app.perception.reid_engine import ReIDMatchingEngine
from app.perception.journey import JourneyReconstructionEngine
from app.graph.temporal_graph import TemporalTrafficGraphEngine
from app.graph.graph_schema import NodeType, EdgeType
from multi_camera_runner import (
    MultiCameraRunner,
    MultiCameraRunnerConfig,
    CameraSourceConfig,
)


def run_point_f_validation():
    print("=" * 80)
    print("CHRONOEYE INFINITY - MODULE 2 MULTI-CAMERA VALIDATION (POINT F)")
    print("=" * 80)

    results = {
        "dataset_inspection": {},
        "multi_camera_runner_execution": {},
        "scenarios": [],
        "module3_graph_handoff": {},
    }

    # -------------------------------------------------------------------------
    # 1. Inspect Available Datasets & Limitations
    # -------------------------------------------------------------------------
    video_path = os.path.join("data", "videos", "traffic_video_modified.mp4")
    video_exists = os.path.exists(video_path)
    video_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2) if video_exists else 0.0

    print("\n[1. DATASET INSPECTION]")
    print(f"  Primary Video File: {video_path} (Exists: {video_exists}, Size: {video_size_mb} MB)")
    print("  Available Distinct Camera Video Feeds in data/videos: 1 single-viewpoint video file.")
    print("  Limitation Notice: Genuine physical multi-camera multi-angle footage across disjoint geographic junctions is NOT present in data/videos.")
    print("  Validation Strategy: Separating (A) MultiCameraRunner pipeline execution with temporal offset on real footage, (B) Rigorous multi-scenario 7-signal cross-camera evaluations covering all mathematical, physical, and topological edge cases, and (C) Downstream Module 3 spatio-temporal graph integration.")

    results["dataset_inspection"] = {
        "primary_video": video_path,
        "exists": video_exists,
        "size_mb": video_size_mb,
        "distinct_physical_cameras_available": 1,
        "limitation_stated": True,
    }

    # -------------------------------------------------------------------------
    # 2. MultiCameraRunner Real Execution Test (Sequential mode across 2 cameras)
    # -------------------------------------------------------------------------
    print("\n[2. MULTI-CAMERA RUNNER REAL VIDEO EXECUTION]")
    runner_cfg = MultiCameraRunnerConfig(
        cameras=[
            CameraSourceConfig(
                camera_id="CAM_A_EAST",
                source=video_path,
                frame_skip=2,
                max_frames=15,
                timestamp_offset_seconds=0.0,
            ),
            CameraSourceConfig(
                camera_id="CAM_B_WEST",
                source=video_path,
                frame_skip=2,
                max_frames=15,
                timestamp_offset_seconds=12.0,  # 12-second travel delay between junctions
            ),
        ],
        model_path="yolov8n.pt",
        device="cpu",
        conf_threshold=0.35,
        anpr_enabled=True,
        ocr_frame_interval=5,
        reid_enabled=True,
        processing_mode="sequential",
        journey_timeout_seconds=60.0,
        verbose=False,
    )

    runner = MultiCameraRunner(config=runner_cfg)
    run_result = runner.run()

    print(f"  Total Runner Runtime: {run_result.total_runtime_seconds:.2f}s")
    for cam_id, stats in run_result.per_camera_stats.items():
        print(f"  Camera [{cam_id}]: {stats.frames_processed} frames processed, {len(stats.unique_tracks)} unique tracks, {stats.ocr_attempts} OCR attempts, {len(stats.confirmed_plates)} confirmed plates")
    
    summary_rep = run_result.journey_summary
    print(f"  Reconstructed Journeys Count: {len(summary_rep)}")
    for j_item in summary_rep[:5]:
        print(f"    Journey {j_item['journey_id']} (Global: {j_item['global_vehicle_id']}): Plate={j_item['plate_number']}, Cameras={j_item['camera_sequence']}, Segments={j_item['number_of_segments']}")

    results["multi_camera_runner_execution"] = {
        "runtime_seconds": run_result.total_runtime_seconds,
        "per_camera_stats": {
            cam_id: {
                "frames_processed": s.frames_processed,
                "unique_tracks_count": len(s.unique_tracks),
                "ocr_attempts": s.ocr_attempts,
                "confirmed_plates_count": len(s.confirmed_plates),
            }
            for cam_id, s in run_result.per_camera_stats.items()
        },
        "reconstructed_journeys_count": len(summary_rep),
        "journey_sample": summary_rep[:5],
    }

    # -------------------------------------------------------------------------
    # 3. Comprehensive 7-Signal Cross-Camera Scenario Evaluations
    # -------------------------------------------------------------------------
    print("\n[3. COMPREHENSIVE 7-SIGNAL CROSS-CAMERA SCENARIO EVALUATIONS]")
    topology = CityCameraTopology()
    reid_cfg = ReIDConfig()
    matching_engine = ReIDMatchingEngine(topology=topology, config=reid_cfg)
    graph_engine = TemporalTrafficGraphEngine()
    journey_engine = JourneyReconstructionEngine(
        matching_engine=matching_engine,
        config=reid_cfg,
        temporal_graph_engine=graph_engine,
        journey_timeout_seconds=60.0,
    )

    # Scenarios Definition:
    # 1. Matching Plate, Consistent Travel -> Confirm
    # 2. Unreadable Plate, Matching Appearance/Geometry/Travel -> Probable
    # 3. Conflicting Plate -> Insufficient Evidence / Split ID
    # 4. Physically Impossible Speed (dt < min_t) -> Rejected
    # 5. Time Gap Exceeded (dt > 3600s) -> Rejected
    # 6. Negative Time Delta (dt < 0) -> Rejected
    # 7. Unobserved Gap (Non-adjacent camera transition) -> Confirm with has_unobserved_gap=True
    # 8. Intra-Camera Track Drop Recovery -> Merge
    # 9. Same-Camera Co-presence Exclusion -> No Merge

    scenarios = [
        {
            "name": "Scenario 1: Cross-Camera Re-ID with Confirmed Plate & Valid Travel (CAM_A_EAST -> CAM_B_WEST)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_B_WEST",
            "trk_a": "TRK_101",
            "trk_b": "TRK_201",
            "plate_a": "KA01AB1234",
            "plate_b": "KA01AB1234",
            "plate_conf_a": 0.95,
            "plate_conf_b": 0.94,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 100.0,
            "t_b": 120.0,  # 20s travel time across 300m = 15 m/s = 54 km/h
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.EAST,
            "bbox_a": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "bbox_b": BoundingBoxXYXY(x1=105, y1=102, x2=255, y2=202),
            "emb_a": [0.1] * 128,
            "emb_b": [0.1] * 128,
        },
        {
            "name": "Scenario 2: Cross-Camera Re-ID with Missing/Unreadable Plate via Appearance & Geometry (CAM_A_EAST -> CAM_C_NORTH)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_C_NORTH",
            "trk_a": "TRK_102",
            "trk_b": "TRK_302",
            "plate_a": None,
            "plate_b": None,
            "plate_conf_a": 0.0,
            "plate_conf_b": 0.0,
            "plate_stat_a": "UNKNOWN",
            "plate_stat_b": "UNKNOWN",
            "t_a": 200.0,
            "t_b": 218.0,  # 18s travel time across 250m = 50 km/h
            "type_a": "truck",
            "type_b": "truck",
            "dir_a": Direction.NORTH,
            "dir_b": Direction.NORTH,
            "bbox_a": BoundingBoxXYXY(x1=200, y1=150, x2=450, y2=350),
            "bbox_b": BoundingBoxXYXY(x1=205, y1=152, x2=455, y2=352),
            "emb_a": [0.08] * 128,
            "emb_b": [0.082] * 128,
        },
        {
            "name": "Scenario 3: Different Vehicles with Conflicting Plate Numbers (CAM_A_EAST -> CAM_B_WEST)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_B_WEST",
            "trk_a": "TRK_103",
            "trk_b": "TRK_203",
            "plate_a": "MH12DE3333",
            "plate_b": "DL04XY9999",
            "plate_conf_a": 0.92,
            "plate_conf_b": 0.90,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 300.0,
            "t_b": 322.0,
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.EAST,
            "bbox_a": BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=180),
            "bbox_b": BoundingBoxXYXY(x1=100, y1=100, x2=200, y2=180),
            "emb_a": [0.1] * 128,
            "emb_b": [-0.1] * 128,
        },
        {
            "name": "Scenario 4: Spatio-Temporal Violation — Impossible Speed / Instantaneous Travel (CAM_A_EAST -> CAM_B_WEST)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_B_WEST",
            "trk_a": "TRK_104",
            "trk_b": "TRK_204",
            "plate_a": "KA05JK5555",
            "plate_b": "KA05JK5555",
            "plate_conf_a": 0.95,
            "plate_conf_b": 0.95,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 400.0,
            "t_b": 401.0,  # 1.0s travel across 300m = 300 m/s = 1080 km/h (Physically impossible, min_t=9.0s)
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.EAST,
            "bbox_a": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "bbox_b": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "emb_a": [0.1] * 128,
            "emb_b": [0.1] * 128,
        },
        {
            "name": "Scenario 5: Spatio-Temporal Violation — Exceeded Time Gap / Inactivity (CAM_A_EAST -> CAM_B_WEST)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_B_WEST",
            "trk_a": "TRK_105",
            "trk_b": "TRK_205",
            "plate_a": "TN09ZZ7777",
            "plate_b": "TN09ZZ7777",
            "plate_conf_a": 0.95,
            "plate_conf_b": 0.95,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 500.0,
            "t_b": 4500.0,  # 4000s gap > max_time_gap (3600s)
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.EAST,
            "bbox_a": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "bbox_b": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "emb_a": [0.1] * 128,
            "emb_b": [0.1] * 128,
        },
        {
            "name": "Scenario 6: Spatio-Temporal Violation — Negative Time Delta (CAM_A_EAST -> CAM_B_WEST)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_B_WEST",
            "trk_a": "TRK_106",
            "trk_b": "TRK_206",
            "plate_a": "KA04MM8888",
            "plate_b": "KA04MM8888",
            "plate_conf_a": 0.95,
            "plate_conf_b": 0.95,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 620.0,
            "t_b": 600.0,  # Negative delta t (-20s)
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.EAST,
            "bbox_a": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "bbox_b": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "emb_a": [0.1] * 128,
            "emb_b": [0.1] * 128,
        },
        {
            "name": "Scenario 7: Non-Directly Connected Urban Transition / Unobserved Gap (CAM_A_EAST -> CAM_D_NORTH)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_D_NORTH",  # Not directly connected in topology
            "trk_a": "TRK_107",
            "trk_b": "TRK_407",
            "plate_a": "KA53MN1111",
            "plate_b": "KA53MN1111",
            "plate_conf_a": 0.95,
            "plate_conf_b": 0.95,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 700.0,
            "t_b": 735.0,  # 35s travel across multi-hop path
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.NORTH,
            "bbox_a": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "bbox_b": BoundingBoxXYXY(x1=105, y1=102, x2=255, y2=202),
            "emb_a": [0.1] * 128,
            "emb_b": [0.1] * 128,
        },
        {
            "name": "Scenario 8: Intra-Camera Track Drop & Recovery on Same Camera (CAM_A_EAST -> CAM_A_EAST)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_A_EAST",
            "trk_a": "TRK_108",
            "trk_b": "TRK_109",  # New track assigned after temporary occlusion
            "plate_a": "KA02CD2222",
            "plate_b": "KA02CD2222",
            "plate_conf_a": 0.90,
            "plate_conf_b": 0.92,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 800.0,
            "t_b": 804.0,  # 4s later on same camera
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.EAST,
            "bbox_a": BoundingBoxXYXY(x1=100, y1=100, x2=250, y2=200),
            "bbox_b": BoundingBoxXYXY(x1=180, y1=100, x2=330, y2=200),
            "emb_a": [0.1] * 128,
            "emb_b": [0.1] * 128,
        },
        {
            "name": "Scenario 9: Same-Camera Co-Presence Mutual Exclusion (CAM_A_EAST -> CAM_A_EAST)",
            "cam_a": "CAM_A_EAST",
            "cam_b": "CAM_A_EAST",
            "trk_a": "TRK_110",
            "trk_b": "TRK_111",  # Simultaneous distinct vehicle at same timestamp
            "plate_a": "KA03EE3333",
            "plate_b": "KA03FF4444",
            "plate_conf_a": 0.90,
            "plate_conf_b": 0.90,
            "plate_stat_a": "CONFIRMED",
            "plate_stat_b": "CONFIRMED",
            "t_a": 900.0,
            "t_b": 900.0,  # Exactly simultaneous
            "type_a": "car",
            "type_b": "car",
            "dir_a": Direction.EAST,
            "dir_b": Direction.EAST,
            "bbox_a": BoundingBoxXYXY(x1=50, y1=100, x2=150, y2=200),
            "bbox_b": BoundingBoxXYXY(x1=250, y1=100, x2=350, y2=200),
            "emb_a": [0.1] * 128,
            "emb_b": [0.1] * 128,
        },
    ]

    for idx, sc in enumerate(scenarios, 1):
        print(f"\n--- Running {sc['name']} ---")

        # Step A: Feed observation A
        ev_a = VehicleIdentityEvidence(
            track_id=sc["trk_a"],
            camera_id=sc["cam_a"],
            vehicle_type=sc["type_a"],
            last_updated_timestamp=sc["t_a"],
            timestamp_uncertainty_seconds=0.05,
        )
        if sc["plate_a"]:
            ev_a.associated_plate = FusedPlateIdentity(
                track_id=sc["trk_a"],
                camera_id=sc["cam_a"],
                best_plate_number=sc["plate_a"],
                overall_confidence=sc["plate_conf_a"],
                status=sc["plate_stat_a"],
                confirmed=(sc["plate_stat_a"] == "CONFIRMED"),
                first_seen_timestamp=sc["t_a"],
                last_seen_timestamp=sc["t_a"],
            )

        trk_obj_a = TrackState(
            track_id=sc["trk_a"],
            camera_id=sc["cam_a"],
            vehicle_type=sc["type_a"],
            current_bbox=sc["bbox_a"],
            current_center=sc["bbox_a"].center,
            direction=sc["dir_a"],
            speed_estimate=45.0,
            confidence=0.95,
            first_seen_timestamp=sc["t_a"],
            last_seen_timestamp=sc["t_a"],
        )

        journey_a = journey_engine.process_track_evidence(evidence=ev_a, track=trk_obj_a)
        if sc["emb_a"]:
            journey_engine.track_embeddings[sc["trk_a"]] = (sc["t_a"], sc["emb_a"])
            if journey_a.segments:
                journey_a.segments[-1].appearance_embedding = sc["emb_a"]

        # Step B: Direct 7-signal evaluation between Observation A and Observation B
        ev_b = VehicleIdentityEvidence(
            track_id=sc["trk_b"],
            camera_id=sc["cam_b"],
            vehicle_type=sc["type_b"],
            last_updated_timestamp=sc["t_b"],
            timestamp_uncertainty_seconds=0.05,
        )
        if sc["plate_b"]:
            ev_b.associated_plate = FusedPlateIdentity(
                track_id=sc["trk_b"],
                camera_id=sc["cam_b"],
                best_plate_number=sc["plate_b"],
                overall_confidence=sc["plate_conf_b"],
                status=sc["plate_stat_b"],
                confirmed=(sc["plate_stat_b"] == "CONFIRMED"),
                first_seen_timestamp=sc["t_b"],
                last_seen_timestamp=sc["t_b"],
            )

        trk_obj_b = TrackState(
            track_id=sc["trk_b"],
            camera_id=sc["cam_b"],
            vehicle_type=sc["type_b"],
            current_bbox=sc["bbox_b"],
            current_center=sc["bbox_b"].center,
            direction=sc["dir_b"],
            speed_estimate=45.0,
            confidence=0.95,
            first_seen_timestamp=sc["t_b"],
            last_seen_timestamp=sc["t_b"],
        )

        if sc["emb_b"]:
            journey_engine.track_embeddings[sc["trk_b"]] = (sc["t_b"], sc["emb_b"])

        breakdown = matching_engine.compute_match_score(
            evidence_a=ev_a,
            track_a=trk_obj_a,
            evidence_b=ev_b,
            track_b=trk_obj_b,
            embedding_a=sc["emb_a"],
            embedding_b=sc["emb_b"],
        )

        # Step C: Feed observation B into journey engine
        journey_b = journey_engine.process_track_evidence(evidence=ev_b, track=trk_obj_b)

        last_seg = journey_b.segments[-1] if journey_b.segments else None
        has_gap = getattr(last_seg, "has_unobserved_gap", False)

        rep_item = {
            "scenario_index": idx,
            "scenario_name": sc["name"],
            "camera_a": sc["cam_a"],
            "camera_b": sc["cam_b"],
            "track_id_a": sc["trk_a"],
            "track_id_b": sc["trk_b"],
            "plate_a": sc["plate_a"] or "UNKNOWN",
            "plate_b": sc["plate_b"] or "UNKNOWN",
            "plate_similarity": breakdown.plate_similarity,
            "appearance_similarity": breakdown.appearance_similarity,
            "visual_feature_similarity": breakdown.visual_features_similarity,
            "vehicle_type_similarity": breakdown.vehicle_type_similarity,
            "movement_direction_similarity": breakdown.direction_similarity,
            "temporal_compatibility": breakdown.temporal_compatibility,
            "spatial_compatibility": breakdown.spatial_compatibility,
            "weighted_final_score": breakdown.overall_score,
            "decision": breakdown.decision.value if hasattr(breakdown.decision, "value") else str(breakdown.decision),
            "rejection_reason": breakdown.rejection_reason,
            "global_vehicle_id_a": journey_a.global_vehicle_id,
            "global_vehicle_id_b": journey_b.global_vehicle_id,
            "same_global_vehicle": (journey_a.global_vehicle_id == journey_b.global_vehicle_id),
            "journey_id": journey_b.journey_id,
            "camera_sequence": list(journey_b.cameras),
            "timestamp_sequence": [seg.timestamp for seg in journey_b.segments],
            "has_unobserved_gap": has_gap,
        }

        results["scenarios"].append(rep_item)

        print(f"  Comparison: {sc['cam_a']} [{sc['trk_a']}] vs {sc['cam_b']} [{sc['trk_b']}]")
        print(f"  Plates: '{sc['plate_a']}' vs '{sc['plate_b']}'")
        print(f"  Scores: Plate={breakdown.plate_similarity:.2f}, App={breakdown.appearance_similarity:.2f}, Geom={breakdown.visual_features_similarity:.2f}, Type={breakdown.vehicle_type_similarity:.2f}, Dir={breakdown.direction_similarity:.2f}, Time={breakdown.temporal_compatibility:.2f}, Space={breakdown.spatial_compatibility:.2f}")
        print(f"  Final Weighted Score: {breakdown.overall_score:.3f} | Decision: {breakdown.decision.value} (Rejection: {breakdown.rejection_reason})")
        print(f"  Global Vehicle ID: {journey_b.global_vehicle_id} (Merged: {journey_a.global_vehicle_id == journey_b.global_vehicle_id}) | Journey: {journey_b.journey_id} | Gap: {has_gap}")

    # -------------------------------------------------------------------------
    # 4. Module 3 Downstream Temporal Graph Handoff Verification
    # -------------------------------------------------------------------------
    print("\n[4. MODULE 3 TEMPORAL GRAPH HANDOFF VERIFICATION]")
    snapshot = graph_engine.snapshot(timestamp=1000.0)
    print(f"  Graph Nodes Count: {snapshot['node_count']}")
    print(f"  Graph Edges Count: {snapshot['edge_count']}")
    print(f"  Vehicles in Graph: {len(snapshot['vehicles'])}")
    print(f"  Cameras in Graph: {len(snapshot['cameras'])}")

    transition_edges = []
    for u, v, k, d in graph_engine.builder.graph.edges(keys=True, data=True):
        if d.get("type") == EdgeType.VEHICLE_TRANSITIONS_TO_CAMERA:
            transition_edges.append({
                "from_cam": u,
                "to_cam": v,
                "metadata": d.get("metadata", {}),
                "speed": d.get("speed"),
                "travel_time": d.get("travel_time"),
            })

    print(f"  Vehicle Transition Edges in Graph: {len(transition_edges)}")
    for te in transition_edges[:3]:
        print(f"    Edge: {te['from_cam']} -> {te['to_cam']} (Speed: {te['speed']} km/h, Time: {te['travel_time']}s, Meta: {list(te['metadata'].keys())})")

    results["module3_graph_handoff"] = {
        "node_count": snapshot["node_count"],
        "edge_count": snapshot["edge_count"],
        "vehicle_nodes_count": len(snapshot["vehicles"]),
        "camera_nodes_count": len(snapshot["cameras"]),
        "transition_edges_count": len(transition_edges),
        "transition_edge_samples": transition_edges[:3],
    }

    # Save validation output JSON
    out_json_path = os.path.join("scratch", "point_f_validation_results.json")
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Validation JSON saved to: {out_json_path}]")
    print("=" * 80)
    print("MODULE 2 VALIDATION RUN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_point_f_validation()
