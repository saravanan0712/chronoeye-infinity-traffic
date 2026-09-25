"""
ChronoEye Infinity Verification Runner
Executes test_video_traffic_extraction_engine.py directly and prints test results.
"""

import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests.test_video_traffic_extraction_engine import (
    test_video_ingestion_service_synthetic,
    test_video_preprocessing_and_roi,
    test_vehicle_detector_filtering,
    test_multi_object_tracking_and_trajectories,
    test_camera_speed_calibration,
    test_virtual_line_counting_and_direction,
    test_traffic_state_engine_master_observation,
    test_video_trust_bridge,
    test_simulation_fallback_adapter,
    test_realtime_api_endpoints,
)

if __name__ == "__main__":
    tests = [
        ("Video Ingestion & Metadata", test_video_ingestion_service_synthetic),
        ("Video Preprocessing & ROI", test_video_preprocessing_and_roi),
        ("Vehicle Detector & Filtering", test_vehicle_detector_filtering),
        ("Multi-Object Tracking & Trajectories", test_multi_object_tracking_and_trajectories),
        ("Camera Speed Calibration", test_camera_speed_calibration),
        ("Virtual Line Counting & Direction", test_virtual_line_counting_and_direction),
        ("Traffic State Engine & Master Observation", test_traffic_state_engine_master_observation),
        ("Video Trust Bridge", test_video_trust_bridge),
        ("Simulation Fallback Adapter", test_simulation_fallback_adapter),
        ("Real-time API Endpoints", test_realtime_api_endpoints),
    ]

    print("============================================================")
    print("CHRONOEYE INFINITY - VIDEO-TO-TRAFFIC ENGINE VERIFICATION")
    print("============================================================")

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("------------------------------------------------------------")
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED / TOTAL {len(tests)}")
    print("============================================================")

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
