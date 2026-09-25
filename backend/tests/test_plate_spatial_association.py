"""
Unit & Integration tests for Point F - Vehicle <-> Plate Spatial Association Protection.
Verifies spatial ownership resolution, foreground/background overlap handling,
duplicate candidate rejection, and TRK_117 scenario validation.
"""

import pytest
import numpy as np
from app.schemas.detection import BoundingBoxXYXY
from app.schemas.tracking import TrackState
from app.schemas.plate import PlateValidationStatus
from app.perception.plate_association import (
    compute_bbox_intersection_area,
    compute_plate_containment_fraction,
    compute_plate_vehicle_iou,
    resolve_plate_ownership,
    PlateTrackerAssociationManager,
)
from app.perception.ocr_engine import BaseOCREngine


class MockOCREngine(BaseOCREngine):
    """Mock OCR engine for deterministic test assertions."""

    def __init__(self, text_map=None, default_text="TN09AB1234", default_conf=0.92):
        self.text_map = text_map or {}
        self.default_text = default_text
        self.default_conf = default_conf

    def recognize_text(self, variants):
        # Return mapped text if found
        for v in variants:
            if isinstance(v, dict) and "sim_plate" in v:
                return v["sim_plate"], v["sim_plate"], 0.95, "SIM"
        return self.default_text, self.default_text, self.default_conf, "MOCK"


def make_track(track_id: str, bbox: BoundingBoxXYXY, cam_id: str = "CAM_01") -> TrackState:
    """Helper to construct TrackState."""
    return TrackState(
        track_id=track_id,
        camera_id=cam_id,
        current_bbox=bbox,
        first_frame=1,
        last_frame=10,
        total_detections=10,
        confirmed=True,
    )


class TestPlateSpatialAssociation:
    """Focused tests for spatial association rules (Point F)."""

    def test_1_one_plate_one_vehicle_unchanged(self):
        """Rule 1 & 2: Single vehicle containing single plate is assigned cleanly without modification."""
        vehicle = make_track("TRK_V1", BoundingBoxXYXY(x1=100.0, y1=100.0, x2=500.0, y2=400.0))
        plate = BoundingBoxXYXY(x1=200.0, y1=250.0, x2=350.0, y2=300.0)

        owner = resolve_plate_ownership(plate, [vehicle])
        assert owner is not None
        assert owner.track_id == "TRK_V1"

    def test_2_one_plate_two_overlapping_vehicles_assigned_to_one(self):
        """Rule 3: One plate inside two overlapping vehicle boxes is assigned to exactly one vehicle."""
        # Vehicle 1 is a compact car in foreground (area: 300x200 = 60,000)
        v1 = make_track("TRK_CAR", BoundingBoxXYXY(x1=100.0, y1=100.0, x2=400.0, y2=300.0))
        # Vehicle 2 is a large truck overlapping behind it (area: 800x600 = 480,000)
        v2 = make_track("TRK_TRUCK", BoundingBoxXYXY(x1=50.0, y1=50.0, x2=850.0, y2=650.0))

        # Plate located squarely on the car
        plate = BoundingBoxXYXY(x1=200.0, y1=220.0, x2=320.0, y2=260.0)

        owner = resolve_plate_ownership(plate, [v1, v2])
        assert owner is not None
        # TRK_CAR has much higher IoU with the plate than the massive TRK_TRUCK
        assert owner.track_id == "TRK_CAR"

    def test_3_foreground_plate_overlapping_background_bbox(self):
        """Rule 3: Foreground vehicle plate overlapping background vehicle bbox is assigned to foreground."""
        # Foreground car: (100, 200, 400, 500)
        fg_car = make_track("TRK_FG", BoundingBoxXYXY(x1=100.0, y1=200.0, x2=400.0, y2=500.0))
        # Background bus: (50, 100, 700, 450) overlaps upper half of foreground car
        bg_bus = make_track("TRK_BG", BoundingBoxXYXY(x1=50.0, y1=100.0, x2=700.0, y2=450.0))

        # Plate is on the foreground car's front/rear bumper (200, 380, 320, 430)
        # It is inside fg_car (containment=1.0) and inside bg_bus (containment=1.0)
        plate = BoundingBoxXYXY(x1=200.0, y1=380.0, x2=320.0, y2=430.0)

        owner = resolve_plate_ownership(plate, [bg_bus, fg_car])
        assert owner is not None
        assert owner.track_id == "TRK_FG"

    def test_4_two_plates_two_overlapping_vehicles(self):
        """Rule 3 & 4: Two distinct plates on two overlapping vehicles resolve to their respective owners."""
        # Car 1 on the left: (50, 100, 350, 400), plate at (150, 300, 270, 340)
        car1 = make_track("TRK_1", BoundingBoxXYXY(x1=50.0, y1=100.0, x2=350.0, y2=400.0))
        plate1 = BoundingBoxXYXY(x1=150.0, y1=300.0, x2=270.0, y2=340.0)

        # Car 2 on the right, overlapping car 1: (250, 100, 550, 400), plate at (380, 300, 500, 340)
        car2 = make_track("TRK_2", BoundingBoxXYXY(x1=250.0, y1=100.0, x2=550.0, y2=400.0))
        plate2 = BoundingBoxXYXY(x1=380.0, y1=300.0, x2=500.0, y2=340.0)

        owner1 = resolve_plate_ownership(plate1, [car1, car2])
        owner2 = resolve_plate_ownership(plate2, [car1, car2])

        assert owner1 is not None and owner1.track_id == "TRK_1"
        assert owner2 is not None and owner2.track_id == "TRK_2"

    def test_5_plate_outside_all_vehicle_boxes(self):
        """Rule 4: Plate outside all vehicle bounding boxes returns None."""
        v1 = make_track("TRK_1", BoundingBoxXYXY(x1=100.0, y1=100.0, x2=300.0, y2=300.0))
        outside_plate = BoundingBoxXYXY(x1=500.0, y1=500.0, x2=600.0, y2=550.0)

        owner = resolve_plate_ownership(outside_plate, [v1])
        assert owner is None

    def test_6_tie_or_ambiguous_overlap_no_duplicate(self):
        """Rule 3 & 4: Ambiguous / tied overlap resolves deterministically without duplication."""
        v1 = make_track("TRK_A", BoundingBoxXYXY(x1=100.0, y1=100.0, x2=300.0, y2=300.0))
        v2 = make_track("TRK_B", BoundingBoxXYXY(x1=100.0, y1=100.0, x2=300.0, y2=300.0))
        plate = BoundingBoxXYXY(x1=150.0, y1=150.0, x2=250.0, y2=200.0)

        owner = resolve_plate_ownership(plate, [v1, v2])
        assert owner is not None
        assert owner.track_id in ("TRK_A", "TRK_B")

    def test_7_containment_fraction_primary_over_iou(self):
        """Plate partially inside Vehicle A (30%) vs fully inside Vehicle B (100%): Vehicle B wins on containment."""
        v_a = make_track("TRK_A", BoundingBoxXYXY(x1=0.0, y1=0.0, x2=150.0, y2=150.0))
        v_b = make_track("TRK_B", BoundingBoxXYXY(x1=100.0, y1=0.0, x2=400.0, y2=300.0))

        # Plate at (120, 50, 220, 90) -> x from 120 to 220 (width 100).
        # Inside v_a: x from 120 to 150 (width 30 -> 30% containment).
        # Inside v_b: x from 120 to 220 (width 100 -> 100% containment).
        plate = BoundingBoxXYXY(x1=120.0, y1=50.0, x2=220.0, y2=90.0)

        owner = resolve_plate_ownership(plate, [v_a, v_b])
        assert owner is not None
        assert owner.track_id == "TRK_B"


class TestTRK117TargetedReproduction:
    """
    Targeted geometry reproduction of the TRK_117 scenario.
    In frame 237, TRK_117 (background taxi) overlapped with the foreground taxi (rear plate JZ8479).
    Verify that JZ8479 is NOT assigned to TRK_117.
    """

    def test_trk117_jz8479_cross_contamination_prevented(self):
        # TRK_117: background vehicle bbox (54, 538, 450, 990)
        trk_117 = make_track("TRK_117", BoundingBoxXYXY(x1=54.0, y1=538.0, x2=450.0, y2=990.0))

        # TRK_FG: foreground taxi crossing on the left (0, 500, 180, 950)
        trk_fg = make_track("TRK_FG", BoundingBoxXYXY(x1=0.0, y1=500.0, x2=180.0, y2=950.0))

        # Plate JZ8479: rear plate of foreground taxi at (105, 850, 160, 915)
        # Notice: JZ8479 is inside TRK_FG (x: 105..160 <= 180) and inside TRK_117 (x: 105..160 >= 54)
        plate_jz8479 = BoundingBoxXYXY(x1=105.0, y1=850.0, x2=160.0, y2=915.0)

        # TRK_117's own plate WG2119: located on TRK_117's front bumper (300, 750, 420, 810)
        plate_wg2119 = BoundingBoxXYXY(x1=300.0, y1=750.0, x2=420.0, y2=810.0)

        all_tracks = [trk_117, trk_fg]

        # 1. Resolve JZ8479 ownership
        owner_jz = resolve_plate_ownership(plate_jz8479, all_tracks)
        assert owner_jz is not None
        assert owner_jz.track_id == "TRK_FG", "JZ8479 must be assigned to TRK_FG, NOT TRK_117"
        assert owner_jz.track_id != "TRK_117", "TRK_117 must NOT claim JZ8479"

        # 2. Resolve WG2119 ownership
        owner_wg = resolve_plate_ownership(plate_wg2119, all_tracks)
        assert owner_wg is not None
        assert owner_wg.track_id == "TRK_117", "WG2119 must be assigned to TRK_117"

    def test_pipeline_multi_track_batch_isolation(self):
        """End-to-end test with PlateTrackerAssociationManager on overlapping tracks."""
        mgr = PlateTrackerAssociationManager(
            ocr_engine=MockOCREngine(default_text="TN09AB1234", default_conf=0.92)
        )

        trk_117 = make_track("TRK_117", BoundingBoxXYXY(x1=54.0, y1=538.0, x2=450.0, y2=990.0))
        trk_fg = make_track("TRK_FG", BoundingBoxXYXY(x1=0.0, y1=500.0, x2=180.0, y2=950.0))
        tracks = [trk_fg, trk_117]

        # Synthetic frame
        synthetic_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128

        # Process frame with multi-track spatial protection
        evidence_map = mgr.process_frame_tracks(
            tracks=tracks,
            frame=synthetic_frame,
            timestamp=1.0,
            frame_id=237,
            img_w=1920,
            img_h=1080,
        )

        assert "TRK_117" in evidence_map
        assert "TRK_FG" in evidence_map
