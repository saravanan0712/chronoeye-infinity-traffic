"""
ChronoEye Infinity - Phase 4: License Plate Candidate ROI Detector
Extracts vehicle region-of-interest (ROI) crops from image frames, performs boundary validation,
applies aspect-ratio and area candidate filtering, and locates license plate ROI crops.
"""

from typing import List, Tuple, Optional, Any
from app.schemas.detection import BoundingBoxXYXY


class PlateDetector:
    """
    Modular License Plate Candidate ROI Extractor.
    Extracts vehicle crops from image frames and locates plate candidates using a dedicated
    YOLO License Plate Detector model or precise spatial priority constraints.
    """

    def __init__(
        self,
        min_aspect_ratio: float = 0.75,
        max_aspect_ratio: float = 6.5,
        model_path: str = "backend/models/license_plate_detector.pt",
        conf_threshold: float = 0.25,
    ):
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.plate_model = None
        self._is_yolo_available = False

        # Diagnostic counters for false-positive filtering
        self.fp_rejection_stats = {
            "yolo_accepted": 0,
            "yolo_geometry_rejected": 0,
            "fallback_accepted": 0,
            "fallback_too_small_rejected": 0,
            "fallback_dark_rejected": 0,
            "fallback_low_contrast_rejected": 0,
            "fallback_geometry_rejected": 0,
        }

        self._initialize_plate_model()

    def _initialize_plate_model(self):
        """Attempts to load or download Ultralytics YOLO dedicated license plate detector model."""
        import os
        import urllib.request

        m_path = self.model_path
        if not os.path.isabs(m_path):
            possible_paths = [
                os.path.abspath(m_path),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", m_path)),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", m_path)),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", os.path.basename(m_path))),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    m_path = p
                    break

        target_dir = os.path.dirname(os.path.abspath(m_path))
        os.makedirs(target_dir, exist_ok=True)

        # Download dedicated license plate detector model weights if not present locally
        if not os.path.exists(m_path):
            urls = [
                "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/best.pt",
                "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
            ]
            for url in urls:
                try:
                    print(f"[PlateDetector] Downloading dedicated license plate model weights from {url} ...")
                    req = urllib.request.Request(
                        url,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    with urllib.request.urlopen(req, timeout=45) as response, open(m_path, 'wb') as out_file:
                        out_file.write(response.read())
                    print(f"[PlateDetector] Successfully downloaded model weights to {m_path}")
                    break
                except Exception as e:
                    print(f"[PlateDetector WARNING] Could not download weights from {url}: {e}")
                    if os.path.exists(m_path):
                        try:
                            os.remove(m_path)
                        except Exception:
                            pass

        if os.path.exists(m_path):
            try:
                from ultralytics import YOLO
                self.plate_model = YOLO(m_path)
                self._is_yolo_available = True
                self.model_path = m_path
                print(f"[PlateDetector] Loaded dedicated plate model: {m_path}")
                return
            except Exception as e:
                print(f"[PlateDetector ERROR] Failed loading model at {m_path}: {e}")

        print(f"[PlateDetector ERROR] Dedicated plate model file not found at '{m_path}' and download failed.")
        self.plate_model = None
        self._is_yolo_available = False

    def validate_roi_bounds(
        self, bbox: BoundingBoxXYXY, img_w: int = 1920, img_h: int = 1080
    ) -> bool:
        """Validates crop boundaries."""
        if bbox.x1 < 0 or bbox.y1 < 0:
            return False
        if bbox.x2 > img_w or bbox.y2 > img_h:
            return False
        if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
            return False
        if bbox.area <= 0:
            return False
        return True

    def calculate_roi_quality(
        self,
        plate_crop: Any,
        bbox: BoundingBoxXYXY,
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> float:
        """
        Calculates a deterministic ROI quality score (0.0 to 1.0) based on:
        width, height, aspect ratio, brightness, contrast, sharpness (Laplacian variance),
        edge density, and border clipping percentage.
        Does NOT determine plate text.
        """
        if not self.validate_roi_bounds(bbox, img_w, img_h):
            return 0.0

        w = bbox.width
        h = bbox.height
        if h <= 0 or w <= 0:
            return 0.0

        area = w * h
        aspect_ratio = w / float(h)

        # 1. Aspect Ratio Score (Target 1.8 to 6.5)
        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            aspect_score = 1.0 - abs(aspect_ratio - 3.5) / 3.5
            aspect_score = max(0.2, min(1.0, aspect_score))
        else:
            aspect_score = 0.05

        # 2. Size / Usable Area Score (Target > 600 px area)
        size_score = min(1.0, area / 1500.0)

        # Default image quality metrics for fallback/mock objects
        brightness_score = 0.5
        contrast_score = 0.5
        sharpness_score = 0.5
        edge_score = 0.5

        try:
            import cv2
            import numpy as np

            if isinstance(plate_crop, np.ndarray) and plate_crop.size > 0:
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop

                # Brightness (Target mean 40..210)
                mean_val = float(np.mean(gray))
                if 40.0 <= mean_val <= 210.0:
                    brightness_score = 1.0 - abs(mean_val - 125.0) / 125.0
                else:
                    brightness_score = 0.1

                # Contrast (Std Dev, target > 20)
                std_val = float(np.std(gray))
                contrast_score = min(1.0, std_val / 40.0)

                # Sharpness (Laplacian Variance, target > 30)
                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_score = min(1.0, laplacian_var / 100.0)

                # Edge Density (Canny edges ratio)
                edges = cv2.Canny(gray, 50, 150)
                edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
                edge_score = min(1.0, edge_ratio * 4.0)
        except Exception:
            pass

        # Weighted quality score combination
        quality = (
            0.25 * sharpness_score
            + 0.20 * contrast_score
            + 0.20 * size_score
            + 0.15 * aspect_score
            + 0.10 * brightness_score
            + 0.10 * edge_score
        )
        return round(max(0.0, min(1.0, quality)), 3)

    def correct_perspective(self, plate_crop: Any) -> Tuple[Any, bool]:
        """
        Performs perspective normalization if reliable 4-corner trapezoidal plate geometry is found.
        Converts angled plate regions into a normalized rectangular representation (200x50).
        If reliable corners are unavailable, safely returns original crop and False.
        """
        try:
            import cv2
            import numpy as np

            if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
                return plate_crop, False

            h, w = plate_crop.shape[:2]
            if h < 10 or w < 30:
                return plate_crop, False

            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_quad_area = 0.25 * (w * h)

            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                area = cv2.contourArea(cnt)
                if area < min_quad_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                if len(approx) == 4:
                    pts = approx.reshape(4, 2)

                    # Sort points: top-left, top-right, bottom-right, bottom-left
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]  # top-left
                    rect[2] = pts[np.argmax(s)]  # bottom-right

                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]  # top-right
                    rect[3] = pts[np.argmax(diff)]  # bottom-left

                    dst_w = 200
                    dst_h = 50
                    dst = np.array(
                        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                        dtype="float32",
                    )

                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(plate_crop, M, (dst_w, dst_h))
                    return warped, True
        except Exception:
            pass

        return plate_crop, False

    def extract_vehicle_crop(
        self, frame: Any, vehicle_bbox: BoundingBoxXYXY, img_w: int = 1920, img_h: int = 1080
    ) -> Tuple[bool, Optional[Any], BoundingBoxXYXY]:
        """
        Crops vehicle ROI from image frame with boundary validation.
        Validates numeric finiteness, coordinate ordering, and clamps bounds BEFORE constructing BoundingBoxXYXY.
        """
        import math
        try:
            raw_x1 = float(vehicle_bbox.x1)
            raw_y1 = float(vehicle_bbox.y1)
            raw_x2 = float(vehicle_bbox.x2)
            raw_y2 = float(vehicle_bbox.y2)
        except (ValueError, TypeError, AttributeError):
            return False, None, vehicle_bbox

        if not (math.isfinite(raw_x1) and math.isfinite(raw_y1) and math.isfinite(raw_x2) and math.isfinite(raw_y2)):
            return False, None, vehicle_bbox

        # Clamp raw coordinates to image boundaries
        x1 = max(0, min(img_w, int(raw_x1)))
        y1 = max(0, min(img_h, int(raw_y1)))
        x2 = max(0, min(img_w, int(raw_x2)))
        y2 = max(0, min(img_h, int(raw_y2)))

        # Verify positive width and height (x2 > x1 and y2 > y1)
        if x2 <= x1 or y2 <= y1:
            return False, None, vehicle_bbox

        validated_bbox = BoundingBoxXYXY(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        if not self.validate_roi_bounds(validated_bbox, img_w, img_h):
            return False, None, vehicle_bbox

        try:
            import numpy as np
            if isinstance(frame, np.ndarray):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    return False, None, validated_bbox
                return True, crop, validated_bbox
        except ImportError:
            pass

        # Fallback dictionary frame representation
        return True, {"type": "vehicle_crop", "bbox": validated_bbox}, validated_bbox

    def extract_plate_candidates(
        self, vehicle_crop: Any, vehicle_bbox: BoundingBoxXYXY
    ) -> List[Tuple[Any, BoundingBoxXYXY]]:
        """
        Extracts candidate license plate ROI crops from vehicle ROI using dedicated YOLO detector
        if loaded, or precise spatial priority constraints fallback.
        """
        candidates: List[Tuple[Any, BoundingBoxXYXY]] = []
        v_w = vehicle_bbox.width
        v_h = vehicle_bbox.height
        # Flag: True when a real NumPy vehicle crop was processed.
        # When True, a rejected fallback strip must return [] — not a synthetic candidate.
        _numpy_path_ran = False

        try:
            import numpy as np

            if isinstance(vehicle_crop, np.ndarray) and vehicle_crop.size > 0:
                _numpy_path_ran = True

                # 1. Run Dedicated YOLO Plate Detector if available
                if self._is_yolo_available and self.plate_model is not None:
                    try:
                        results = self.plate_model(vehicle_crop, conf=self.conf_threshold, verbose=False)
                        if results and len(results) > 0 and results[0].boxes is not None:
                            boxes = results[0].boxes
                            for box in boxes:
                                conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                                
                                raw_box = box.xyxy[0]
                                if hasattr(raw_box, "tolist"):
                                    xyxy = raw_box.tolist()
                                elif isinstance(raw_box, (list, tuple)):
                                    xyxy = list(raw_box)
                                else:
                                    xyxy = list(box.xyxy)
                                
                                px1, py1, px2, py2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                                h_crop, w_crop = vehicle_crop.shape[:2]
                                v_w, v_h = float(w_crop), float(h_crop)
                                
                                # Clip to vehicle crop bounds
                                px1 = max(0, min(w_crop, px1))
                                py1 = max(0, min(h_crop, py1))
                                px2 = max(0, min(w_crop, px2))
                                py2 = max(0, min(h_crop, py2))

                                box_w = px2 - px1
                                box_h = py2 - py1
                                
                                # 1. Minimum Size Check (Unconditionally minimum 40x12)
                                if box_w < 40 or box_h < 12:
                                    self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                    continue

                                # 2. Aspect Ratio Check (0.75 <= aspect <= 6.0)
                                aspect = box_w / float(max(1, box_h))
                                if aspect < 0.75 or aspect > 6.0:
                                    self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                    continue
                                
                                # 3 & 4. Relative Vehicle Width and Height (w <= 60% and h <= 40%)
                                if box_w > v_w * 0.60 or box_h > v_h * 0.40:
                                    self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                    continue

                                # 5. Relative Area (<= 20% of vehicle crop)
                                if (box_w * box_h) > (v_w * v_h * 0.20):
                                    self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                    continue
                                
                                # 6. Location Component (generally lower/central; center Y must be >= 25% of vehicle height, avoid roof)
                                cy = py1 + (box_h / 2.0)
                                if cy < v_h * 0.25:
                                    self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                    continue

                                # Add 8% horizontal and 12% vertical margin padding around YOLO box
                                pad_w = max(4, int(box_w * 0.08))
                                pad_h = max(3, int(box_h * 0.12))
                                px1 = max(0, px1 - pad_w)
                                py1 = max(0, py1 - pad_h)
                                px2 = min(w_crop, px2 + pad_w)
                                py2 = min(h_crop, py2 + pad_h)

                                plate_crop = vehicle_crop[py1:py2, px1:px2]
                                global_pbox = BoundingBoxXYXY(
                                    x1=vehicle_bbox.x1 + px1,
                                    y1=vehicle_bbox.y1 + py1,
                                    x2=vehicle_bbox.x1 + px2,
                                    y2=vehicle_bbox.y1 + py2
                                )
                                candidates.append((plate_crop, global_pbox))
                                self.fp_rejection_stats["yolo_accepted"] += 1
                    except Exception:
                        pass

                # 2. Targeted Secondary Recovery Pass on Lower Vehicle ROI (if primary pass found no candidates)
                if not candidates and self._is_yolo_available and self.plate_model is not None:
                    try:
                        h_crop, w_crop = vehicle_crop.shape[:2]
                        v_w, v_h = float(w_crop), float(h_crop)
                        if v_h >= 40 and v_w >= 40:
                            y_offset = int(v_h * 0.35)
                            lower_roi = vehicle_crop[y_offset:, :]
                            rec_results = self.plate_model(lower_roi, conf=0.15, verbose=False)
                            if rec_results and len(rec_results) > 0 and rec_results[0].boxes is not None:
                                for box in rec_results[0].boxes:
                                    raw_box = box.xyxy[0]
                                    if hasattr(raw_box, "tolist"):
                                        xyxy = raw_box.tolist()
                                    elif isinstance(raw_box, (list, tuple)):
                                        xyxy = list(raw_box)
                                    else:
                                        xyxy = list(box.xyxy)
                                    
                                    r_px1, r_py1, r_px2, r_py2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                    px1 = max(0, min(w_crop, r_px1))
                                    py1 = max(0, min(h_crop, r_py1 + y_offset))
                                    px2 = max(0, min(w_crop, r_px2))
                                    py2 = max(0, min(h_crop, r_py2 + y_offset))
                                    box_w = px2 - px1
                                    box_h = py2 - py1

                                    if box_w < 40 or box_h < 12:
                                        self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                        continue
                                    aspect = box_w / float(max(1, box_h))
                                    if aspect < 0.75 or aspect > 6.0:
                                        self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                        continue
                                    if box_w > v_w * 0.55 or box_h > v_h * 0.35:
                                        self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                        continue
                                    if (box_w * box_h) > (v_w * v_h * 0.18):
                                        self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                        continue
                                    cy = py1 + (box_h / 2.0)
                                    if cy < v_h * 0.30:
                                        self.fp_rejection_stats["yolo_geometry_rejected"] += 1
                                        continue

                                    pad_w = max(4, int(box_w * 0.08))
                                    pad_h = max(3, int(box_h * 0.12))
                                    px1 = max(0, px1 - pad_w)
                                    py1 = max(0, py1 - pad_h)
                                    px2 = min(w_crop, px2 + pad_w)
                                    py2 = min(h_crop, py2 + pad_h)

                                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                                    global_pbox = BoundingBoxXYXY(
                                        x1=vehicle_bbox.x1 + px1,
                                        y1=vehicle_bbox.y1 + py1,
                                        x2=vehicle_bbox.x1 + px2,
                                        y2=vehicle_bbox.y1 + py2
                                    )
                                    candidates.append((plate_crop, global_pbox))
                                    self.fp_rejection_stats["yolo_accepted"] += 1
                                    break
                    except Exception:
                        pass

                # 3. Contrast & Morphology-Guided Plate Band Localizer (if YOLO passes found no candidate)
                if not candidates:
                    try:
                        import cv2 as _cv2
                        h_crop, w_crop = vehicle_crop.shape[:2]
                        v_w, v_h = float(w_crop), float(h_crop)
                        if v_h >= 30 and v_w >= 50:
                            s_y1, s_y2 = int(v_h * 0.45), int(v_h * 0.90)
                            s_x1, s_x2 = int(v_w * 0.15), int(v_w * 0.85)
                            search_roi = vehicle_crop[s_y1:s_y2, s_x1:s_x2]
                            s_h, s_w = search_roi.shape[:2]
                            if s_h >= 20 and s_w >= 40:
                                gray_s = _cv2.cvtColor(search_roi, _cv2.COLOR_BGR2GRAY) if len(search_roi.shape) == 3 else search_roi
                                grad_x = _cv2.Sobel(gray_s, _cv2.CV_16S, 1, 0, ksize=3)
                                abs_grad_x = _cv2.convertScaleAbs(grad_x)
                                kernel_m = _cv2.getStructuringElement(_cv2.MORPH_RECT, (17, 3))
                                closed_m = _cv2.morphologyEx(abs_grad_x, _cv2.MORPH_CLOSE, kernel_m)
                                _, thresh_m = _cv2.threshold(closed_m, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
                                kernel_m2 = _cv2.getStructuringElement(_cv2.MORPH_RECT, (9, 3))
                                thresh_m = _cv2.morphologyEx(thresh_m, _cv2.MORPH_CLOSE, kernel_m2)
                                
                                contours, _ = _cv2.findContours(thresh_m, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                                best_band = None
                                best_score = -1.0
                                for cnt in contours:
                                    bx, by, bw, bh = _cv2.boundingRect(cnt)
                                    if bw < 40 or bh < 12:
                                        continue
                                    b_aspect = bw / float(max(1, bh))
                                    if b_aspect < 1.0 or b_aspect > 5.5:
                                        continue
                                    if bw > s_w * 0.60 or bh > s_h * 0.70:
                                        continue
                                    if (bw * bh) > (v_w * v_h * 0.15):
                                        continue
                                    center_dist = abs((bx + bw / 2.0) - (s_w / 2.0)) / float(max(1, s_w))
                                    score = (bw * bh) * (1.0 - 0.5 * center_dist)
                                    if score > best_score:
                                        best_score = score
                                        best_band = (bx, by, bw, bh)
                                
                                if best_band is not None:
                                    bx, by, bw, bh = best_band
                                    px1 = s_x1 + bx
                                    py1 = s_y1 + by
                                    px2 = px1 + bw
                                    py2 = py1 + bh
                                    pad_w = max(4, int(bw * 0.08))
                                    pad_h = max(3, int(bh * 0.12))
                                    px1 = max(0, px1 - pad_w)
                                    py1 = max(0, py1 - pad_h)
                                    px2 = min(w_crop, px2 + pad_w)
                                    py2 = min(h_crop, py2 + pad_h)
                                    
                                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                                    global_pbox = BoundingBoxXYXY(
                                        x1=vehicle_bbox.x1 + px1, y1=vehicle_bbox.y1 + py1,
                                        x2=vehicle_bbox.x1 + px2, y2=vehicle_bbox.y1 + py2
                                    )
                                    candidates.append((plate_crop, global_pbox))
                                    self.fp_rejection_stats["fallback_accepted"] += 1
                    except Exception:
                        pass

                # 4. Fallback: Lower-middle vehicle strip without artificial over-expansion
                if not candidates:
                    cy1, cy2 = int(v_h * 0.65), int(v_h * 0.88)
                    cx1, cx2 = int(v_w * 0.15), int(v_w * 0.85)
                    fallback_crop = vehicle_crop[cy1:cy2, cx1:cx2]
                    fc_h, fc_w = fallback_crop.shape[:2]
                    aspect_fb = fc_w / float(max(1, fc_h))

                    # Guard 1: Minimum pixel dimensions (40x12)
                    if fc_w < 40 or fc_h < 12:
                        self.fp_rejection_stats["fallback_too_small_rejected"] += 1
                    # Guard 1.5: Aspect ratio, absolute width, relative width/height/area constraints
                    elif aspect_fb < 0.75 or aspect_fb > 6.0 or fc_w > 400 or fc_w > v_w * 0.73 or fc_h > v_h * 0.50 or (fc_w * fc_h) > (v_w * v_h * 0.35):
                        self.fp_rejection_stats["fallback_geometry_rejected"] += 1
                    else:
                        try:
                            import cv2 as _cv2
                            gray_fb = _cv2.cvtColor(fallback_crop, _cv2.COLOR_BGR2GRAY) if len(fallback_crop.shape) == 3 else fallback_crop
                            mean_fb = float(np.mean(gray_fb))
                            std_fb = float(np.std(gray_fb))
                        except Exception:
                            mean_fb, std_fb = 128.0, 30.0

                        # Guard 2: brightness — reject near-black regions (dark panels/wheel arches)
                        if mean_fb < 20.0:
                            self.fp_rejection_stats["fallback_dark_rejected"] += 1
                        # Guard 3: contrast — reject near-uniform regions (solid colour panels / blown-out headlights)
                        elif std_fb < 10.0:
                            self.fp_rejection_stats["fallback_low_contrast_rejected"] += 1
                        else:
                            plate_bbox = BoundingBoxXYXY(
                                x1=vehicle_bbox.x1 + cx1, y1=vehicle_bbox.y1 + cy1,
                                x2=vehicle_bbox.x1 + cx2, y2=vehicle_bbox.y1 + cy2
                            )
                            candidates.append((fallback_crop, plate_bbox))
                            self.fp_rejection_stats["fallback_accepted"] += 1

        except Exception:
            pass

        # For non-NumPy inputs (dict mocks, simulation objects) produce a single synthetic
        # candidate so that simulation/test paths continue to work.
        # For real NumPy video frames (_numpy_path_ran=True), a rejected candidate means
        # the crop is genuinely unusable — return [] rather than manufacturing a fake region.
        if not candidates and not _numpy_path_ran:
            p_x1, p_y1 = vehicle_bbox.x1 + v_w * 0.20, vehicle_bbox.y1 + v_h * 0.55
            p_x2, p_y2 = vehicle_bbox.x1 + v_w * 0.80, vehicle_bbox.y1 + v_h * 0.88
            plate_bbox = BoundingBoxXYXY(x1=p_x1, y1=p_y1, x2=p_x2, y2=p_y2)
            candidates.append(({"type": "plate_crop", "bbox": plate_bbox}, plate_bbox))

        return candidates


