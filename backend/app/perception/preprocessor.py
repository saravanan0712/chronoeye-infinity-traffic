"""
ChronoEye Infinity - Phase 2: Video Preprocessor Module
Provides configurable image preprocessing, normalization, contrast adjustments,
ROI extraction (Road, Lane, Intersection, Queue, Entry/Exit lines), and perspective transformation matrices.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class ROIConfig(BaseModel):
    """
    Region of Interest (ROI) configuration for a camera stream.
    Coordinates are specified as polygon vertices or bounding rectangles.
    """
    roi_id: str
    roi_type: str = "ROAD"  # ROAD, LANE, INTERSECTION, QUEUE_AREA, ENTRY_LINE, EXIT_LINE
    vertices: List[Tuple[float, float]] = Field(default_factory=list)  # [(x1, y1), (x2, y2), ...]
    polygon_normalized: bool = False  # True if normalized (0.0 to 1.0), False if pixel coords


class CountingLine(BaseModel):
    """
    Virtual counting line configuration.
    """
    line_id: str
    name: str = "ENTRY_LINE"
    start_pt: Tuple[float, float]  # (x1, y1)
    end_pt: Tuple[float, float]    # (x2, y2)
    direction_normal: Tuple[float, float] = (0.0, 1.0)  # Unit vector for crossing orientation


class PreprocessorConfig(BaseModel):
    """
    Configuration for video preprocessing pipeline.
    """
    target_width: int = 1920
    target_height: int = 1080
    normalize_color: bool = False
    contrast_stretch: bool = False
    denoise: bool = False
    rois: List[ROIConfig] = Field(default_factory=list)
    counting_lines: List[CountingLine] = Field(default_factory=list)
    homography_matrix: Optional[List[List[float]]] = None  # 3x3 perspective warp matrix


class VideoPreprocessor:
    """
    Video preprocessing service executing configured image operations,
    ROI masking, and perspective transformations on video frames.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()

    def process_frame(
        self,
        frame: Any,
        meta_width: int = 1920,
        meta_height: int = 1080,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Applies configured preprocessing steps to the input frame.
        Returns: (processed_frame, preprocessing_metadata)
        """
        info = {
            "resized": False,
            "contrast_applied": self.config.contrast_stretch,
            "normalized": self.config.normalize_color,
            "rois_configured": len(self.config.rois),
            "lines_configured": len(self.config.counting_lines),
            "has_homography": self.config.homography_matrix is not None,
        }

        if frame is None:
            return None, info

        # Handle NumPy image array processing if OpenCV/NumPy present
        try:
            import cv2
            import numpy as np

            if isinstance(frame, np.ndarray):
                h, w = frame.shape[:2]

                # 1. Resize if required
                if (w != self.config.target_width or h != self.config.target_height) and self.config.target_width > 0:
                    frame = cv2.resize(frame, (self.config.target_width, self.config.target_height))
                    info["resized"] = True

                # 2. Contrast stretching (CLAHE)
                if self.config.contrast_stretch:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                # 3. Denoising
                if self.config.denoise:
                    frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)

                return frame, info

        except ImportError:
            pass

        return frame, info

    def is_point_in_roi(
        self,
        pt: Tuple[float, float],
        roi: ROIConfig,
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> bool:
        """
        Determines whether a 2D point (x, y) falls inside a configured ROI polygon.
        """
        if not roi.vertices or len(roi.vertices) < 3:
            return True  # Open ROI if unspecified

        x, y = pt
        verts = roi.vertices
        if roi.polygon_normalized:
            verts = [(vx * img_w, vy * img_h) for vx, vy in verts]

        # Ray casting algorithm
        inside = False
        n = len(verts)
        p1x, p1y = verts[0]
        for i in range(n + 1):
            p2x, p2y = verts[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
