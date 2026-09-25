"""
ChronoEye Infinity - Phase 4: License Plate Image Preprocessor
Applies image enhancement algorithms (grayscale conversion, contrast enhancement, adaptive thresholding,
denoising, sharpening, and morphological cleanup) to produce OCR-ready image variants.
"""

from typing import List, Any, Tuple, Optional


class PlatePreprocessor:
    """
    License Plate Preprocessing Engine.
    Transforms raw plate ROI crops into multiple enhanced, binarized, and noise-reduced image variants
    to maximize OCR recognition accuracy.
    Supports both single-line plates and stacked/two-line plate decomposition.
    """

    @staticmethod
    def is_likely_stacked_plate(plate_crop: Any, aspect_threshold: float = 2.2) -> bool:
        """
        Detects whether a plate crop is likely a stacked / two-line plate based on crop geometry.
        Standard Indian single-line plates have an aspect ratio (width / height) >= 2.5 (typically 3.0 - 5.0).
        Stacked / two-line plates have an aspect ratio <= 2.2 (typically 1.1 - 1.8).
        """
        if plate_crop is None:
            return False
        if hasattr(plate_crop, "shape") and len(plate_crop.shape) >= 2:
            h, w = plate_crop.shape[:2]
            if h <= 0 or w <= 0:
                return False
            aspect = w / float(h)
            return aspect <= aspect_threshold
        return False

    @staticmethod
    def split_stacked_plate(plate_crop: Any, overlap_ratio: float = 0.02) -> Tuple[Any, Any]:
        """
        Splits a stacked/two-line plate crop into upper (top) and lower (bottom) regions
        with a small vertical overlap (default 2% around mid-line) to prevent clipping
        characters in either half.
        Returns (top_crop, bottom_crop) or (None, None) if crop is invalid or too small.
        """
        if plate_crop is None or not hasattr(plate_crop, "shape") or len(plate_crop.shape) < 2:
            return None, None
        h, w = plate_crop.shape[:2]
        if h < 10 or w < 10:
            return None, None

        split_top_end = min(h, int(round(h * (0.5 + overlap_ratio))))
        split_bottom_start = max(0, int(round(h * (0.5 - overlap_ratio))))

        top_crop = plate_crop[0:split_top_end, :]
        bottom_crop = plate_crop[split_bottom_start:h, :]
        return top_crop, bottom_crop

    def process_stacked_plate(
        self,
        plate_crop: Any,
        ocr_engine: Any,
        aspect_threshold: float = 2.2,
    ) -> Optional[Tuple[str, str, float, str]]:
        """
        Processes a stacked/two-line plate using the existing OCR recognition engine:
        1. Validates geometry: returns None if crop is not likely stacked.
        2. Splits into upper region and lower region.
        3. Runs existing OCR recognition on upper region (preprocessed variants).
        4. Runs existing OCR recognition on lower region (preprocessed variants).
        5. Combines recognized text as: TOP + BOTTOM (e.g. SX + 8525 -> SX8525).
        6. Guards against incomplete/empty split results: returns None if either half
           fails to recognize valid text or if the combined plate has no digits/letters,
           preserving fallback to existing single-line recognition without creating fake plates.
        """
        if not self.is_likely_stacked_plate(plate_crop, aspect_threshold=aspect_threshold):
            return None

        top_crop, bottom_crop = self.split_stacked_plate(plate_crop)
        if top_crop is None or bottom_crop is None or ocr_engine is None:
            return None

        top_variants = self.preprocess_plate_roi(top_crop)
        bottom_variants = self.preprocess_plate_roi(bottom_crop)

        top_res = ocr_engine.recognize_text(top_variants)
        bottom_res = ocr_engine.recognize_text(bottom_variants)

        t_raw, t_norm, t_conf = top_res[:3]
        b_raw, b_norm, b_conf = bottom_res[:3]

        # Incomplete/empty split results must NOT create fake plate numbers
        if not t_norm or not b_norm or len(t_norm) < 1 or len(b_norm) < 1:
            return None

        combined_raw = f"{t_raw}{b_raw}"
        combined_norm = f"{t_norm}{b_norm}"

        # Must contain both alphabetic characters and digits to avoid fake noise plates
        if not any(c.isalpha() for c in combined_norm) or not any(c.isdigit() for c in combined_norm):
            return None

        ocr_conf = round(float((t_conf + b_conf) / 2.0), 3)
        variant_type = "STACKED_SPLIT"

        return combined_raw, combined_norm, ocr_conf, variant_type


    def preprocess_plate_roi(self, plate_crop: Any) -> List[Any]:
        """
        Generates up to 8 bounded, deterministic OCR-ready image variants:
        1. ORIGINAL_RESIZED
        2. GRAYSCALE
        3. CLAHE (Contrast-enhanced grayscale)
        4. SHARPENED
        5. ADAPTIVE_THRESH
        6. OTSU_THRESH
        7. DENOISED
        8. MORPH_CLEAN

        Returns a list of variant items. If NumPy is available, items are dicts with
        'variant_type' and 'image' or raw numpy arrays for backwards compatibility.
        """
        variants: List[Any] = []

        try:
            import cv2
            import numpy as np

            if isinstance(plate_crop, np.ndarray) and plate_crop.size > 0:
                # Upscale to standard target height (80px) while strictly preserving aspect ratio
                h, w = plate_crop.shape[:2]
                target_h = 80
                aspect_ratio = w / float(max(1, h))
                target_w = max(100, int(round(target_h * aspect_ratio)))
                resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

                # Add 10px WHITE border padding so edge characters do not touch image boundaries
                resized = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

                # 1. Grayscale (fastest & high accuracy)
                if len(resized.shape) == 3:
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                else:
                    gray = resized.copy()
                variants.append({"variant_type": "GRAYSCALE", "image": gray})
                
                # Add mild Gaussian denoising before CLAHE/thresholding
                smoothed_gray = cv2.GaussianBlur(gray, (3, 3), 0)

                denoised = cv2.GaussianBlur(gray, (3, 3), 0)
                variants.append({
                    "variant_type": "DENOISED",
                    "image": denoised
                })

                # 2. CLAHE Contrast Enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                contrast = clahe.apply(smoothed_gray)
                variants.append({"variant_type": "CLAHE", "image": contrast})

                # 3. Adaptive Thresholding
                adaptive_thresh = cv2.adaptiveThreshold(
                    contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
                )
                variants.append({"variant_type": "ADAPTIVE_THRESH", "image": adaptive_thresh})

                morph_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (3, 3)
                )

                morph_clean = cv2.morphologyEx(
                    adaptive_thresh,
                    cv2.MORPH_CLOSE,
                    morph_kernel
                )

                variants.append({
                    "variant_type": "MORPH_CLEAN",
                    "image": morph_clean
                })

                # 4. Otsu Thresholding
                _, otsu_thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variants.append({"variant_type": "OTSU_THRESH", "image": otsu_thresh})

                # 5. Original Resized (fallback)
                variants.append({"variant_type": "ORIGINAL_RESIZED", "image": resized})

                # 6. Sharpened
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharpened = cv2.filter2D(gray, -1, kernel)
                variants.append({"variant_type": "SHARPENED", "image": sharpened})

                return variants
        except Exception as e:
            print(f"Plate preprocessing error: {e}")

        # Fallback dictionary variant representation for test mocks
        variants.append({"variant_type": "ORIGINAL_RESIZED", "raw": plate_crop})
        variants.append({"variant_type": "GRAYSCALE", "raw": plate_crop})
        variants.append({"variant_type": "ADAPTIVE_THRESH", "raw": plate_crop})
        return variants

    # Backward compatible alias for diagnostic runners
    preprocess_plate = preprocess_plate_roi
