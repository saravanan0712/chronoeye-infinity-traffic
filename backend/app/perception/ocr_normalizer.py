"""
ChronoEye Infinity - OCR Normalizer
Handles post-OCR candidate normalization, including prefix stripping for 2-row plate corruption.
"""

import re
from typing import List

# Plausible plate structure: 2-3 letters followed by 2-4 digits
# Constrained to 5-6 characters total
_PLAUSIBLE_PLATE_PATTERN = re.compile(r"^[A-Z]{2,3}[0-9]+$")


def generate_normalized_plate_candidates(raw_text: str, norm_text: str = "") -> List[str]:
    """
    Given raw OCR text (and optionally existing normalized text), generates candidate plate strings.
    1. Preserves original normalized text (or cleaned raw text) as the primary candidate.
    2. If raw or normalized text length >= 7:
       - Generates secondary candidate(s) by stripping 1 or 2 leading characters.
       - Accepts a stripped candidate ONLY if:
         * total length is between 5 and 6 characters (inclusive)
         * matches plausible plate structure: letters followed by digits (2-3 letters + digits)
    3. Guarantees no duplicate candidates and preserves candidate order (primary first).
    """
    candidates: List[str] = []

    clean_raw = raw_text.upper().replace(" ", "").replace("-", "").replace(".", "").strip() if raw_text else ""
    clean_norm = norm_text.upper().replace(" ", "").replace("-", "").replace(".", "").strip() if norm_text else ""

    primary = clean_norm if clean_norm else clean_raw
    if primary:
        candidates.append(primary)

    bases_to_check: List[str] = []
    if clean_raw and len(clean_raw) >= 7:
        bases_to_check.append(clean_raw)
    if clean_norm and len(clean_norm) >= 7 and clean_norm not in bases_to_check:
        bases_to_check.append(clean_norm)

    for base in bases_to_check:
        for strip_count in (1, 2):
            if len(base) > strip_count:
                stripped = base[strip_count:]
                if 5 <= len(stripped) <= 6:
                    if _PLAUSIBLE_PLATE_PATTERN.match(stripped):
                        if stripped not in candidates:
                            candidates.append(stripped)

    return candidates
