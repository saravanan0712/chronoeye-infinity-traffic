"""
ChronoEye Infinity - Phase 3.5: Video Trust Bridge
Links computer vision traffic observations to the 17-point Video Forensic Engine,
attaching video_trust_score, integrity_status, provenance_status, and SHA-256 hash.
Enforces strict data lineage (Prediction -> Graph -> Traffic Observation -> Track -> Frame -> Source).
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from app.forensics.video_forensic_engine import VideoForensicEngine, VideoForensicReport


class ObservationTrustEnvelope(BaseModel):
    """
    Trust envelope wrapping traffic observation metrics with video authenticity details.
    """
    video_id: str
    camera_id: str
    source_id: str
    sha256_hash: str
    video_trust_score: float = 0.95
    integrity_status: str = "VERIFIED"  # VERIFIED, SUSPICIOUS, FAILED
    provenance_status: str = "ON_CHAIN_REGISTERED"
    forensic_status_label: str = "VERIFIED / HIGH CONFIDENCE"
    is_genuine_stream: bool = True
    lineage_path: str = "Prediction -> Graph -> Traffic Observation -> Track -> Frame -> Camera -> Source"


class VideoTrustBridge:
    """
    Bridge connecting Video Forensic Engine to Traffic Acquisition Engine.
    Ensures every visual measurement knows the digital authenticity of its video source.
    """

    def __init__(self):
        self.forensic_engine = VideoForensicEngine()
        self._cached_reports: Dict[str, VideoForensicReport] = {}

    def get_trust_envelope(
        self,
        video_id: str,
        camera_id: str,
        source_type: str = "LIVE_CCTV",
    ) -> ObservationTrustEnvelope:
        """
        Retrieves or generates a forensic report and builds an ObservationTrustEnvelope.
        """
        cache_key = f"{video_id}_{camera_id}"
        if cache_key not in self._cached_reports:
            report = self.forensic_engine.analyze_video(
                video_id=video_id,
                camera_id=camera_id,
                source_type=source_type,
            )
            self._cached_reports[cache_key] = report
        else:
            report = self._cached_reports[cache_key]

        return ObservationTrustEnvelope(
            video_id=video_id,
            camera_id=camera_id,
            source_id=f"SRC_{camera_id}",
            sha256_hash=report.file_integrity.sha256_hash,
            video_trust_score=round(report.trust_breakdown.overall_trust_score, 2),
            integrity_status=report.file_integrity.status.value,
            provenance_status="ON_CHAIN_REGISTERED" if report.file_integrity.registered_on_chain else "UNREGISTERED",
            forensic_status_label=report.trust_breakdown.trust_label,
            is_genuine_stream=report.file_integrity.hash_match,
        )
