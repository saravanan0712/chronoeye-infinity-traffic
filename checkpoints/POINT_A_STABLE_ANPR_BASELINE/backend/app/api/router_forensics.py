"""
ChronoEye Infinity - Video Forensics & Blockchain Provenance API Router
Exposes REST endpoints for 17-point video authenticity inspection, cryptographic SHA-256 hashing,
blockchain provenance registration/verification, forensic timeline analysis, and Video Trust Scoring.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query
from app.forensics.forensic_models import (
    VideoForensicReport,
    VideoTrustScoreBreakdown,
    BlockchainProvenanceRecord,
)
from app.forensics.video_forensic_engine import VideoForensicEngine

router = APIRouter(prefix="/api/v1/forensics", tags=["Video Authenticity & Forensics"])
_engine = VideoForensicEngine()


class ForensicAnalysisRequest(BaseModel):
    video_id: str = "VID_001"
    camera_id: str = "CAM_A_EAST"
    source_type: str = "LIVE_CCTV"
    is_suspicious_demo: bool = False


class BlockchainRegisterRequest(BaseModel):
    video_id: str
    sha256_hash: str


@router.post("/analyze", response_model=VideoForensicReport)
def analyze_video_authenticity(req: ForensicAnalysisRequest):
    """
    Triggers full 17-point forensic analysis inspecting file integrity, temporal cuts,
    video splices, AI manipulation risk, and blockchain provenance.
    """
    return _engine.analyze_video(
        video_id=req.video_id,
        camera_id=req.camera_id,
        source_type=req.source_type,
        is_suspicious_demo=req.is_suspicious_demo,
    )


@router.get("/trust/{video_id}", response_model=VideoTrustScoreBreakdown)
def get_video_trust_score(video_id: str, is_suspicious_demo: bool = Query(default=False)):
    """Retrieves multi-factor Video Trust Score breakdown and risk metrics."""
    report = _engine.analyze_video(
        video_id=video_id,
        camera_id="CAM_A_EAST",
        is_suspicious_demo=is_suspicious_demo,
    )
    return report.trust_breakdown


@router.get("/timeline/{video_id}")
def get_forensic_timeline(video_id: str, is_suspicious_demo: bool = Query(default=False)) -> Dict[str, Any]:
    """Retrieves forensic timeline with suspicious timestamps and flagged frame anomalies."""
    report = _engine.analyze_video(
        video_id=video_id,
        camera_id="CAM_A_EAST",
        is_suspicious_demo=is_suspicious_demo,
    )
    return {
        "video_id": video_id,
        "overall_trust_score": report.trust_breakdown.overall_trust_score,
        "trust_label": report.trust_breakdown.trust_label,
        "suspicious_anomalies_count": len(report.suspicious_frame_anomalies),
        "anomalies": [a.model_dump() for a in report.suspicious_frame_anomalies],
        "cuts_detected": report.temporal_cuts.cuts_detected,
        "splices_detected": report.splices.splices_detected,
        "ai_manipulation_risk": report.ai_manipulation.ai_generated_score,
    }


@router.post("/blockchain/register", response_model=BlockchainProvenanceRecord)
def register_blockchain_provenance(req: BlockchainRegisterRequest):
    """Registers video SHA-256 cryptographic signature onto the blockchain ledger."""
    return _engine.blockchain_ledger.register_video_hash(req.video_id, req.sha256_hash)


@router.get("/blockchain/verify/{video_id}", response_model=BlockchainProvenanceRecord)
def verify_blockchain_provenance(video_id: str, current_hash: str = Query(...)):
    """Verifies video hash against registered on-chain provenance records."""
    return _engine.blockchain_ledger.verify_video_hash(video_id, current_hash)
