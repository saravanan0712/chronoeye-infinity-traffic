"""
ChronoEye Infinity - Video Authenticity & Forensics Schemas
Defines structured forensic results, blockchain provenance records, cut/splice detection models,
AI manipulation indicators, and multi-factor Video Trust Score schemas.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ForensicStatus(str, Enum):
    VERIFIED = "VERIFIED"
    SUSPICIOUS = "SUSPICIOUS"
    UNVERIFIED = "UNVERIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    FAILED = "FAILED"


class FileIntegrityResult(BaseModel):
    video_id: str
    sha256_hash: str
    file_size_bytes: int
    container_format: str = "MP4/H.264"
    registered_on_chain: bool = True
    hash_match: bool = True
    status: ForensicStatus = ForensicStatus.VERIFIED


class BlockchainProvenanceRecord(BaseModel):
    video_id: str
    sha256_hash: str
    blockchain_network: str = "Ethereum / Polygon PoS"
    contract_address: str = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    transaction_hash: str = "0xa1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"
    block_number: int = 19482710
    timestamp: float
    verifier: str = "ChronoEye_Blockchain_Provenance_Node"
    verification_status: ForensicStatus = ForensicStatus.VERIFIED


class TemporalCutResult(BaseModel):
    cuts_detected: bool = False
    missing_segment_count: int = 0
    total_missing_duration_seconds: float = 0.0
    suspicious_time_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.95
    evidence: str = "No frame-rate or timestamp discontinuities detected."


class SpliceResult(BaseModel):
    splices_detected: bool = False
    splice_locations_seconds: List[float] = Field(default_factory=list)
    gop_discontinuities: int = 0
    codec_signature_mismatch_score: float = 0.0
    confidence: float = 0.92
    evidence: str = "Consistent GOP structure and color profile across frames."


class AIManipulationResult(BaseModel):
    ai_generated_score: float = 0.04  # 0.0 to 1.0
    deepfake_risk: float = 0.05       # 0.0 to 1.0
    synthetic_region_detected: bool = False
    confidence: float = 0.94
    indicators: List[str] = Field(
        default_factory=lambda: ["Natural pixel noise profile", "Consistent lighting and shadow geometry"]
    )
    status: str = "NO_STRONG_SYNTHETIC_INDICATORS_DETECTED"


class FrameAnomaly(BaseModel):
    frame_id: int
    timestamp_seconds: float
    anomaly_type: str  # "POSSIBLE_CUT", "POSSIBLE_SPLICE", "DUPLICATE_FRAME", "TIMESTAMP_JUMP"
    confidence: float
    description: str


class VideoTrustScoreBreakdown(BaseModel):
    file_integrity_score: float = 1.0          # 0.0 - 1.0
    source_authentication_score: float = 0.95   # 0.0 - 1.0
    provenance_score: float = 1.0               # 0.0 - 1.0
    metadata_consistency_score: float = 0.96   # 0.0 - 1.0
    temporal_consistency_score: float = 0.98   # 0.0 - 1.0
    forensic_consistency_score: float = 0.97   # 0.0 - 1.0
    ai_manipulation_risk: float = 0.04         # 0.0 - 1.0 (Lower is safer)
    replay_risk: float = 0.02                  # 0.0 - 1.0
    overall_trust_score: float = 0.94          # 0.0 - 1.0 (Percentage representation: 94%)
    trust_label: str = "VERIFIED / HIGH CONFIDENCE"


class VideoForensicReport(BaseModel):
    video_id: str
    camera_id: str
    source_type: str = "LIVE_CCTV"  # LIVE_CCTV, RECORDED, UPLOADED_FILE, SIMULATED
    analysis_timestamp: float
    file_integrity: FileIntegrityResult
    blockchain_provenance: BlockchainProvenanceRecord
    temporal_cuts: TemporalCutResult
    splices: SpliceResult
    ai_manipulation: AIManipulationResult
    suspicious_frame_anomalies: List[FrameAnomaly] = Field(default_factory=list)
    trust_breakdown: VideoTrustScoreBreakdown
    limitations: List[str] = Field(
        default_factory=lambda: [
            "Analysis performed on H.264 stream; re-encoding may alter fine-grained quantization tables."
        ]
    )
