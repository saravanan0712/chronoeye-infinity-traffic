"""
ChronoEye Infinity - Video Forensics & Blockchain Integrity Test Suite
Automated Python test suite verifying Pipeline A (17-point Video Authenticity, SHA-256 Hashing,
Blockchain Provenance Registration/Verification, Cut/Splice Detection, AI Risk, and Trust Score Breakdown).
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.main import app
from app.forensics.video_forensic_engine import VideoForensicEngine, BlockchainIntegrityLedger


class TestVideoForensicsPipeline(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.engine = VideoForensicEngine()

    def test_1_sha256_hash_calculation(self):
        """Test 1: Verify cryptographic SHA-256 hash calculation."""
        sample_bytes = b"CHRONOEYE_TRAFFIC_FRAME_001_SAMPLE"
        h = self.engine.calculate_hash(sample_bytes)
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_2_blockchain_ledger_registration_and_verification(self):
        """Test 2: Verify on-chain blockchain provenance registration & verification."""
        ledger = BlockchainIntegrityLedger()
        video_id = "VID_TEST_101"
        sample_hash = "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"

        reg_rec = ledger.register_video_hash(video_id, sample_hash)
        self.assertEqual(reg_rec.video_id, video_id)
        self.assertEqual(reg_rec.sha256_hash, sample_hash)
        self.assertEqual(reg_rec.verification_status, "VERIFIED")

        ver_rec = ledger.verify_video_hash(video_id, sample_hash)
        self.assertEqual(ver_rec.verification_status, "VERIFIED")

    def test_3_forensic_analysis_genuine_video(self):
        """Test 3: Verify 17-point forensic inspection report for genuine video feed."""
        report = self.engine.analyze_video(
            video_id="VID_GENUINE_001",
            camera_id="CAM_A_EAST",
            source_type="LIVE_CCTV",
            is_suspicious_demo=False,
        )
        self.assertEqual(report.file_integrity.status, "VERIFIED")
        self.assertFalse(report.temporal_cuts.cuts_detected)
        self.assertFalse(report.splices.splices_detected)
        self.assertLess(report.ai_manipulation.ai_generated_score, 0.10)
        self.assertGreaterEqual(report.trust_breakdown.overall_trust_score, 0.90)

    def test_4_forensic_analysis_suspicious_video(self):
        """Test 4: Verify forensic report detects missing segments, splices, and flags suspicious trust score."""
        report = self.engine.analyze_video(
            video_id="VID_SUSPICIOUS_009",
            camera_id="CAM_B_WEST",
            source_type="UPLOADED_FILE",
            is_suspicious_demo=True,
        )
        self.assertTrue(report.temporal_cuts.cuts_detected)
        self.assertTrue(report.splices.splices_detected)
        self.assertGreater(len(report.suspicious_frame_anomalies), 0)
        self.assertLess(report.trust_breakdown.overall_trust_score, 0.50)
        self.assertIn("SUSPICIOUS", report.trust_breakdown.trust_label)

    def test_5_forensics_api_analyze_endpoint(self):
        """Test 5: Verify POST /api/v1/forensics/analyze endpoint."""
        payload = {
            "video_id": "VID_API_001",
            "camera_id": "CAM_C_NORTH",
            "source_type": "LIVE_CCTV",
            "is_suspicious_demo": False,
        }
        res = self.client.post("/api/v1/forensics/analyze", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("trust_breakdown", data)
        self.assertEqual(data["file_integrity"]["video_id"], "VID_API_001")

    def test_6_forensics_api_trust_score_endpoint(self):
        """Test 6: Verify GET /api/v1/forensics/trust/{video_id} endpoint."""
        res = self.client.get("/api/v1/forensics/trust/VID_001")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("overall_trust_score", data)
        self.assertIn("trust_label", data)

    def test_7_forensics_api_timeline_endpoint(self):
        """Test 7: Verify GET /api/v1/forensics/timeline/{video_id} endpoint."""
        res = self.client.get("/api/v1/forensics/timeline/VID_001?is_suspicious_demo=true")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("anomalies", data)
        self.assertTrue(data["cuts_detected"])
        self.assertTrue(data["splices_detected"])

    def test_8_blockchain_register_and_verify_api(self):
        """Test 8: Verify POST /api/v1/forensics/blockchain/register and GET /verify endpoints."""
        v_id = "VID_CHAIN_88"
        h = "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"

        reg_res = self.client.post("/api/v1/forensics/blockchain/register", json={"video_id": v_id, "sha256_hash": h})
        self.assertEqual(reg_res.status_code, 200)
        reg_data = reg_res.json()
        self.assertEqual(reg_data["verification_status"], "VERIFIED")

        ver_res = self.client.get(f"/api/v1/forensics/blockchain/verify/{v_id}?current_hash={h}")
        self.assertEqual(ver_res.status_code, 200)
        ver_data = ver_res.json()
        self.assertEqual(ver_data["verification_status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
