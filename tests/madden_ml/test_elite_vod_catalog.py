"""Elite VOD research catalog registers remote candidates without downloading."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cfb_coach.madden.model.expert_film import catalog_remote_candidates

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research" / "elite_vods" / "candidate_manifest.json"


class EliteVodCatalogTests(unittest.TestCase):
    def test_manifest_lists_at_least_five_full_match_candidates(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        candidates = payload["candidates"]
        self.assertGreaterEqual(len(candidates), 5)
        self.assertFalse(payload["local_authorized_copy_available"])
        self.assertFalse(payload["unauthorized_download_performed"])
        rights = payload["rights_summary"]
        self.assertEqual(rights["download"]["status"], "not_authorized")
        self.assertIn("pending", rights["local_analysis_and_model_training"]["status"])
        self.assertTrue(all(not row.get("training_authorized") for row in candidates))
        first = next(row for row in candidates if row.get("recommended_first_match"))
        self.assertEqual(first["match_id"], "mcs27-gametimen-noah-vs-sebatron")
        self.assertIn("youtube", first["view_links"])

    def test_catalog_writes_stubs_without_claiming_local_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = catalog_remote_candidates(MANIFEST, store=tmp)
            self.assertTrue(report["ok"])
            self.assertGreaterEqual(report["registered"], 5)
            self.assertFalse(report["downloaded"])
            stub = Path(tmp) / "remote_candidates" / "mcs27-gametimen-noah-vs-sebatron.json"
            self.assertTrue(stub.is_file())
            body = json.loads(stub.read_text(encoding="utf-8"))
            self.assertIsNone(body["local_recording_path"])
            self.assertEqual(body["permission_status"], "unknown")
            self.assertFalse(body["download_authorized"])


if __name__ == "__main__":
    unittest.main()
