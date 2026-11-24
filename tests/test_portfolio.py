import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from portfolio_demo import payload, verify_replay_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/final_report/robust"))
from rolling_cache import MODELS, load_complete
from refit_cache import artifact_paths, publish, read_complete


class PortfolioTests(unittest.TestCase):
    def test_aggregate_dashboard_can_run_without_weights(self):
        result = payload()
        self.assertEqual(result["rows"], 405992)
        self.assertIsNone(result["scenarios"])
        self.assertAlmostEqual(result["support"][-1]["outside"], 0.8129)

    def test_resume_rejects_partial_and_misaligned_folds(self):
        complete = [{"origin": "ok", "model": model, "test_rows": 2}
                    for model in sorted(MODELS)]
        partial = [{"origin": "partial", "model": "MLP Poisson", "test_rows": 1}]
        missing_prediction = [
            {**row, "origin": "no_predictions"} for row in complete
        ]
        frame = pd.DataFrame({"origin": ["ok", "ok", "partial"],
                              **{model: [1, 2, 3] for model in MODELS}})
        summaries, predictions = load_complete(
            complete + partial + missing_prediction, frame
        )
        self.assertEqual(summaries, complete)
        self.assertEqual(predictions[0].origin.tolist(), ["ok", "ok"])

    def test_replay_fails_on_mismatched_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weights"
            manifest = root / "manifest"
            artifact.write_bytes(b"expected")
            manifest.write_text(f"{hashlib.sha256(b'expected').hexdigest()}  weights\n")
            verify_replay_inputs(root, manifest)
            artifact.write_bytes(b"another run")
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_replay_inputs(root, manifest)

    def test_bootstrap_cache_requires_matching_feature_table_and_every_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "refit_parts").mkdir()
            for path in artifact_paths(root, 0):
                path.write_text("original artifact")
            publish(root, {"replicate": 0}, "features-a")
            self.assertIsNotNone(read_complete(root, 0, "features-a"))
            self.assertIsNone(read_complete(root, 0, "features-b"))
            for path in artifact_paths(root, 0):
                with self.subTest(path=path.name):
                    path.write_text("altered")
                    self.assertIsNone(read_complete(root, 0, "features-a"))
                    path.unlink()
                    self.assertIsNone(read_complete(root, 0, "features-a"))
                    path.write_text("original artifact")
