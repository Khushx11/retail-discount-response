import sys
import unittest
from pathlib import Path

import pandas as pd

from portfolio_demo import payload

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/final_report/robust"))
from rolling_cache import MODELS, load_complete


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
