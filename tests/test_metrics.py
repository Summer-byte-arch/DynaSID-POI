import unittest
import pandas as pd

from dynasid.metrics import ranking_metrics, user_paired_bootstrap


class MetricTests(unittest.TestCase):
    def test_ranking_metrics(self):
        metrics = ranking_metrics([4, 2, 7], 2, [1, 12, 3], top_k=3, infeasible_threshold_km=10)
        self.assertEqual(metrics["Acc@1"], 0)
        self.assertEqual(metrics["Recall@3"], 1)
        self.assertAlmostEqual(metrics["NDCG@3"], 1 / 1.5849625007)
        self.assertAlmostEqual(metrics["Infeasible@3"], 1 / 3)

    def test_bootstrap_resamples_users(self):
        frame = pd.DataFrame({"user": [1, 1, 2, 2], "new": [1, 1, 0, 0], "base": [0, 0, 0, 0]})
        result = user_paired_bootstrap(frame, "Recall@10", samples=100, seed=3)
        self.assertEqual(result["n_users"], 2)
        self.assertAlmostEqual(result["mean_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
