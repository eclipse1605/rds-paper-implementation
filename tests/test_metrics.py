"""Hand-checked MAE and top-K ranking metrics."""

import math
import unittest

import numpy as np

from src.metrics import mae, ranking_metrics


class MaeTests(unittest.TestCase):
    def test_mae_half(self):
        self.assertEqual(mae(np.array([1, 3]), np.array([2, 3])), 0.5)


class RankingMetricsTests(unittest.TestCase):
    def test_macro_metrics_on_four_item_toy(self):
        test_user = np.array([0, 0, 0, 1, 2, 2])
        test_item = np.array([0, 1, 2, 0, 0, 1])
        test_rating = np.array([5, 3, 4, 2, 5, 1], dtype=float)
        scores = np.array(
            [
                [0.9, 0.1, 0.5, 0.0],
                [0.4, 0.0, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
            ],
            dtype=float,
        )
        metrics = ranking_metrics(
            test_user,
            test_item,
            test_rating,
            scores,
            user_rows={0: 0, 1: 1, 2: 2},
            seen_items={},
            k=2,
            relevance_threshold=4.0,
        )
        self.assertEqual(metrics["n_ranking_users"], 2.0)
        self.assertAlmostEqual(metrics["f1"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["f2"], 11.0 / 12.0)
        self.assertAlmostEqual(metrics["map"], 0.75)
        self.assertAlmostEqual(metrics["mrr"], 0.75)
        self.assertAlmostEqual(metrics["fcp"], 0.75)

    def test_mrr_is_zero_when_relevant_item_is_outside_top_one(self):
        metrics = ranking_metrics(
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([5.0, 1.0]),
            np.array([[0.1, 0.9, 0.0, 0.0]]),
            user_rows={0: 0},
            seen_items={},
            k=1,
            relevance_threshold=4.0,
        )
        self.assertEqual(metrics["n_ranking_users"], 1.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_predicted_tie_makes_fcp_nan(self):
        metrics = ranking_metrics(
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([5.0, 1.0]),
            np.array([[0.5, 0.5, 0.0, 0.0]]),
            user_rows={0: 0},
            seen_items={},
            k=2,
            relevance_threshold=4.0,
        )
        self.assertTrue(math.isnan(metrics["fcp"]))


if __name__ == "__main__":
    unittest.main()
