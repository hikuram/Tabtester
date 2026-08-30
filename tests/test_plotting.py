from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from tabtester.plotting import plot_classification, plot_regression


class PlottingTest(unittest.TestCase):
    def test_regression_plot_is_created(self):
        fig = plot_regression(
            np.array([1.0, 2.0, 3.0]),
            {"Model A": np.array([1.1, 1.9, 3.1])},
            "target",
        )
        try:
            self.assertEqual(fig.axes[0].get_xlabel(), "Actual target")
            self.assertEqual(fig.axes[0].get_ylabel(), "Predicted target")
        finally:
            plt.close(fig)

    def test_classification_plot_is_created(self):
        fig = plot_classification(
            np.array(["a", "a", "b", "b"]),
            {"Model A": np.array(["a", "a", "b", "a"])},
            "class",
        )
        try:
            self.assertIn("Best confusion matrix", fig.axes[0].get_title())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
