import unittest

import numpy as np
import pandas as pd

from tabtester.recommendation import (
    generate_candidates,
    pareto_efficient_mask,
    score_candidates,
    select_shortlist,
    suggest_sample_count,
    validate_design_space,
    validate_target_spec,
)


def design_table(step=1.0):
    return pd.DataFrame(
        [
            {
                "Variable": "A",
                "Observed min": 0.0,
                "Observed max": 10.0,
                "Search min": 0.0,
                "Search max": 10.0,
                "Step": step,
                "Active": True,
            },
            {
                "Variable": "B",
                "Observed min": 0.0,
                "Observed max": 10.0,
                "Search min": 0.0,
                "Search max": 10.0,
                "Step": step,
                "Active": True,
            },
        ]
    )


class RecommendationTests(unittest.TestCase):
    def test_discrete_small_space_uses_exhaustive(self):
        plan = suggest_sample_count(design_table(step=5.0), effort="Balanced")
        self.assertEqual(plan.mode, "exhaustive")
        self.assertEqual(plan.initial_count, 9)

    def test_mixture_candidates_respect_total(self):
        space = design_table(step=1.0)
        errors = validate_design_space(space, ["A", "B"], 10.0)
        self.assertEqual(errors, [])
        candidates, mode = generate_candidates(
            space,
            {},
            100,
            mixture_variables=["A", "B"],
            mixture_total=10.0,
        )
        self.assertEqual(mode, "exhaustive")
        self.assertTrue(np.allclose(candidates[["A", "B"]].sum(axis=1), 10.0))
        self.assertEqual(len(candidates), 11)

    def test_continuous_mixture_sampling(self):
        space = design_table(step=np.nan)
        candidates, mode = generate_candidates(
            space,
            {},
            128,
            mixture_variables=["A", "B"],
            mixture_total=10.0,
        )
        self.assertEqual(mode, "space-filling")
        self.assertEqual(len(candidates), 128)
        self.assertTrue(np.allclose(candidates[["A", "B"]].sum(axis=1), 10.0))

    def test_target_validation(self):
        target = pd.DataFrame(
            [
                {
                    "Property": "Tg",
                    "Goal": "Range",
                    "Lower": 100.0,
                    "Target": 120.0,
                    "Upper": 130.0,
                    "Priority": "High",
                    "Hard": True,
                }
            ]
        )
        self.assertEqual(validate_target_spec(target), [])

    def test_pareto_mask(self):
        costs = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.8], [1.0, 1.0]])
        mask = pareto_efficient_mask(costs)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_scoring_and_shortlist(self):
        candidates = pd.DataFrame({"A": [2.0, 5.0, 8.0], "B": [8.0, 5.0, 2.0]})
        predictions = {
            "M1": {"Tg": np.array([110.0, 120.0, 135.0])},
            "M2": {"Tg": np.array([112.0, 121.0, 132.0])},
        }
        target = pd.DataFrame(
            [
                {
                    "Property": "Tg",
                    "Goal": "Range",
                    "Lower": 115.0,
                    "Target": 120.0,
                    "Upper": 125.0,
                    "Priority": "High",
                    "Hard": True,
                }
            ]
        )
        observed_targets = pd.DataFrame({"Tg": [100.0, 120.0, 140.0]})
        training = pd.DataFrame({"A": [0.0, 5.0, 10.0], "B": [10.0, 5.0, 0.0]})
        scored = score_candidates(
            candidates,
            predictions,
            target,
            observed_targets,
            design_table(step=np.nan),
            training,
        )
        self.assertEqual(int(scored.results.loc[1, "Hard violations"]), 0)
        self.assertGreater(scored.results.loc[1, "Target fit"], scored.results.loc[0, "Target fit"])
        shortlist = select_shortlist(scored.results, ["A", "B"], limit=2)
        self.assertLessEqual(len(shortlist), 2)
        self.assertEqual(int(shortlist.iloc[0]["Hard violations"]), 0)


if __name__ == "__main__":
    unittest.main()
