from __future__ import annotations

import unittest

from src.dataset.exposure_assignment import (
    build_count_plan,
    validate_balanced_targets,
)
from src.dataset.renderer import DEFAULT_CONFIG_DIR, load_config


class ExposureAssignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.concepts = load_config(DEFAULT_CONFIG_DIR / "concepts.yaml")["concepts"]
        cls.targets = load_config(DEFAULT_CONFIG_DIR / "target_concepts.yaml")[
            "target_concepts"
        ]
        cls.concept_ids = [concept["id"] for concept in cls.concepts]

    def test_targets_are_balanced(self) -> None:
        validate_balanced_targets(self.concepts, self.targets)

    def test_all_exposures_keep_constant_budget(self) -> None:
        expected_counts = {
            1.00: (100, 100),
            0.75: (75, 105),
            0.50: (50, 110),
            0.25: (25, 115),
            0.05: (5, 119),
            0.00: (0, 120),
        }
        target_set = set(self.targets)
        for exposure, (target_count, non_target_count) in expected_counts.items():
            with self.subTest(exposure=exposure):
                plan = build_count_plan(
                    self.concept_ids, self.targets, n_full=100,
                    exposure_level=exposure,
                )
                self.assertEqual(sum(plan.values()), 3600)
                self.assertEqual(
                    {plan[concept] for concept in target_set}, {target_count}
                )
                self.assertEqual(
                    {
                        count
                        for concept, count in plan.items()
                        if concept not in target_set
                    },
                    {non_target_count},
                )

    def test_checked_in_regime_configs_match_count_plans(self) -> None:
        target_set = set(self.targets)
        for regime in ("E100", "E075", "E050", "E025", "E005", "E000"):
            with self.subTest(regime=regime):
                config = load_config(
                    DEFAULT_CONFIG_DIR / f"concept_generalization_{regime}.yaml"
                )
                plan = build_count_plan(
                    self.concept_ids,
                    self.targets,
                    n_full=100,
                    exposure_level=config["exposure_level"],
                )
                target_counts = {plan[concept] for concept in target_set}
                non_target_counts = {
                    count
                    for concept, count in plan.items()
                    if concept not in target_set
                }
                self.assertEqual(target_counts, {config["target_count"]})
                self.assertEqual(non_target_counts, {config["non_target_count"]})
                self.assertEqual(sum(plan.values()), config["total_train_size"])

    def test_uneven_redistribution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_count_plan(
                self.concept_ids,
                self.targets[:5],
                n_full=100,
                exposure_level=0.75,
            )


if __name__ == "__main__":
    unittest.main()
