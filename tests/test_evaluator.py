from __future__ import annotations

import unittest

import torch

from src.dataset.renderer import render_sample
from src.evaluator.model import ShapeColorEvaluator
from src.evaluator.rule_based import RuleBasedEvaluator


class EvaluatorTest(unittest.TestCase):
    def test_dual_head_output_and_target_score_shapes(self) -> None:
        model = ShapeColorEvaluator()
        images = torch.zeros((4, 3, 32, 32))
        color_logits, shape_logits = model(images)
        self.assertEqual(tuple(color_logits.shape), (4, 6))
        self.assertEqual(tuple(shape_logits.shape), (4, 6))
        scores = model.target_score(
            color_logits,
            shape_logits,
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([3, 2, 1, 0]),
        )
        self.assertEqual(tuple(scores.shape), (4,))

    def test_rule_based_evaluator_recognizes_all_concepts(self) -> None:
        evaluator = RuleBasedEvaluator()
        colors = ["red", "blue", "green", "yellow", "cyan", "magenta"]
        shapes = ["circle", "square", "triangle", "diamond", "cross", "star"]
        for color in colors:
            for shape in shapes:
                for seed in (1, 17, 101):
                    with self.subTest(color=color, shape=shape, seed=seed):
                        image, _ = render_sample(color, shape, seed)
                        prediction = evaluator.predict(image)
                        self.assertEqual(prediction["concept_id"], f"{color}_{shape}")


if __name__ == "__main__":
    unittest.main()
