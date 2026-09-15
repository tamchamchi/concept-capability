from __future__ import annotations

import unittest

import numpy as np

from src.dataset.renderer import render_sample
from src.dataset.diversity import normalized_parameter_distance
from src.dataset.renderer import DEFAULT_CONFIG_DIR, DEFAULT_RENDERER_CONFIG, load_config


COLORS = ["red", "blue", "green", "yellow", "cyan", "magenta"]
SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star"]


class RendererTest(unittest.TestCase):
    def test_every_concept_renders(self) -> None:
        for color in COLORS:
            for shape in SHAPES:
                with self.subTest(color=color, shape=shape):
                    image, metadata = render_sample(color, shape, renderer_seed=123)
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (32, 32))
                    self.assertEqual(metadata["concept_id"], f"{color}_{shape}")
                    self.assertGreater(np.asarray(image).sum(), 0)

    def test_same_seed_is_pixel_and_metadata_deterministic(self) -> None:
        first_image, first_metadata = render_sample("magenta", "star", 987654)
        second_image, second_metadata = render_sample("magenta", "star", 987654)
        np.testing.assert_array_equal(np.asarray(first_image), np.asarray(second_image))
        self.assertEqual(first_metadata, second_metadata)

    def test_different_seeds_create_diversity(self) -> None:
        hashes = {
            render_sample("green", "triangle", seed)[1]["image_sha256"]
            for seed in range(20)
        }
        self.assertEqual(len(hashes), 20)

    def test_shapes_remain_inside_padded_canvas(self) -> None:
        for shape in SHAPES:
            for seed in range(50):
                with self.subTest(shape=shape, seed=seed):
                    image, _ = render_sample("yellow", shape, seed)
                    pixels = np.asarray(image)
                    self.assertEqual(int(pixels[0, :, :].sum()), 0)
                    self.assertEqual(int(pixels[-1, :, :].sum()), 0)
                    self.assertEqual(int(pixels[:, 0, :].sum()), 0)
                    self.assertEqual(int(pixels[:, -1, :].sum()), 0)

    def test_invalid_concept_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render_sample("orange", "circle", 1)
        with self.assertRaises(ValueError):
            render_sample("red", "hexagon", 1)
        with self.assertRaises(ValueError):
            render_sample("red", "circle", -1)

    def test_parameter_distance_is_symmetric(self) -> None:
        renderer_config = load_config(DEFAULT_CONFIG_DIR / "renderer.yaml")
        _, left = render_sample("blue", "square", 10, config=renderer_config)
        _, right = render_sample("blue", "square", 20, config=renderer_config)
        forward = normalized_parameter_distance(left, right, "square", renderer_config)
        backward = normalized_parameter_distance(right, left, "square", renderer_config)
        self.assertAlmostEqual(forward, backward)
        self.assertGreater(forward, 0.0)
        self.assertEqual(
            normalized_parameter_distance(left, left, "square", renderer_config), 0.0
        )

    def test_base_renderer_only_varies_position(self) -> None:
        base_config = load_config(DEFAULT_RENDERER_CONFIG)
        samples = [
            render_sample("red", "triangle", seed, config=base_config)[1]
            for seed in range(20)
        ]

        self.assertTrue(all(sample["size"] == 8.0 for sample in samples))
        self.assertTrue(all(sample["rotation"] == 0.0 for sample in samples))
        self.assertTrue(
            all(
                (sample["final_rgb_r"], sample["final_rgb_g"], sample["final_rgb_b"])
                == (230, 30, 30)
                for sample in samples
            )
        )
        self.assertTrue(all(sample["brightness"] == 1.0 for sample in samples))
        self.assertGreater(len({sample["center_x"] for sample in samples}), 1)
        self.assertGreater(len({sample["center_y"] for sample in samples}), 1)


if __name__ == "__main__":
    unittest.main()
