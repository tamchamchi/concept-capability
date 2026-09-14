from __future__ import annotations

import unittest

import numpy as np

from src.dataset.renderer import render_sample


COLORS = ["red", "blue", "green", "yellow", "gray", "purple"]
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
        first_image, first_metadata = render_sample("purple", "star", 987654)
        second_image, second_metadata = render_sample("purple", "star", 987654)
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


if __name__ == "__main__":
    unittest.main()
