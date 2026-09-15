from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dataset.generate_master_pool import generate, validate_master_pool


class MasterPoolTest(unittest.TestCase):
    def test_small_master_pool_is_complete_and_exposure_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "master"
            audit = generate(output_root, samples_per_concept=2)

            self.assertTrue(audit["passed"])
            self.assertEqual(audit["actual_total"], 72)
            self.assertEqual(audit["duplicate_image_hash_count"], 0)
            self.assertTrue((output_root / "metadata.csv").is_file())
            self.assertTrue((output_root / "metadata.parquet").is_file())
            self.assertTrue((output_root / "manifest.json").is_file())

            metadata = pd.read_parquet(output_root / "metadata.parquet")
            self.assertEqual(set(metadata["split"]), {"master"})
            self.assertNotIn("support_level", metadata.columns)
            self.assertEqual(metadata.groupby("concept_id").size().nunique(), 1)
            self.assertEqual(int(metadata.groupby("concept_id").size().iloc[0]), 2)

            second_audit = validate_master_pool(output_root, samples_per_concept=2)
            self.assertTrue(second_audit["passed"])


if __name__ == "__main__":
    unittest.main()
