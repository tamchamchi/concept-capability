"""Re-run validation for an existing concept-generalization master pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate_master_pool import validate_master_pool
from .renderer import DEFAULT_CONFIG_DIR, load_config


def main() -> None:
    storage_config = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    default_master = (
        Path(storage_config["data_root"])
        / storage_config["directories"]["master"]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, default=default_master)
    parser.add_argument("--samples-per-concept", type=int, default=500)
    args = parser.parse_args()

    audit = validate_master_pool(args.master_root, args.samples_per_concept)
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit("Master-pool validation failed")


if __name__ == "__main__":
    main()
