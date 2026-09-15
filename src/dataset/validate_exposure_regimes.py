"""Re-run joint validation for all concept-generalization regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_concept_generalization_splits import validate_exposure_regimes
from .renderer import DEFAULT_CONFIG_DIR, load_config


def main() -> None:
    storage = load_config(DEFAULT_CONFIG_DIR / "storage.yaml")
    data_root = Path(storage["data_root"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-root",
        type=Path,
        default=data_root / storage["directories"]["master"],
    )
    parser.add_argument(
        "--regimes-root",
        type=Path,
        default=data_root / storage["directories"]["concept_generalization"],
    )
    parser.add_argument(
        "--write-audit",
        action="store_true",
        help="Replace the combined audit.json with the current validation result.",
    )
    args = parser.parse_args()
    audit = validate_exposure_regimes(args.regimes_root, args.master_root)
    if args.write_audit:
        with (args.regimes_root / "audit.json").open("w", encoding="utf-8") as file:
            json.dump(audit, file, indent=2)
            file.write("\n")
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit("Exposure-regime validation failed")


if __name__ == "__main__":
    main()
