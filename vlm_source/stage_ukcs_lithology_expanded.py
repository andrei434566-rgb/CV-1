"""Stage more UKCS source-labeled crops without changing the first pilot."""

from __future__ import annotations

from pathlib import Path

import stage_ukcs_lithology_candidates as stage


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "ukcs_q204_q205_lithology_expanded"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Expanded UKCS staging already exists; preserving all review edits")
    stage.OUTPUT = OUTPUT
    stage.MAX_PER_CLASS_PER_WELL = 24
    stage.main()


if __name__ == "__main__":
    main()
