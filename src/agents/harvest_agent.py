"""
harvest_agent.py — Autonomous training data harvesting for underrepresented categories.

Wraps the existing fetch_epmc_direct.py harvester to target specific interaction
categories identified by DataQualityAgent.  Can be called from the /harvest-data
skill or run standalone for overnight batch harvesting.

Usage:
    # Harvest for a specific category
    python classifier/src/agents/harvest_agent.py --category VECTOR

    # Harvest for all gaps identified by DataQualityAgent
    python classifier/src/agents/harvest_agent.py --fill-gaps

    # Dry-run: show what would be harvested without running
    python classifier/src/agents/harvest_agent.py --fill-gaps --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.data_quality_agent import DataQualityAgent, CoverageGap  # noqa: E402
from data.interaction_taxonomy import CANONICAL_CATEGORIES             # noqa: E402

# Harvester script path
_HARVESTER = Path(__file__).parent.parent.parent / "scripts/fetch_epmc_direct.py"
_DATA_DIR = Path(__file__).parent.parent.parent / "data/training"
_VENV_PYTHON = Path(__file__).parent.parent.parent.parent.parent / "MPvenv/bin/python"

# GloBI types to request per canonical category
_CATEGORY_TO_GLOBI: dict[str, List[str]] = {
    "PREDATION":      ["preysOn", "eats", "hunts"],
    "PARASITISM":     ["parasiteOf", "ectoparasiteOf", "kleptoparasiteOf", "hyperparasiteOf"],
    "ENDOPARASITISM": ["endoparasiteOf"],
    "PARASITOIDISM":  ["parasitoidOf"],
    "INFECTION":      ["pathogenOf", "causesDisease"],
    "VECTOR":         ["vectorOf", "transmits"],
    "POLLINATION":    ["pollinates", "visitsFlowersOf"],
    "HERBIVORY":      ["grazesOn", "feedsOn"],
    "DISPERSAL":      ["dispersesSeedsOf"],
    "SYMBIOSIS":      ["symbioticWith", "mutualistOf", "commensalistOf"],
    "REGULATION":     ["negativelyRegulates", "positivelyRegulates"],
    "GENERIC":        ["interactsWith"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HarvestResult:
    category: str
    globi_types: List[str]
    output_path: Path
    n_positives: int
    n_negatives: int
    n_total: int
    success: bool
    error: str = ""
    example_positives: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HarvestAgent
# ---------------------------------------------------------------------------

class HarvestAgent:
    """Autonomously harvests training sentences for underrepresented categories."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_pairs: int = 200,
        python_bin: Optional[Path] = None,
    ):
        """
        Args:
            data_dir:   Output directory for harvested CSVs.
            max_pairs:  Max GloBI species pairs to search per harvest run.
            python_bin: Python interpreter to use (default: finds venv python).
        """
        self.data_dir = data_dir or _DATA_DIR
        self.max_pairs = max_pairs
        self.python_bin = python_bin or _VENV_PYTHON
        if not self.python_bin.exists():
            self.python_bin = Path(sys.executable)

    def harvest_for_category(
        self,
        category: str,
        target_count: int = 100,
        output_path: Optional[Path] = None,
        dry_run: bool = False,
    ) -> HarvestResult:
        """Harvest new sentences for a canonical interaction category.

        Args:
            category:     One of the 12 canonical category codes (e.g. "VECTOR").
            target_count: Approximate number of new positives to aim for.
            output_path:  Output CSV path. Defaults to data/training/<category>_harvest.csv.
            dry_run:      If True, print the command but don't execute.

        Returns:
            HarvestResult with counts and example sentences.
        """
        if category not in CANONICAL_CATEGORIES:
            return HarvestResult(
                category=category, globi_types=[], output_path=Path(""),
                n_positives=0, n_negatives=0, n_total=0, success=False,
                error=f"Unknown category '{category}'. "
                      f"Valid: {', '.join(CANONICAL_CATEGORIES)}",
            )

        globi_types = _CATEGORY_TO_GLOBI.get(category, [])
        if not globi_types:
            return HarvestResult(
                category=category, globi_types=[], output_path=Path(""),
                n_positives=0, n_negatives=0, n_total=0, success=False,
                error=f"No GloBI types mapped for category '{category}'",
            )

        if output_path is None:
            timestamp = time.strftime("%Y%m%d")
            output_path = self.data_dir / f"{category.lower()}_harvest_{timestamp}.csv"

        # Build command
        cmd = [
            str(self.python_bin),
            str(_HARVESTER),
            "--interaction-types", ",".join(globi_types),
            "--max-pairs", str(self.max_pairs),
            "--output", str(output_path),
        ]

        print(f"\n[HarvestAgent] Category: {category}")
        print(f"  GloBI types: {globi_types}")
        print(f"  Output: {output_path}")
        print(f"  Command: {' '.join(cmd)}")

        if dry_run:
            return HarvestResult(
                category=category, globi_types=globi_types, output_path=output_path,
                n_positives=0, n_negatives=0, n_total=0, success=True,
                error="DRY RUN — command not executed",
            )

        if not _HARVESTER.exists():
            return HarvestResult(
                category=category, globi_types=globi_types, output_path=output_path,
                n_positives=0, n_negatives=0, n_total=0, success=False,
                error=f"Harvester script not found: {_HARVESTER}",
            )

        # Run harvester
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )
            if result.returncode != 0:
                return HarvestResult(
                    category=category, globi_types=globi_types, output_path=output_path,
                    n_positives=0, n_negatives=0, n_total=0, success=False,
                    error=result.stderr[-2000:],
                )
        except subprocess.TimeoutExpired:
            return HarvestResult(
                category=category, globi_types=globi_types, output_path=output_path,
                n_positives=0, n_negatives=0, n_total=0, success=False,
                error="Harvester timed out after 1 hour",
            )
        except Exception as e:
            return HarvestResult(
                category=category, globi_types=globi_types, output_path=output_path,
                n_positives=0, n_negatives=0, n_total=0, success=False,
                error=str(e),
            )

        # Parse output
        if not output_path.exists():
            return HarvestResult(
                category=category, globi_types=globi_types, output_path=output_path,
                n_positives=0, n_negatives=0, n_total=0, success=False,
                error="Output file not created by harvester",
            )

        df = pd.read_csv(output_path)
        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        examples = df[df["label"] == 1]["text"].head(5).tolist()

        print(f"  → {n_pos} positives, {n_neg} negatives ({len(df)} total)")

        return HarvestResult(
            category=category,
            globi_types=globi_types,
            output_path=output_path,
            n_positives=n_pos,
            n_negatives=n_neg,
            n_total=len(df),
            success=True,
            example_positives=examples,
        )

    def fill_gaps(
        self,
        data_path: Optional[Path] = None,
        dry_run: bool = False,
    ) -> List[HarvestResult]:
        """Harvest for all categories currently below minimum thresholds.

        Args:
            data_path: Path to current training data CSV.
            dry_run:   If True, print commands but don't execute.

        Returns:
            List of HarvestResult for each gap.
        """
        agent = DataQualityAgent(data_path=data_path)
        gaps = agent.identify_gaps()

        if not gaps:
            print("No gaps found — all categories meet minimums.")
            return []

        print(f"\n[HarvestAgent] fill_gaps() — {len(gaps)} categories to harvest:")
        for g in gaps:
            print(f"  [{g.status}] {g.category}: needs {g.deficit} more positives")

        results = []
        for gap in gaps:
            r = self.harvest_for_category(
                category=gap.category,
                target_count=gap.deficit + 20,  # overshoot slightly
                dry_run=dry_run,
            )
            results.append(r)
            if r.success:
                print(f"  ✓ {gap.category}: {r.n_positives} new positives harvested")
            else:
                print(f"  ✗ {gap.category}: FAILED — {r.error}")

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest training data for underrepresented interaction categories"
    )
    parser.add_argument(
        "--category",
        help="Harvest for a specific category (e.g. VECTOR, HERBIVORY)",
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Harvest for all categories below minimum (uses DataQualityAgent)",
    )
    parser.add_argument(
        "--data",
        help="Path to current training CSV (for --fill-gaps gap detection)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Max GloBI species pairs per harvest run (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print harvest commands without executing",
    )
    args = parser.parse_args()

    agent = HarvestAgent(max_pairs=args.max_pairs)

    if args.category:
        result = agent.harvest_for_category(
            category=args.category.upper(),
            dry_run=args.dry_run,
        )
        if result.success:
            print(f"\nHarvested {result.n_positives} positives for {result.category}")
            if result.example_positives:
                print("Examples:")
                for ex in result.example_positives:
                    print(f"  - {ex[:100]}")
        else:
            print(f"\nHarvest failed: {result.error}", file=sys.stderr)
            sys.exit(1)

    elif args.fill_gaps:
        results = agent.fill_gaps(
            data_path=Path(args.data) if args.data else None,
            dry_run=args.dry_run,
        )
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        print(f"\nSummary: {len(successes)} successful, {len(failures)} failed")
        if successes:
            total_new = sum(r.n_positives for r in successes)
            print(f"  Total new positives: {total_new}")
        if failures:
            print("  Failures:")
            for r in failures:
                print(f"    {r.category}: {r.error}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
