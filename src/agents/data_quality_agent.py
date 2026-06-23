"""
data_quality_agent.py — Autonomous training data coverage monitoring.

Scans the current training dataset, maps all positives to the 12 canonical
interaction categories, and reports which categories are below the required
minimums.  Can be run standalone for overnight monitoring or called by the
/analyze-coverage skill.

Usage:
    python classifier/src/agents/data_quality_agent.py [--data <path>]
    python classifier/src/agents/data_quality_agent.py --suggest-harvests
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.interaction_taxonomy import (  # noqa: E402
    coverage_report,
    CANONICAL_CATEGORIES,
    get_required_minimum,
    GLOBI_TYPE_TO_CATEGORY,
    _ensure_loaded,
)

# Default training data path (latest version)
_DEFAULT_DATA = Path(__file__).parent.parent.parent / "data/training"


def _find_latest_training_data(data_dir: Path) -> Optional[Path]:
    """Return the path to the highest-versioned training_data_vNN.csv.

    Uses numeric version sort so v12 > v9 (not alphabetical).
    Prefers plain vN.csv over vN_suffix.csv when versions tie.
    """
    import re

    def _version_key(p: Path) -> tuple:
        m = re.search(r'_v(\d+)', p.name)
        return (int(m.group(1)) if m else 0, p.name)

    candidates = list(data_dir.glob("training_data_v*.csv"))
    if not candidates:
        return None
    return max(candidates, key=_version_key)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoverageGap:
    category: str
    current_count: int
    required: int
    deficit: int
    status: str  # "MISSING" | "LOW"
    suggested_globi_types: List[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    data_path: str
    total_positives: int
    per_category: dict      # {category: {count, required, deficit, status}}
    gaps: List[CoverageGap]
    ok_categories: List[str]


# GloBI types to suggest for harvesting per category
_HARVEST_TARGETS: dict[str, List[str]] = {
    "PREDATION":      ["preysOn", "eats", "hunts"],
    "PARASITISM":     ["parasiteOf", "ectoparasiteOf", "kleptoparasiteOf", "hyperparasiteOf"],
    "ENDOPARASITISM": ["endoparasiteOf"],
    "PARASITOIDISM":  ["parasitoidOf", "idiobiontParasitoidOf", "koinobiontParasitoidOf"],
    "INFECTION":      ["pathogenOf", "causesDisease"],
    "VECTOR":         ["vectorOf", "transmits"],
    "POLLINATION":    ["pollinates", "visitsFlowersOf"],
    "HERBIVORY":      ["grazesOn", "feedsOn"],
    "DISPERSAL":      ["dispersesSeedsOf"],
    "SYMBIOSIS":      ["symbioticWith", "mutualistOf", "commensalistOf", "epiphyteOf"],
    "REGULATION":     ["negativelyRegulates", "positivelyRegulates"],
    "GENERIC":        ["interactsWith"],
}


# ---------------------------------------------------------------------------
# DataQualityAgent
# ---------------------------------------------------------------------------

class DataQualityAgent:
    """Monitors training data interaction-type coverage autonomously."""

    def __init__(self, data_path: Optional[str | Path] = None):
        """
        Args:
            data_path: Path to training CSV.  If None, uses the latest
                       training_data_vNN.csv in the default data directory.
        """
        if data_path is None:
            data_dir = _DEFAULT_DATA
            data_path = _find_latest_training_data(data_dir)
            if data_path is None:
                raise FileNotFoundError(
                    f"No training_data_vNN.csv found in {data_dir}"
                )
        self.data_path = Path(data_path)
        self._df: Optional[pd.DataFrame] = None
        _ensure_loaded()

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.data_path)
        return self._df

    def run_coverage_report(self) -> CoverageReport:
        """Analyse the training data and return a CoverageReport."""
        df = self._load()
        positives = df[df["label"] == 1]
        if "interaction_type" not in positives.columns:
            # Older datasets without interaction_type — treat all as GENERIC
            interaction_types: list = ["interactsWith"] * len(positives)
        else:
            interaction_types = positives["interaction_type"].fillna("none").tolist()
        rep = coverage_report(interaction_types)

        gaps: List[CoverageGap] = []
        ok: List[str] = []

        for cat in CANONICAL_CATEGORIES:
            info = rep[cat]
            if info["status"] in ("MISSING", "LOW"):
                gaps.append(CoverageGap(
                    category=cat,
                    current_count=info["count"],
                    required=info["required"],
                    deficit=info["deficit"],
                    status=info["status"],
                    suggested_globi_types=_HARVEST_TARGETS.get(cat, []),
                ))
            else:
                ok.append(cat)

        return CoverageReport(
            data_path=str(self.data_path),
            total_positives=len(positives),
            per_category=rep,
            gaps=gaps,
            ok_categories=ok,
        )

    def identify_gaps(self) -> List[CoverageGap]:
        """Return only the categories below minimum, sorted by deficit (largest first)."""
        return sorted(
            self.run_coverage_report().gaps,
            key=lambda g: g.deficit,
            reverse=True,
        )

    def suggest_harvest_targets(self) -> List[dict]:
        """Return harvest targets ordered by priority (MISSING before LOW)."""
        gaps = self.identify_gaps()
        targets = []
        for gap in gaps:
            targets.append({
                "category": gap.category,
                "status": gap.status,
                "deficit": gap.deficit,
                "globi_types": gap.suggested_globi_types,
                "harvest_command": (
                    f"python classifier/scripts/fetch_epmc_direct.py "
                    f"--interaction-types {','.join(gap.suggested_globi_types)} "
                    f"--max-pairs 200 "
                    f"--output classifier/data/training/{gap.category.lower()}_harvest.csv"
                ),
            })
        return targets

    def print_report(self) -> None:
        """Print a human-readable coverage report to stdout."""
        rep = self.run_coverage_report()

        print(f"\n{'='*60}")
        print(f"DATA QUALITY AGENT — Coverage Report")
        print(f"{'='*60}")
        print(f"Data: {rep.data_path}")
        print(f"Total positives: {rep.total_positives:,}")
        print()
        print(f"  {'Category':<22} {'Count':>7} {'Required':>9} {'Deficit':>8} {'Status'}")
        print(f"  {'-'*54}")
        for cat in CANONICAL_CATEGORIES:
            info = rep.per_category[cat]
            status = info["status"]
            marker = "✗" if status in ("MISSING", "LOW") else "✓"
            print(
                f"  {marker} {cat:<20} {info['count']:>7} {info['required']:>9} "
                f"{info['deficit']:>8}   {status}"
            )

        if rep.gaps:
            print(f"\n  {'='*54}")
            print(f"  HARVEST TARGETS ({len(rep.gaps)} gaps, priority order):")
            print(f"  {'='*54}")
            for target in self.suggest_harvest_targets():
                print(f"\n  [{target['status']}] {target['category']} "
                      f"(need {target['deficit']} more)")
                print(f"    GloBI types: {', '.join(target['globi_types'])}")
                print(f"    Command:")
                print(f"      {target['harvest_command']}")
        else:
            print("\n  All categories meet minimum thresholds!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Training data coverage monitor")
    parser.add_argument("--data", default=None,
                        help="Path to training CSV (default: latest vNN)")
    parser.add_argument("--suggest-harvests", action="store_true",
                        help="Print harvest commands for gaps only")
    args = parser.parse_args()

    agent = DataQualityAgent(data_path=args.data)

    if args.suggest_harvests:
        targets = agent.suggest_harvest_targets()
        if not targets:
            print("No gaps found — all categories meet minimums.")
        else:
            print(f"# Harvest commands for {len(targets)} gaps:")
            for t in targets:
                print(f"\n# [{t['status']}] {t['category']} (deficit: {t['deficit']})")
                print(t["harvest_command"])
    else:
        agent.print_report()


if __name__ == "__main__":
    main()
