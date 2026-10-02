"""Class Balance & Source Distribution Analyzer for VoxGuard.

Calculates human vs. synthetic sample counts and percentages per split,
analyzes generator/source distribution, and identifies shared sources across partitions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from app.evaluation.manifest_validator import ManifestRow

logger = logging.getLogger("voxguard.class_balance")


@dataclass
class SplitStats:
    """Class balance statistics for an individual dataset split."""
    total: int = 0
    human: int = 0
    synthetic: int = 0
    human_pct: float = 0.0
    synthetic_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "human": self.human,
            "synthetic": self.synthetic,
            "human_pct": round(self.human_pct, 2),
            "synthetic_pct": round(self.synthetic_pct, 2)
        }


@dataclass
class ClassBalanceReport:
    """Consolidated class balance and source distribution report."""
    overall: SplitStats
    splits: Dict[str, SplitStats]
    source_distribution: Dict[str, int]
    source_id_distribution: Dict[str, int]
    sources_per_split: Dict[str, List[str]]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "splits": {s: stats.to_dict() for s, stats in self.splits.items()},
            "source_distribution": self.source_distribution,
            "source_id_distribution": self.source_id_distribution,
            "sources_per_split": self.sources_per_split,
            "warnings": self.warnings
        }


class ClassBalanceAnalyzer:
    """Analyzer computing class balance ratios and source distribution metadata."""

    def analyze(self, rows: List[ManifestRow]) -> ClassBalanceReport:
        """Analyzes a list of ManifestRow entries and generates class balance statistics.

        Args:
            rows: List of validated ManifestRow entries.

        Returns:
            ClassBalanceReport detailing class balance per split and source distributions.
        """
        overall = SplitStats()
        split_data: Dict[str, SplitStats] = {
            "train": SplitStats(),
            "validation": SplitStats(),
            "test": SplitStats(),
            "ood": SplitStats()
        }

        source_counts: Dict[str, int] = {}
        source_id_counts: Dict[str, int] = {}
        sources_per_split: Dict[str, Set[str]] = {
            "train": set(),
            "validation": set(),
            "test": set(),
            "ood": set()
        }

        warnings: List[str] = []

        for r in rows:
            overall.total += 1
            if r.label == "human":
                overall.human += 1
            elif r.label == "synthetic":
                overall.synthetic += 1

            split_key = r.split.lower()
            if split_key in split_data:
                split_data[split_key].total += 1
                if r.label == "human":
                    split_data[split_key].human += 1
                elif r.label == "synthetic":
                    split_data[split_key].synthetic += 1
                
                if r.source:
                    sources_per_split[split_key].add(r.source)

            # Source statistics
            if r.source:
                source_counts[r.source] = source_counts.get(r.source, 0) + 1
            if r.source_id:
                source_id_counts[r.source_id] = source_id_counts.get(r.source_id, 0) + 1

        # Calculate percentages
        if overall.total > 0:
            overall.human_pct = (overall.human / overall.total) * 100.0
            overall.synthetic_pct = (overall.synthetic / overall.total) * 100.0

        for s_key, s_stats in split_data.items():
            if s_stats.total > 0:
                s_stats.human_pct = (s_stats.human / s_stats.total) * 100.0
                s_stats.synthetic_pct = (s_stats.synthetic / s_stats.total) * 100.0
                
                # Check severe imbalance
                if s_stats.human == 0 or s_stats.synthetic == 0:
                    warnings.append(f"Split '{s_key}' has extreme single-class imbalance (human={s_stats.human}, synthetic={s_stats.synthetic}).")

        # Format sources per split
        formatted_sources_per_split = {s: sorted(list(srcs)) for s, srcs in sources_per_split.items()}

        return ClassBalanceReport(
            overall=overall,
            splits=split_data,
            source_distribution=source_counts,
            source_id_distribution=source_id_counts,
            sources_per_split=formatted_sources_per_split,
            warnings=warnings
        )
