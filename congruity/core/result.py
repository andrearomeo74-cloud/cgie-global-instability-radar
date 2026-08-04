"""
Congruity Framework Core

Experiment Result

Immutable scientific output of one completed experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentResult:
    """
    Frozen scientific outcome of one experiment.

    The result summarizes what the experiment produced.
    It does not modify the ExperimentContext.
    """

    experiment_id: str

    status: str

    summary: Mapping[str, Any]

    outputs: Mapping[str, Any]

    scientific_claims: Mapping[str, bool]

    statistics: Mapping[str, Any] = field(
        default_factory=dict
    )

    warnings: tuple[str, ...] = ()

    provenance: Mapping[str, Any] = field(
        default_factory=dict
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
