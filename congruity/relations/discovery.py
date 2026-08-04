"""
Congruity Framework Core v0.1
Domain-independent candidate relation discovery.

This module:

- generates deterministic candidate component pairs;
- estimates undirected Spearman relations;
- preserves sign, strength, sample count and estimability;
- produces RelationEstimate objects defined by the Identity Core.

It does not:

- identify indispensable relations;
- infer causality;
- select primary identity relations;
- calculate Congruity metrics;
- generate alerts.

Eligibility and primary-relation selection require a separate,
frozen experimental configuration.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from congruity.identity.models import (
    EstimabilityStatus,
    RelationDirection,
    RelationEstimate,
    RelationStatus,
)


class RelationDiscoveryError(ValueError):
    """Raised when relation discovery inputs violate the contract."""


def require_component_ids(
    component_ids: Sequence[str],
) -> tuple[str, ...]:
    """
    Validate an ordered sequence of unique component identifiers.
    """
    normalized = tuple(
        str(component_id).strip()
        for component_id in component_ids
    )

    if len(normalized) < 2:
        raise RelationDiscoveryError(
            "At least two component IDs are required."
        )

    if any(
        not component_id
        for component_id in normalized
    ):
        raise RelationDiscoveryError(
            "Component IDs must not be empty."
        )

    if len(normalized) != len(set(normalized)):
        raise RelationDiscoveryError(
            "Component IDs must be unique."
        )

    return normalized


def canonical_relation_id(
    source_id: str,
    target_id: str,
    *,
    estimator_id: str,
) -> str:
    """
    Build a deterministic identifier for an undirected relation.

    Component names are sorted lexicographically, so A--B and B--A
    always receive the same relation identifier.
    """
    source = str(source_id).strip()
    target = str(target_id).strip()
    estimator = str(estimator_id).strip()

    if not source or not target:
        raise RelationDiscoveryError(
            "Relation endpoints must not be empty."
        )

    if source == target:
        raise RelationDiscoveryError(
            "A relation cannot connect a component to itself."
        )

    if not estimator:
        raise RelationDiscoveryError(
            "estimator_id must not be empty."
        )

    left, right = sorted(
        (source, target)
    )

    return (
        f"{estimator}::"
        f"{left}--{right}"
    )


def generate_candidate_pairs(
    component_ids: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """
    Generate all unique undirected component pairs deterministically.

    The output order is lexicographic and therefore reproducible.
    """
    normalized = tuple(
        sorted(
            require_component_ids(
                component_ids
            )
        )
    )

    return tuple(
        combinations(
            normalized,
            2,
        )
    )


def validate_feature_frame(
    frame: pd.DataFrame,
    component_ids: Sequence[str],
) -> tuple[str, ...]:
    """
    Validate the feature table and return normalized component IDs.
    """
    if not isinstance(frame, pd.DataFrame):
        raise RelationDiscoveryError(
            "frame must be a pandas DataFrame."
        )

    if frame.empty:
        raise RelationDiscoveryError(
            "Feature frame must not be empty."
        )

    normalized = require_component_ids(
        component_ids
    )

    missing_columns = sorted(
        set(normalized)
        - set(frame.columns)
    )

    if missing_columns:
        raise RelationDiscoveryError(
            "Feature frame is missing components: "
            + ", ".join(missing_columns)
        )

    return normalized


def relation_sign(
    strength: float,
    *,
    zero_tolerance: float = 1e-12,
) -> int:
    """
    Convert a signed relation strength into -1, 0 or 1.
    """
    if not math.isfinite(strength):
        raise RelationDiscoveryError(
            "Relation strength must be finite."
        )

    if abs(strength) <= zero_tolerance:
        return 0

    return 1 if strength > 0.0 else -1


def prepare_pair_data(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
) -> pd.DataFrame:
    """
    Extract finite paired numeric observations for one relation.
    """
    pair = frame[
        [
            source_id,
            target_id,
        ]
    ].copy()

    pair[source_id] = pd.to_numeric(
        pair[source_id],
        errors="coerce",
    )

    pair[target_id] = pd.to_numeric(
        pair[target_id],
        errors="coerce",
    )

    pair = pair.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return pair.dropna(
        subset=[
            source_id,
            target_id,
        ]
    )


def estimate_spearman_relation(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
    *,
    window_id: str,
    timestamp_utc: str,
    minimum_samples: int,
    zero_tolerance: float = 1e-12,
    metadata: Mapping[str, object] | None = None,
) -> RelationEstimate:
    """
    Estimate one signed, undirected Spearman relation.

    Non-estimable conditions remain explicit and never receive a
    numerical strength.
    """
    if minimum_samples < 3:
        raise RelationDiscoveryError(
            "minimum_samples must be at least 3."
        )

    relation_id = canonical_relation_id(
        source_id,
        target_id,
        estimator_id="spearman",
    )

    pair = prepare_pair_data(
        frame,
        source_id,
        target_id,
    )

    sample_count = int(
        len(pair)
    )

    common_metadata: dict[str, object] = {
        "minimum_samples":
            int(minimum_samples),
        "paired_observation_count":
            sample_count,
    }

    if metadata:
        common_metadata.update(
            dict(metadata)
        )

    if sample_count < minimum_samples:
        return RelationEstimate(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            estimator_id="spearman",
            direction=(
                RelationDirection.UNDIRECTED
            ),
            status=(
                RelationStatus.NON_ESTIMABLE
            ),
            estimability=(
                EstimabilityStatus
                .INSUFFICIENT_OBSERVATIONS
            ),
            strength=None,
            sign=None,
            uncertainty=None,
            sample_count=sample_count,
            persistence=None,
            window_id=window_id,
            timestamp_utc=timestamp_utc,
            metadata={
                **common_metadata,
                "non_estimable_reason":
                    "insufficient_observations",
            },
        )

    source_values = pair[
        source_id
    ].to_numpy(
        dtype=float
    )

    target_values = pair[
        target_id
    ].to_numpy(
        dtype=float
    )

    if (
        np.all(
            source_values
            == source_values[0]
        )
        or np.all(
            target_values
            == target_values[0]
        )
    ):
        return RelationEstimate(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            estimator_id="spearman",
            direction=(
                RelationDirection.UNDIRECTED
            ),
            status=(
                RelationStatus.NON_ESTIMABLE
            ),
            estimability=(
                EstimabilityStatus
                .NON_IDENTIFIABLE
            ),
            strength=None,
            sign=None,
            uncertainty=None,
            sample_count=sample_count,
            persistence=None,
            window_id=window_id,
            timestamp_utc=timestamp_utc,
            metadata={
                **common_metadata,
                "non_estimable_reason":
                    "constant_component",
            },
        )

    result = spearmanr(
        source_values,
        target_values,
        nan_policy="omit",
    )

    strength = float(
        result.statistic
    )

    p_value = float(
        result.pvalue
    )

    if (
        not math.isfinite(strength)
        or not math.isfinite(p_value)
    ):
        return RelationEstimate(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            estimator_id="spearman",
            direction=(
                RelationDirection.UNDIRECTED
            ),
            status=(
                RelationStatus.NON_ESTIMABLE
            ),
            estimability=(
                EstimabilityStatus
                .NUMERICAL_FAILURE
            ),
            strength=None,
            sign=None,
            uncertainty=None,
            sample_count=sample_count,
            persistence=None,
            window_id=window_id,
            timestamp_utc=timestamp_utc,
            metadata={
                **common_metadata,
                "non_estimable_reason":
                    "non_finite_estimate",
            },
        )

    return RelationEstimate(
        relation_id=relation_id,
        source_id=source_id,
        target_id=target_id,
        estimator_id="spearman",
        direction=(
            RelationDirection.UNDIRECTED
        ),
        status=RelationStatus.CANDIDATE,
        estimability=(
            EstimabilityStatus.ESTIMABLE
        ),
        strength=strength,
        sign=relation_sign(
            strength,
            zero_tolerance=zero_tolerance,
        ),
        uncertainty=None,
        sample_count=sample_count,
        persistence=None,
        window_id=window_id,
        timestamp_utc=timestamp_utc,
        metadata={
            **common_metadata,
            "p_value":
                p_value,
            "absolute_strength":
                abs(strength),
            "zero_tolerance":
                float(zero_tolerance),
        },
    )


def discover_spearman_relations(
    frame: pd.DataFrame,
    component_ids: Sequence[str],
    *,
    window_id: str,
    timestamp_utc: str,
    minimum_samples: int,
    zero_tolerance: float = 1e-12,
    metadata: Mapping[str, object] | None = None,
) -> tuple[RelationEstimate, ...]:
    """
    Estimate all deterministic Spearman candidate relations.

    Every possible component pair receives exactly one output,
    including relations that are non-estimable.
    """
    normalized_components = (
        validate_feature_frame(
            frame,
            component_ids,
        )
    )

    pairs = generate_candidate_pairs(
        normalized_components
    )

    estimates = tuple(
        estimate_spearman_relation(
            frame=frame,
            source_id=source_id,
            target_id=target_id,
            window_id=window_id,
            timestamp_utc=timestamp_utc,
            minimum_samples=minimum_samples,
            zero_tolerance=zero_tolerance,
            metadata=metadata,
        )
        for source_id, target_id in pairs
    )

    expected_count = (
        len(normalized_components)
        * (
            len(normalized_components) - 1
        )
        // 2
    )

    if len(estimates) != expected_count:
        raise RelationDiscoveryError(
            "Candidate relation count differs "
            "from the deterministic expectation."
        )

    relation_ids = [
        estimate.relation_id
        for estimate in estimates
    ]

    if len(relation_ids) != len(
        set(relation_ids)
    ):
        raise RelationDiscoveryError(
            "Duplicate relation IDs were generated."
        )

    return estimates


def relation_estimates_to_frame(
    estimates: Iterable[
        RelationEstimate
    ],
) -> pd.DataFrame:
    """
    Convert relation estimates into a deterministic tabular form.
    """
    records: list[dict[str, object]] = []

    for estimate in estimates:
        metadata = dict(
            estimate.metadata
        )

        records.append(
            {
                "relation_id":
                    estimate.relation_id,
                "source_id":
                    estimate.source_id,
                "target_id":
                    estimate.target_id,
                "estimator_id":
                    estimate.estimator_id,
                "direction":
                    estimate.direction.value,
                "status":
                    estimate.status.value,
                "estimability":
                    estimate.estimability.value,
                "strength":
                    estimate.strength,
                "absolute_strength":
                    (
                        abs(estimate.strength)
                        if estimate.strength
                        is not None
                        else None
                    ),
                "sign":
                    estimate.sign,
                "uncertainty":
                    estimate.uncertainty,
                "sample_count":
                    estimate.sample_count,
                "persistence":
                    estimate.persistence,
                "window_id":
                    estimate.window_id,
                "timestamp_utc":
                    estimate.timestamp_utc,
                "p_value":
                    metadata.get(
                        "p_value"
                    ),
                "non_estimable_reason":
                    metadata.get(
                        "non_estimable_reason"
                    ),
            }
        )

    frame = pd.DataFrame.from_records(
        records
    )

    if frame.empty:
        return frame

    return frame.sort_values(
        by=[
            "source_id",
            "target_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
  )
