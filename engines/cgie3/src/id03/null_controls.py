"""
CGIE3-ID-03 null-control audit.

This stage evaluates whether primary ID-03 relations exceed
randomized alternatives under the frozen protocol.

Primary null controls:

- circular temporal shift;
- independent calendar-block permutation;
- within-window value permutation.

The stage preserves:

- marginal distributions where required;
- temporal order where required;
- explicit non-admissible and non-estimable outcomes;
- deterministic relation-specific random seeds.

It does not:

- modify ID-02 classifications;
- change frozen thresholds;
- use evaluation-period or target-event information;
- infer causality;
- identify indispensable relations;
- establish earthquake prediction.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class NullControlAuditError(ValueError):
    """Raised when a null-control audit violates its contract."""


def fail(message: str) -> None:
    """Raise a normalized null-control audit error."""
    raise NullControlAuditError(
        str(message).strip()
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like value."""
    if not isinstance(
        value,
        Mapping,
    ):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """Require an integer greater than zero."""
    if isinstance(
        value,
        bool,
    ):
        fail(
            f"{field_name} must be an integer."
        )

    try:
        normalized = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise NullControlAuditError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
        )

    return normalized


def require_fraction(
    value: Any,
    field_name: str,
) -> float:
    """Require a finite value inside [0, 1]."""
    try:
        normalized = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise NullControlAuditError(
            f"{field_name} must be numeric."
        ) from exc

    if (
        not np.isfinite(
            normalized
        )
        or not 0.0 <= normalized <= 1.0
    ):
        fail(
            f"{field_name} must lie inside [0, 1]."
        )

    return normalized


def require_non_negative_float(
    value: Any,
    field_name: str,
) -> float:
    """Require a finite non-negative number."""
    try:
        normalized = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise NullControlAuditError(
            f"{field_name} must be numeric."
        ) from exc

    if (
        not np.isfinite(
            normalized
        )
        or normalized < 0.0
    ):
        fail(
            f"{field_name} must be finite and non-negative."
        )

    return normalized


def require_string_list(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    """Require a non-empty ordered list of unique strings."""
    if not isinstance(
        value,
        (list, tuple),
    ):
        fail(
            f"{field_name} must be a list."
        )

    normalized = tuple(
        str(
            item
        ).strip()
        for item in value
    )

    if not normalized:
        fail(
            f"{field_name} must not be empty."
        )

    if any(
        not item
        for item in normalized
    ):
        fail(
            f"{field_name} contains empty values."
        )

    if len(
        normalized
    ) != len(
        set(
            normalized
        )
    ):
        fail(
            f"{field_name} contains duplicates."
        )

    return normalized


def validate_context(
    context: ID03ExperimentContext,
) -> None:
    """Validate null-control prerequisites."""
    if not isinstance(
        context,
        ID03ExperimentContext,
    ):
        fail(
            "context must be an ID03ExperimentContext."
        )

    if context.experiment_id != "CGIE3_ID_03":
        fail(
            "Unexpected experiment ID: "
            f"{context.experiment_id}"
        )

    if (
        context.runtime.get(
            "overlap_audit_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Overlap audit must complete before "
            "the null-control audit."
        )

    required_outputs = {
        "primary_population",
        "relation_dependencies",
        "multiscale_relations",
        "overlap_sensitivity",
    }

    missing_outputs = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing_outputs:
        fail(
            "Null-control prerequisites are missing: "
            + ", ".join(
                missing_outputs
            )
        )


def get_null_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate the frozen null-control contract."""
    null_controls = require_mapping(
        context.configuration.get(
            "null_controls"
        ),
        "null_controls",
    )

    if (
        null_controls.get(
            "enabled"
        )
        is not True
    ):
        fail(
            "Null controls must remain enabled."
        )

    repetitions = require_mapping(
        null_controls.get(
            "repetitions"
        ),
        "null_controls.repetitions",
    )

    primary_nulls = require_string_list(
        null_controls.get(
            "primary_nulls"
        ),
        "null_controls.primary_nulls",
    )

    expected_primary_nulls = (
        "circular_temporal_shift",
        "independent_block_permutation",
        "within_window_value_permutation",
    )

    if (
        primary_nulls
        != expected_primary_nulls
    ):
        fail(
            "Unexpected primary null-control order."
        )

    repetition_counts = {
        null_id:
            require_positive_integer(
                repetitions.get(
                    null_id
                ),
                (
                    "null_controls.repetitions."
                    f"{null_id}"
                ),
            )
        for null_id
        in primary_nulls
    }

    if any(
        count != 300
        for count in repetition_counts.values()
    ):
        fail(
            "Each primary null control must use "
            "exactly 300 repetitions."
        )

    circular_shift = require_mapping(
        null_controls.get(
            "circular_temporal_shift"
        ),
        "null_controls.circular_temporal_shift",
    )

    block_permutation = require_mapping(
        null_controls.get(
            "independent_block_permutation"
        ),
        "null_controls.independent_block_permutation",
    )

    value_permutation = require_mapping(
        null_controls.get(
            "within_window_value_permutation"
        ),
        "null_controls.within_window_value_permutation",
    )

    empirical_significance = require_mapping(
        null_controls.get(
            "empirical_significance"
        ),
        "null_controls.empirical_significance",
    )

    outcome_classes = require_mapping(
        null_controls.get(
            "outcome_classes"
        ),
        "null_controls.outcome_classes",
    )

    if (
        circular_shift.get(
            "preserve_internal_order"
        )
        is not True
    ):
        fail(
            "Circular shift must preserve internal order."
        )

    if (
        circular_shift.get(
            "preserve_marginal_distribution"
        )
        is not True
    ):
        fail(
            "Circular shift must preserve the marginal distribution."
        )

    if (
        block_permutation.get(
            "block_unit"
        )
        != "calendar_month"
    ):
        fail(
            "Independent block permutation must use calendar months."
        )

    if (
        block_permutation.get(
            "preserve_within_block_order"
        )
        is not True
    ):
        fail(
            "Block permutation must preserve within-block order."
        )

    if (
        block_permutation.get(
            "permute_one_component_only"
        )
        is not True
    ):
        fail(
            "Block permutation must permute one component only."
        )

    if (
        value_permutation.get(
            "preserve_marginal_distribution"
        )
        is not True
    ):
        fail(
            "Value permutation must preserve "
            "the marginal distribution."
        )

    if (
        empirical_significance.get(
            "correction"
        )
        != "benjamini_hochberg"
    ):
        fail(
            "ID-03 requires Benjamini-Hochberg correction."
        )

    if (
        empirical_significance.get(
            "correction_scope"
        )
        != "within_null_and_scale"
    ):
        fail(
            "Unexpected empirical-significance "
            "correction scope."
        )

    exceeds_null = require_mapping(
        outcome_classes.get(
            "exceeds_null"
        ),
        "null_controls.outcome_classes.exceeds_null",
    )

    partially_exceeds = require_mapping(
        outcome_classes.get(
            "partially_exceeds_null"
        ),
        (
            "null_controls.outcome_classes."
            "partially_exceeds_null"
        ),
    )

    feature_table = require_mapping(
        context.configuration.get(
            "feature_table"
        ),
        "feature_table",
    )

    overlap_audit = require_mapping(
        context.configuration.get(
            "overlap_audit"
        ),
        "overlap_audit",
    )

    overlap_estimability = require_mapping(
        overlap_audit.get(
            "estimability"
        ),
        "overlap_audit.estimability",
    )

    return {
        "primary_nulls":
            primary_nulls,

        "repetition_counts":
            repetition_counts,

        "random_seed":
            require_positive_integer(
                null_controls.get(
                    "random_seed"
                ),
                "null_controls.random_seed",
            ),

        "minimum_shift_fraction":
            require_fraction(
                circular_shift.get(
                    "minimum_shift_fraction"
                ),
                (
                    "null_controls."
                    "circular_temporal_shift."
                    "minimum_shift_fraction"
                ),
            ),

        "maximum_shift_fraction":
            require_fraction(
                circular_shift.get(
                    "maximum_shift_fraction"
                ),
                (
                    "null_controls."
                    "circular_temporal_shift."
                    "maximum_shift_fraction"
                ),
            ),

        "alpha":
            require_fraction(
                empirical_significance.get(
                    "alpha"
                ),
                (
                    "null_controls."
                    "empirical_significance.alpha"
                ),
            ),

        "minimum_primary_nulls_exceeds":
            require_positive_integer(
                exceeds_null.get(
                    "minimum_primary_nulls_passed"
                ),
                (
                    "null_controls.outcome_classes."
                    "exceeds_null."
                    "minimum_primary_nulls_passed"
                ),
            ),

        "exceeds_corrected_p_maximum":
            require_fraction(
                exceeds_null.get(
                    "corrected_p_value_maximum"
                ),
                (
                    "null_controls.outcome_classes."
                    "exceeds_null."
                    "corrected_p_value_maximum"
                ),
            ),

        "minimum_primary_nulls_partial":
            require_positive_integer(
                partially_exceeds.get(
                    "minimum_primary_nulls_passed"
                ),
                (
                    "null_controls.outcome_classes."
                    "partially_exceeds_null."
                    "minimum_primary_nulls_passed"
                ),
            ),

        "partial_corrected_p_maximum":
            require_fraction(
                partially_exceeds.get(
                    "corrected_p_value_maximum"
                ),
                (
                    "null_controls.outcome_classes."
                    "partially_exceeds_null."
                    "corrected_p_value_maximum"
                ),
            ),

        "minimum_samples":
            require_positive_integer(
                overlap_estimability.get(
                    "minimum_paired_observations"
                ),
                (
                    "overlap_audit.estimability."
                    "minimum_paired_observations"
                ),
            ),

        "minimum_unique_values":
            require_positive_integer(
                overlap_estimability.get(
                    "minimum_unique_values_per_component"
                ),
                (
                    "overlap_audit.estimability."
                    "minimum_unique_values_per_component"
                ),
            ),

        "timestamp_column":
            str(
                feature_table[
                    "timestamp_column"
                ]
            ).strip(),

        "window_column":
            str(
                feature_table[
                    "window_column"
                ]
            ).strip(),

        "required_windows":
            tuple(
                str(
                    value
                ).strip()
                for value in feature_table[
                    "required_windows"
                ]
            ),
    }


def deterministic_seed(
    base_seed: int,
    *parts: str,
) -> int:
    """Derive a deterministic independent seed."""
    token = "::".join(
        (
            str(
                base_seed
            ),
            *(
                str(
                    part
                )
                for part in parts
            ),
        )
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        token
    ).digest()

    offset = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return int(
        (
            base_seed
            + offset
        )
        % (
            2**32 - 1
        )
    )


def select_baseline(
    context: ID03ExperimentContext,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """Select only the frozen baseline interval."""
    analysis_period = require_mapping(
        context.configuration.get(
            "analysis_period"
        ),
        "analysis_period",
    )

    baseline = require_mapping(
        analysis_period.get(
            "baseline"
        ),
        "analysis_period.baseline",
    )

    start = pd.Timestamp(
        baseline[
            "start_utc"
        ]
    )

    end = pd.Timestamp(
        baseline[
            "end_utc"
        ]
    )

    if start.tzinfo is None:
        start = start.tz_localize(
            "UTC"
        )
    else:
        start = start.tz_convert(
            "UTC"
        )

    if end.tzinfo is None:
        end = end.tz_localize(
            "UTC"
        )
    else:
        end = end.tz_convert(
            "UTC"
        )

    frame = context.frozen_features.loc[
        (
            context.frozen_features[
                timestamp_column
            ]
            >= start
        )
        & (
            context.frozen_features[
                timestamp_column
            ]
            <= end
        )
    ].copy()

    if frame.empty:
        fail(
            "Null-control audit found no baseline rows."
        )

    return frame


def prepare_pair_data(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
) -> pd.DataFrame:
    """Prepare finite paired numeric observations."""
    required_columns = {
        source_id,
        target_id,
    }

    missing_columns = sorted(
        required_columns
        - set(
            frame.columns
        )
    )

    if missing_columns:
        fail(
            "Null-control frame is missing components: "
            + ", ".join(
                missing_columns
            )
        )

    pair = frame[
        [
            source_id,
            target_id,
        ]
    ].copy()

    pair[
        source_id
    ] = pd.to_numeric(
        pair[
            source_id
        ],
        errors="coerce",
    )

    pair[
        target_id
    ] = pd.to_numeric(
        pair[
            target_id
        ],
        errors="coerce",
    )

    pair = pair.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return pair.dropna(
        subset=[
            source_id,
            target_id,
        ]
    ).reset_index(
        drop=True
    )


def estimate_absolute_spearman(
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    minimum_samples: int,
    minimum_unique_values: int,
) -> tuple[float | None, float | None, str]:
    """
    Estimate signed and absolute Spearman strength.

    Returns:

    - signed strength;
    - absolute strength;
    - estimability reason.
    """
    observation_count = int(
        len(
            source_values
        )
    )

    if observation_count < minimum_samples:
        return (
            None,
            None,
            "insufficient_observations",
        )

    if (
        np.unique(
            source_values
        ).size
        < minimum_unique_values
    ):
        return (
            None,
            None,
            "insufficient_unique_source_values",
        )

    if (
        np.unique(
            target_values
        ).size
        < minimum_unique_values
    ):
        return (
            None,
            None,
            "insufficient_unique_target_values",
        )

    result = spearmanr(
        source_values,
        target_values,
        nan_policy="omit",
    )

    strength = float(
        result.statistic
    )

    if not math.isfinite(
        strength
    ):
        return (
            None,
            None,
            "non_finite_estimate",
        )

    return (
        strength,
        abs(
            strength
        ),
        "estimable",
    )


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    """
    Apply deterministic Benjamini-Hochberg correction.

    Missing values remain missing.
    """
    output = pd.Series(
        np.nan,
        index=p_values.index,
        dtype=float,
    )

    valid = pd.to_numeric(
        p_values,
        errors="coerce",
    ).dropna()

    if valid.empty:
        return output

    valid = valid.clip(
        lower=0.0,
        upper=1.0,
    )

    ordered = valid.sort_values(
        kind="stable"
    )

    count = int(
        len(
            ordered
        )
    )

    ranks = np.arange(
        1,
        count + 1,
        dtype=float,
    )

    adjusted = (
        ordered.to_numpy(
            dtype=float
        )
        * count
        / ranks
    )

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    output.loc[
        ordered.index
    ] = adjusted

    return output


def empirical_p_value(
    observed_absolute_strength: float,
    null_absolute_strengths: np.ndarray,
) -> float | None:
    """Calculate a finite-sample empirical upper-tail p-value."""
    finite_nulls = np.asarray(
        null_absolute_strengths,
        dtype=float,
    )

    finite_nulls = finite_nulls[
        np.isfinite(
            finite_nulls
        )
    ]

    if finite_nulls.size == 0:
        return None

    exceedances = int(
        (
            finite_nulls
            >= observed_absolute_strength
        ).sum()
    )

    return float(
        (
            exceedances
            + 1
        )
        / (
            finite_nulls.size
            + 1
        )
  )
