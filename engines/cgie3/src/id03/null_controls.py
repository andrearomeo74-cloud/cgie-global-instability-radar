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

def circular_shift_null(
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    repetitions: int,
    minimum_shift_fraction: float,
    maximum_shift_fraction: float,
    minimum_samples: int,
    minimum_unique_values: int,
    seed: int,
) -> np.ndarray:
    """
    Generate absolute Spearman strengths after circularly shifting
    the target component.

    The marginal distribution and internal order of the shifted
    component are preserved.
    """
    observation_count = int(
        len(source_values)
    )

    if observation_count < minimum_samples:
        return np.full(
            repetitions,
            np.nan,
            dtype=float,
        )

    minimum_shift = max(
        1,
        int(
            math.ceil(
                observation_count
                * minimum_shift_fraction
            )
        ),
    )

    maximum_shift = min(
        observation_count - 1,
        int(
            math.floor(
                observation_count
                * maximum_shift_fraction
            )
        ),
    )

    if minimum_shift > maximum_shift:
        return np.full(
            repetitions,
            np.nan,
            dtype=float,
        )

    allowed_shifts = np.arange(
        minimum_shift,
        maximum_shift + 1,
        dtype=int,
    )

    if allowed_shifts.size == 0:
        return np.full(
            repetitions,
            np.nan,
            dtype=float,
        )

    rng = np.random.default_rng(
        seed
    )

    output = np.full(
        repetitions,
        np.nan,
        dtype=float,
    )

    for repetition in range(
        repetitions
    ):
        shift = int(
            rng.choice(
                allowed_shifts
            )
        )

        shifted_target = np.roll(
            target_values,
            shift,
        )

        _, absolute_strength, reason = (
            estimate_absolute_spearman(
                source_values,
                shifted_target,
                minimum_samples=minimum_samples,
                minimum_unique_values=(
                    minimum_unique_values
                ),
            )
        )

        if (
            reason == "estimable"
            and absolute_strength is not None
        ):
            output[
                repetition
            ] = absolute_strength

    return output


def build_calendar_blocks(
    timestamps: pd.Series,
) -> tuple[np.ndarray, ...]:
    """Return ordered positional indices for each UTC calendar month."""
    if timestamps.empty:
        return ()

    block_ids = (
        timestamps
        .dt.strftime(
            "%Y-%m"
        )
        .astype(str)
        .to_numpy()
    )

    ordered_block_ids = tuple(
        sorted(
            set(
                block_ids
            )
        )
    )

    return tuple(
        np.flatnonzero(
            block_ids == block_id
        )
        for block_id in ordered_block_ids
    )


def permute_blocks(
    values: np.ndarray,
    blocks: tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """
    Permute calendar blocks while preserving order inside each block.

    Blocks may have different lengths. Their complete contents are
    concatenated in a randomized block order.
    """
    if len(blocks) < 2:
        return None

    order = rng.permutation(
        len(blocks)
    )

    permuted = np.concatenate(
        [
            values[
                blocks[
                    int(block_index)
                ]
            ]
            for block_index in order
        ]
    )

    if len(permuted) != len(values):
        return None

    return permuted


def block_permutation_null(
    source_values: np.ndarray,
    target_values: np.ndarray,
    timestamps: pd.Series,
    *,
    repetitions: int,
    minimum_samples: int,
    minimum_unique_values: int,
    seed: int,
) -> np.ndarray:
    """
    Generate absolute Spearman strengths by independently permuting
    calendar-month blocks of the target component.
    """
    observation_count = int(
        len(source_values)
    )

    if observation_count < minimum_samples:
        return np.full(
            repetitions,
            np.nan,
            dtype=float,
        )

    blocks = build_calendar_blocks(
        timestamps
    )

    if len(blocks) < 2:
        return np.full(
            repetitions,
            np.nan,
            dtype=float,
        )

    rng = np.random.default_rng(
        seed
    )

    output = np.full(
        repetitions,
        np.nan,
        dtype=float,
    )

    for repetition in range(
        repetitions
    ):
        permuted_target = permute_blocks(
            target_values,
            blocks,
            rng,
        )

        if permuted_target is None:
            continue

        _, absolute_strength, reason = (
            estimate_absolute_spearman(
                source_values,
                permuted_target,
                minimum_samples=minimum_samples,
                minimum_unique_values=(
                    minimum_unique_values
                ),
            )
        )

        if (
            reason == "estimable"
            and absolute_strength is not None
        ):
            output[
                repetition
            ] = absolute_strength

    return output


def value_permutation_null(
    source_values: np.ndarray,
    target_values: np.ndarray,
    *,
    repetitions: int,
    minimum_samples: int,
    minimum_unique_values: int,
    seed: int,
) -> np.ndarray:
    """
    Generate absolute Spearman strengths after independently permuting
    all target values inside the temporal window.
    """
    observation_count = int(
        len(source_values)
    )

    if observation_count < minimum_samples:
        return np.full(
            repetitions,
            np.nan,
            dtype=float,
        )

    rng = np.random.default_rng(
        seed
    )

    output = np.full(
        repetitions,
        np.nan,
        dtype=float,
    )

    for repetition in range(
        repetitions
    ):
        permuted_target = rng.permutation(
            target_values
        )

        _, absolute_strength, reason = (
            estimate_absolute_spearman(
                source_values,
                permuted_target,
                minimum_samples=minimum_samples,
                minimum_unique_values=(
                    minimum_unique_values
                ),
            )
        )

        if (
            reason == "estimable"
            and absolute_strength is not None
        ):
            output[
                repetition
            ] = absolute_strength

    return output


def relation_null_records(
    frame: pd.DataFrame,
    row: Any,
    *,
    timestamp_column: str,
    contract: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Run all frozen primary null controls for one relation.

    Returns:

    - one record per repetition;
    - one summary record per null type.
    """
    source_id = str(
        row.source_id
    )

    target_id = str(
        row.target_id
    )

    relation_id = str(
        row.relation_id
    )

    window_id = str(
        row.window_id
    )

    required_columns = {
        timestamp_column,
        source_id,
        target_id,
    }

    missing_columns = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        fail(
            "Null-control window frame is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    ordered = frame.sort_values(
        by=timestamp_column,
        kind="stable",
    ).reset_index(
        drop=True
    )

    pair = ordered[
        [
            timestamp_column,
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
    ).dropna(
        subset=[
            timestamp_column,
            source_id,
            target_id,
        ]
    ).reset_index(
        drop=True
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

    observed_strength, observed_absolute, reason = (
        estimate_absolute_spearman(
            source_values,
            target_values,
            minimum_samples=contract[
                "minimum_samples"
            ],
            minimum_unique_values=contract[
                "minimum_unique_values"
            ],
        )
    )

    replication_records: list[
        dict[str, Any]
    ] = []

    summary_records: list[
        dict[str, Any]
    ] = []

    for null_id in contract[
        "primary_nulls"
    ]:
        repetitions = int(
            contract[
                "repetition_counts"
            ][
                null_id
            ]
        )

        seed = deterministic_seed(
            int(
                contract[
                    "random_seed"
                ]
            ),
            window_id,
            relation_id,
            null_id,
        )

        if reason != "estimable":
            null_values = np.full(
                repetitions,
                np.nan,
                dtype=float,
            )

            null_admissible = False
            non_admissible_reason = (
                "observed_relation_non_estimable"
            )

        elif (
            null_id
            == "circular_temporal_shift"
        ):
            null_values = circular_shift_null(
                source_values,
                target_values,
                repetitions=repetitions,
                minimum_shift_fraction=contract[
                    "minimum_shift_fraction"
                ],
                maximum_shift_fraction=contract[
                    "maximum_shift_fraction"
                ],
                minimum_samples=contract[
                    "minimum_samples"
                ],
                minimum_unique_values=contract[
                    "minimum_unique_values"
                ],
                seed=seed,
            )

            null_admissible = bool(
                np.isfinite(
                    null_values
                ).any()
            )

            non_admissible_reason = (
                None
                if null_admissible
                else "circular_shift_not_estimable"
            )

        elif (
            null_id
            == "independent_block_permutation"
        ):
            null_values = block_permutation_null(
                source_values,
                target_values,
                pair[
                    timestamp_column
                ],
                repetitions=repetitions,
                minimum_samples=contract[
                    "minimum_samples"
                ],
                minimum_unique_values=contract[
                    "minimum_unique_values"
                ],
                seed=seed,
            )

            null_admissible = bool(
                np.isfinite(
                    null_values
                ).any()
            )

            non_admissible_reason = (
                None
                if null_admissible
                else "calendar_block_permutation_not_estimable"
            )

        elif (
            null_id
            == "within_window_value_permutation"
        ):
            null_values = value_permutation_null(
                source_values,
                target_values,
                repetitions=repetitions,
                minimum_samples=contract[
                    "minimum_samples"
                ],
                minimum_unique_values=contract[
                    "minimum_unique_values"
                ],
                seed=seed,
            )

            null_admissible = bool(
                np.isfinite(
                    null_values
                ).any()
            )

            non_admissible_reason = (
                None
                if null_admissible
                else "value_permutation_not_estimable"
            )

        else:
            fail(
                f"Unsupported primary null: {null_id}"
            )

        for repetition_index, null_value in enumerate(
            null_values,
            start=1,
        ):
            replication_records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_03",

                    "window_id":
                        window_id,

                    "relation_id":
                        relation_id,

                    "source_id":
                        source_id,

                    "target_id":
                        target_id,

                    "id02_status":
                        str(
                            row.classification_status
                        ),

                    "null_id":
                        null_id,

                    "repetition":
                        int(
                            repetition_index
                        ),

                    "seed":
                        int(
                            seed
                        ),

                    "observed_strength":
                        observed_strength,

                    "observed_absolute_strength":
                        observed_absolute,

                    "null_absolute_strength":
                        (
                            float(
                                null_value
                            )
                            if np.isfinite(
                                null_value
                            )
                            else np.nan
                        ),

                    "null_estimable":
                        bool(
                            np.isfinite(
                                null_value
                            )
                        ),

                    "id02_status_modified":
                        False,
                }
            )

        finite_nulls = null_values[
            np.isfinite(
                null_values
            )
        ]

        p_value = (
            empirical_p_value(
                float(
                    observed_absolute
                ),
                finite_nulls,
            )
            if (
                observed_absolute
                is not None
                and null_admissible
            )
            else None
        )

        summary_records.append(
            {
                "experiment_id":
                    "CGIE3_ID_03",

                "window_id":
                    window_id,

                "relation_id":
                    relation_id,

                "source_id":
                    source_id,

                "target_id":
                    target_id,

                "id02_status":
                    str(
                        row.classification_status
                    ),

                "null_id":
                    null_id,

                "observed_strength":
                    observed_strength,

                "observed_absolute_strength":
                    observed_absolute,

                "paired_observation_count":
                    int(
                        len(
                            pair
                        )
                    ),

                "requested_repetitions":
                    repetitions,

                "estimable_repetitions":
                    int(
                        len(
                            finite_nulls
                        )
                    ),

                "null_admissible":
                    null_admissible,

                "non_admissible_reason":
                    non_admissible_reason,

                "null_mean_absolute_strength":
                    (
                        float(
                            np.mean(
                                finite_nulls
                            )
                        )
                        if finite_nulls.size
                        else None
                    ),

                "null_median_absolute_strength":
                    (
                        float(
                            np.median(
                                finite_nulls
                            )
                        )
                        if finite_nulls.size
                        else None
                    ),

                "null_q95_absolute_strength":
                    (
                        float(
                            np.quantile(
                                finite_nulls,
                                0.95,
                            )
                        )
                        if finite_nulls.size
                        else None
                    ),

                "empirical_p_value":
                    p_value,

                "id02_status_modified":
                    False,
            }
        )

    return (
        replication_records,
        summary_records,
    )


def build_null_tables(
    context: ID03ExperimentContext,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen primary null controls for all 74 relations."""
    primary = context.outputs[
        "primary_population"
    ]

    if not isinstance(
        primary,
        pd.DataFrame,
    ):
        fail(
            "primary_population must be a DataFrame."
        )

    if len(primary) != 74:
        fail(
            "Null-control audit expects 74 primary relations."
        )

    baseline = select_baseline(
        context,
        timestamp_column=contract[
            "timestamp_column"
        ],
    )

    replication_records: list[
        dict[str, Any]
    ] = []

    summary_records: list[
        dict[str, Any]
    ] = []

    for window_id in contract[
        "required_windows"
    ]:
        window_frame = baseline.loc[
            baseline[
                contract[
                    "window_column"
                ]
            ].astype(str)
            == window_id
        ].copy()

        if window_frame.empty:
            fail(
                f"No baseline rows exist for window {window_id}."
            )

        window_relations = primary.loc[
            primary[
                "window_id"
            ].astype(str)
            == window_id
        ]

        for row in window_relations.itertuples(
            index=False
        ):
            (
                relation_replications,
                relation_summaries,
            ) = relation_null_records(
                window_frame,
                row,
                timestamp_column=contract[
                    "timestamp_column"
                ],
                contract=contract,
            )

            replication_records.extend(
                relation_replications
            )

            summary_records.extend(
                relation_summaries
            )

    replications = pd.DataFrame.from_records(
        replication_records
    )

    summaries = pd.DataFrame.from_records(
        summary_records
    )

    expected_replication_count = (
        74
        * len(
            contract[
                "primary_nulls"
            ]
        )
        * 300
    )

    expected_summary_count = (
        74
        * len(
            contract[
                "primary_nulls"
            ]
        )
    )

    if len(
        replications
    ) != expected_replication_count:
        fail(
            "Null replication table must contain "
            f"{expected_replication_count} rows; observed "
            f"{len(replications)}."
        )

    if len(
        summaries
    ) != expected_summary_count:
        fail(
            "Null summary table must contain "
            f"{expected_summary_count} rows; observed "
            f"{len(summaries)}."
        )

    duplicate_summary_mask = summaries.duplicated(
        subset=[
            "window_id",
            "relation_id",
            "null_id",
        ],
        keep=False,
    )

    if duplicate_summary_mask.any():
        fail(
            "Null summary contains duplicate relation-null keys."
        )

    return (
        replications.sort_values(
            by=[
                "window_id",
                "source_id",
                "target_id",
                "relation_id",
                "null_id",
                "repetition",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
        summaries.sort_values(
            by=[
                "window_id",
                "source_id",
                "target_id",
                "relation_id",
                "null_id",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
    )


def apply_bh_correction(
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg correction within each null type and scale.
    """
    output = summaries.copy()

    output[
        "corrected_p_value"
    ] = np.nan

    for (
        window_id,
        null_id,
    ), index in output.groupby(
        [
            "window_id",
            "null_id",
        ],
        sort=True,
    ).groups.items():
        adjusted = benjamini_hochberg(
            output.loc[
                index,
                "empirical_p_value",
            ]
        )

        output.loc[
            index,
            "corrected_p_value",
        ] = adjusted

    output[
        "passes_alpha_0_05"
    ] = (
        output[
            "corrected_p_value"
        ]
        <= 0.05
    ).fillna(
        False
    )

    output[
        "passes_alpha_0_10"
    ] = (
        output[
            "corrected_p_value"
        ]
        <= 0.10
    ).fillna(
        False
    )

    return output


def classify_relation_null_outcome(
    frame: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign one frozen overall null-control outcome to a relation."""
    admissible = frame.loc[
        frame[
            "null_admissible"
        ]
        == True
    ].copy()

    admissible_count = int(
        len(
            admissible
        )
    )

    primary_pass_count_005 = int(
        admissible[
            "passes_alpha_0_05"
        ].sum()
    )

    primary_pass_count_010 = int(
        admissible[
            "passes_alpha_0_10"
        ].sum()
    )

    if admissible_count == 0:
        outcome = (
            "null_test_not_admissible"
        )

    elif (
        primary_pass_count_005
        >= contract[
            "minimum_primary_nulls_exceeds"
        ]
    ):
        outcome = "exceeds_null"

    elif (
        primary_pass_count_010
        >= contract[
            "minimum_primary_nulls_partial"
        ]
    ):
        outcome = (
            "partially_exceeds_null"
        )

    elif (
        primary_pass_count_005 == 0
        and primary_pass_count_010 == 0
    ):
        outcome = "equivalent_to_null"

    else:
        outcome = (
            "null_test_inconclusive"
        )

    observed_absolute_strength = (
        pd.to_numeric(
            frame[
                "observed_absolute_strength"
            ],
            errors="coerce",
        )
        .dropna()
    )

    null_medians = (
        pd.to_numeric(
            admissible[
                "null_median_absolute_strength"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if (
        not observed_absolute_strength.empty
        and not null_medians.empty
    ):
        null_margin = float(
            observed_absolute_strength.iloc[
                0
            ]
            - null_medians.mean()
        )
    else:
        null_margin = None

    return {
        "primary_nulls_admissible_count":
            admissible_count,

        "primary_nulls_passed_alpha_0_05":
            primary_pass_count_005,

        "primary_nulls_passed_alpha_0_10":
            primary_pass_count_010,

        "null_outcome":
            outcome,

        "primary_null_margin":
            null_margin,

        "minimum_one_primary_null_exceeded":
            bool(
                primary_pass_count_005
                >= 1
            ),
    }


def build_relation_null_summary(
    summaries: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Build one final null-audit row per primary relation."""
    records: list[dict[str, Any]] = []

    for (
        window_id,
        relation_id,
    ), frame in summaries.groupby(
        [
            "window_id",
            "relation_id",
        ],
        sort=True,
    ):
        first = frame.iloc[
            0
        ]

        outcome = (
            classify_relation_null_outcome(
                frame,
                contract=contract,
            )
        )

        record: dict[str, Any] = {
            "experiment_id":
                "CGIE3_ID_03",

            "window_id":
                str(
                    window_id
                ),

            "relation_id":
                str(
                    relation_id
                ),

            "source_id":
                str(
                    first[
                        "source_id"
                    ]
                ),

            "target_id":
                str(
                    first[
                        "target_id"
                    ]
                ),

            "id02_status":
                str(
                    first[
                        "id02_status"
                    ]
                ),

            "observed_strength":
                first[
                    "observed_strength"
                ],

            "observed_absolute_strength":
                first[
                    "observed_absolute_strength"
                ],

            **outcome,

            "id02_status_modified":
                False,
        }

        for null_id in contract[
            "primary_nulls"
        ]:
            null_row = frame.loc[
                frame[
                    "null_id"
                ]
                == null_id
            ]

            if len(
                null_row
            ) != 1:
                fail(
                    "Expected exactly one null summary "
                    f"for {window_id}/{relation_id}/{null_id}."
                )

            null_row = null_row.iloc[
                0
            ]

            prefix = (
                f"null_{null_id}_"
            )

            record[
                f"{prefix}admissible"
            ] = bool(
                null_row[
                    "null_admissible"
                ]
            )

            record[
                f"{prefix}estimable_repetitions"
            ] = int(
                null_row[
                    "estimable_repetitions"
                ]
            )

            record[
                f"{prefix}median_absolute_strength"
            ] = null_row[
                "null_median_absolute_strength"
            ]

            record[
                f"{prefix}q95_absolute_strength"
            ] = null_row[
                "null_q95_absolute_strength"
            ]

            record[
                f"{prefix}empirical_p_value"
            ] = null_row[
                "empirical_p_value"
            ]

            record[
                f"{prefix}corrected_p_value"
            ] = null_row[
                "corrected_p_value"
            ]

            record[
                f"{prefix}passes_0_05"
            ] = bool(
                null_row[
                    "passes_alpha_0_05"
                ]
            )

        records.append(
            record
        )

    output = pd.DataFrame.from_records(
        records
    )

    if len(
        output
    ) != 74:
        fail(
            "Relation-level null audit must contain "
            f"74 rows; observed {len(output)}."
        )

    return output.sort_values(
        by=[
            "window_id",
            "source_id",
            "target_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_null_audit_summary(
    relation_summary: pd.DataFrame,
    corrected_summaries: pd.DataFrame,
    replications: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive null-control audit counts."""
    outcome_counts = (
        relation_summary[
            "null_outcome"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    by_window: dict[str, Any] = {}

    for window_id, frame in (
        relation_summary.groupby(
            "window_id",
            sort=True,
        )
    ):
        counts = (
            frame[
                "null_outcome"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        by_window[
            str(
                window_id
            )
        ] = {
            "relation_count":
                int(
                    len(
                        frame
                    )
                ),

            "outcome_counts": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in counts.items()
            },

            "exceeds_at_least_one_primary_null_count":
                int(
                    frame[
                        "minimum_one_primary_null_exceeded"
                    ].sum()
                ),
        }

    return {
        "status":
            "COMPLETED",

        "primary_relation_count":
            int(
                len(
                    relation_summary
                )
            ),

        "null_summary_row_count":
            int(
                len(
                    corrected_summaries
                )
            ),

        "null_replication_row_count":
            int(
                len(
                    replications
                )
            ),

        "null_outcome_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in outcome_counts.items()
        },

        "minimum_one_primary_null_exceeded_count":
            int(
                relation_summary[
                    "minimum_one_primary_null_exceeded"
                ].sum()
            ),

        "by_window":
            by_window,

        "id02_statuses_modified":
            False,
    }


def audit_null_controls(
    context: ID03ExperimentContext,
) -> ID03ExperimentContext:
    """Execute the frozen CGIE3-ID-03 primary null-control audit."""
    validate_context(
        context
    )

    contract = get_null_contract(
        context
    )

    (
        replications,
        summaries,
    ) = build_null_tables(
        context,
        contract,
    )

    corrected_summaries = (
        apply_bh_correction(
            summaries
        )
    )

    relation_summary = (
        build_relation_null_summary(
            corrected_summaries,
            contract=contract,
        )
    )

    audit_summary = (
        build_null_audit_summary(
            relation_summary,
            corrected_summaries,
            replications,
        )
    )

    context.register_output(
        "null_control_replications",
        replications,
    )

    context.register_output(
        "null_control_summaries",
        corrected_summaries,
    )

    context.register_output(
        "null_controls",
        relation_summary,
    )

    context.register_output(
        "null_control_audit_summary",
        audit_summary,
    )

    context.register_runtime(
        "null_control_audit_status",
        "COMPLETED",
    )

    return context
    
