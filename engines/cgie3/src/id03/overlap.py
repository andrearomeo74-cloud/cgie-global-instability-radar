"""
CGIE3-ID-03 rolling-window overlap audit.

This stage evaluates whether ID-02 eligible and candidate relations
remain observable when temporal overlap among adjacent rolling-window
feature rows is reduced.

Sampling schemes:

- rolling_full;
- half_window_stride;
- non_overlapping.

The stage does not:

- modify ID-02 classifications;
- use evaluation-period data;
- use target-event labels;
- infer causality or earthquake prediction;
- select relational-family representatives.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class OverlapAuditError(ValueError):
    """Raised when the overlap audit violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a normalized overlap-audit error."""
    raise OverlapAuditError(
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
        raise OverlapAuditError(
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
    """Require a finite numeric value inside [0, 1]."""
    try:
        normalized = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise OverlapAuditError(
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
        raise OverlapAuditError(
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


def validate_context(
    context: ID03ExperimentContext,
) -> None:
    """Validate overlap-audit prerequisites."""
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
            "multiscale_audit_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Multiscale audit must complete before "
            "the overlap audit."
        )

    required_outputs = {
        "primary_population",
        "relation_dependencies",
        "multiscale_relations",
        "multiscale_relations_long",
    }

    missing_outputs = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing_outputs:
        fail(
            "Overlap-audit prerequisites are missing: "
            + ", ".join(
                missing_outputs
            )
        )


def get_overlap_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate the frozen overlap-audit contract."""
    audit = require_mapping(
        context.configuration.get(
            "overlap_audit"
        ),
        "overlap_audit",
    )

    if audit.get(
        "enabled"
    ) is not True:
        fail(
            "Overlap audit must remain enabled."
        )

    schemes_raw = audit.get(
        "sampling_schemes"
    )

    if not isinstance(
        schemes_raw,
        (list, tuple),
    ):
        fail(
            "overlap_audit.sampling_schemes "
            "must be a list."
        )

    schemes = tuple(
        str(
            value
        ).strip()
        for value in schemes_raw
    )

    expected_schemes = (
        "rolling_full",
        "half_window_stride",
        "non_overlapping",
    )

    if schemes != expected_schemes:
        fail(
            "Unexpected overlap sampling schemes."
        )

    stride_hours = require_mapping(
        audit.get(
            "stride_hours"
        ),
        "overlap_audit.stride_hours",
    )

    estimability = require_mapping(
        audit.get(
            "estimability"
        ),
        "overlap_audit.estimability",
    )

    uncertainty = require_mapping(
        audit.get(
            "uncertainty"
        ),
        "overlap_audit.uncertainty",
    )

    comparison = require_mapping(
        audit.get(
            "comparison"
        ),
        "overlap_audit.comparison",
    )

    timestamp_spacing = require_mapping(
        audit.get(
            "timestamp_spacing"
        ),
        "overlap_audit.timestamp_spacing",
    )

    if (
        timestamp_spacing.get(
            "infer_from_data"
        )
        is not True
    ):
        fail(
            "Timestamp spacing must be inferred from data."
        )

    repetitions = require_positive_integer(
        uncertainty.get(
            "repetitions"
        ),
        "overlap_audit.uncertainty.repetitions",
    )

    if repetitions != 300:
        fail(
            "ID-03 overlap audit requires exactly "
            "300 bootstrap repetitions."
        )

    return {
        "sampling_schemes":
            schemes,

        "stride_hours":
            stride_hours,

        "minimum_samples":
            require_positive_integer(
                estimability.get(
                    "minimum_paired_observations"
                ),
                (
                    "overlap_audit.estimability."
                    "minimum_paired_observations"
                ),
            ),

        "minimum_unique_values":
            require_positive_integer(
                estimability.get(
                    "minimum_unique_values_per_component"
                ),
                (
                    "overlap_audit.estimability."
                    "minimum_unique_values_per_component"
                ),
            ),

        "bootstrap_repetitions":
            repetitions,

        "confidence_level":
            require_fraction(
                uncertainty.get(
                    "confidence_level"
                ),
                (
                    "overlap_audit.uncertainty."
                    "confidence_level"
                ),
            ),

        "random_seed":
            require_positive_integer(
                uncertainty.get(
                    "random_seed"
                ),
                "overlap_audit.uncertainty.random_seed",
            ),

        "maximum_robust_change":
            require_non_negative_float(
                comparison.get(
                    "maximum_absolute_strength_"
                    "change_for_robust"
                ),
                (
                    "overlap_audit.comparison."
                    "maximum_absolute_strength_"
                    "change_for_robust"
                ),
            ),

        "maximum_moderate_change":
            require_non_negative_float(
                comparison.get(
                    "maximum_absolute_strength_"
                    "change_for_moderate"
                ),
                (
                    "overlap_audit.comparison."
                    "maximum_absolute_strength_"
                    "change_for_moderate"
                ),
            ),

        "minimum_rank_correlation_robust":
            require_fraction(
                comparison.get(
                    "minimum_rank_correlation_for_robust"
                ),
                (
                    "overlap_audit.comparison."
                    "minimum_rank_correlation_for_robust"
                ),
            ),

        "minimum_rank_correlation_moderate":
            require_fraction(
                comparison.get(
                    "minimum_rank_correlation_for_moderate"
                ),
                (
                    "overlap_audit.comparison."
                    "minimum_rank_correlation_for_moderate"
                ),
            ),

        "maximum_irregular_spacing_fraction":
            require_fraction(
                timestamp_spacing.get(
                    "maximum_irregular_spacing_fraction"
                ),
                (
                    "overlap_audit.timestamp_spacing."
                    "maximum_irregular_spacing_fraction"
                ),
            ),
    }


def get_feature_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract frozen feature-table structural fields."""
    feature_table = require_mapping(
        context.configuration.get(
            "feature_table"
        ),
        "feature_table",
    )

    return {
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

        "window_specific_exclusions":
            require_mapping(
                feature_table.get(
                    "window_specific_exclusions",
                    {},
                ),
                (
                    "feature_table."
                    "window_specific_exclusions"
                ),
            ),
    }


def select_baseline(
    context: ID03ExperimentContext,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """Select the frozen 2025 baseline only."""
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
            "Overlap audit found no baseline rows."
        )

    return frame


def infer_nominal_spacing_hours(
    timestamps: pd.Series,
) -> tuple[float, float]:
    """
    Infer nominal timestamp spacing and irregular-spacing fraction.

    Returns:

    - median spacing in hours;
    - fraction of intervals differing from the median by more than
      one numerical tolerance.
    """
    ordered = (
        timestamps
        .dropna()
        .sort_values(
            kind="stable"
        )
        .drop_duplicates()
    )

    if len(
        ordered
    ) < 3:
        fail(
            "At least three timestamps are required "
            "to infer nominal spacing."
        )

    differences = (
        ordered.diff()
        .dropna()
        .dt.total_seconds()
        / 3600.0
    )

    differences = differences.loc[
        differences > 0.0
    ]

    if differences.empty:
        fail(
            "Timestamp spacing cannot be inferred."
        )

    median_spacing = float(
        differences.median()
    )

    tolerance = max(
        1e-9,
        median_spacing
        * 1e-6,
    )

    irregular_fraction = float(
        (
            np.abs(
                differences
                - median_spacing
            )
            > tolerance
        ).mean()
    )

    return (
        median_spacing,
        irregular_fraction,
    )


def deterministic_seed(
    base_seed: int,
    *parts: str,
) -> int:
    """Derive a stable seed from the declared identifiers."""
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


def sample_by_stride(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    stride_hours: int,
) -> pd.DataFrame:
    """
    Select the first available row at or after each target stride.

    Timestamp-based selection avoids assuming a perfectly complete
    hourly table.
    """
    ordered = frame.sort_values(
        by=timestamp_column,
        kind="stable",
    ).reset_index(
        drop=True
    )

    if ordered.empty:
        return ordered

    if stride_hours <= 1:
        return ordered.copy()

    selected_indices: list[int] = []

    next_target = ordered.loc[
        0,
        timestamp_column,
    ]

    for index, timestamp in enumerate(
        ordered[
            timestamp_column
        ]
    ):
        if timestamp >= next_target:
            selected_indices.append(
                index
            )

            next_target = (
                timestamp
                + pd.Timedelta(
                    hours=stride_hours
                )
            )

    return ordered.iloc[
        selected_indices
    ].reset_index(
        drop=True
    )


def prepare_pair_data(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
) -> pd.DataFrame:
    """Prepare finite paired numeric observations."""
    missing_columns = sorted(
        {
            source_id,
            target_id,
        }
        - set(
            frame.columns
        )
    )

    if missing_columns:
        fail(
            "Overlap frame is missing components: "
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


def estimate_relation(
    pair: pd.DataFrame,
    source_id: str,
    target_id: str,
    *,
    minimum_samples: int,
    minimum_unique_values: int,
) -> dict[str, Any]:
    """Estimate one Spearman relation with explicit estimability."""
    sample_count = int(
        len(
            pair
        )
    )

    if sample_count < minimum_samples:
        return {
            "estimability":
                "non_estimable",

            "non_estimable_reason":
                "insufficient_observations",

            "sample_count":
                sample_count,

            "strength":
                None,

            "absolute_strength":
                None,

            "sign":
                None,
        }

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
        np.unique(
            source_values
        ).size
        < minimum_unique_values
    ):
        return {
            "estimability":
                "non_estimable",

            "non_estimable_reason":
                "insufficient_unique_source_values",

            "sample_count":
                sample_count,

            "strength":
                None,

            "absolute_strength":
                None,

            "sign":
                None,
        }

    if (
        np.unique(
            target_values
        ).size
        < minimum_unique_values
    ):
        return {
            "estimability":
                "non_estimable",

            "non_estimable_reason":
                "insufficient_unique_target_values",

            "sample_count":
                sample_count,

            "strength":
                None,

            "absolute_strength":
                None,

            "sign":
                None,
        }

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
        return {
            "estimability":
                "non_estimable",

            "non_estimable_reason":
                "non_finite_estimate",

            "sample_count":
                sample_count,

            "strength":
                None,

            "absolute_strength":
                None,

            "sign":
                None,
        }

    sign = (
        1
        if strength > 0.0
        else (
            -1
            if strength < 0.0
            else 0
        )
    )

    return {
        "estimability":
            "estimable",

        "non_estimable_reason":
            None,

        "sample_count":
            sample_count,

        "strength":
            strength,

        "absolute_strength":
            abs(
                strength
            ),

        "sign":
            sign,
    }


def circular_bootstrap_indices(
    observation_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one circular moving-block resample."""
    if observation_count <= 0:
        fail(
            "observation_count must be positive."
        )

    effective_block_length = min(
        block_length,
        observation_count,
    )

    block_count = int(
        math.ceil(
            observation_count
            / effective_block_length
        )
    )

    starts = rng.integers(
        low=0,
        high=observation_count,
        size=block_count,
    )

    indices = np.concatenate(
        [
            (
                start
                + np.arange(
                    effective_block_length
                )
            )
            % observation_count
            for start in starts
        ]
    )[:observation_count]

    return indices.astype(
        int,
        copy=False,
    )


def bootstrap_uncertainty(
    pair: pd.DataFrame,
    source_id: str,
    target_id: str,
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    minimum_samples: int,
    minimum_unique_values: int,
) -> dict[str, Any]:
    """Estimate bootstrap uncertainty for one sampled relation."""
    if len(
        pair
    ) < minimum_samples:
        return {
            "bootstrap_estimable_count":
                0,

            "bootstrap_ci_lower":
                None,

            "bootstrap_ci_upper":
                None,

            "bootstrap_ci_width":
                None,
        }

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

    observation_count = int(
        len(
            pair
        )
    )

    block_length = max(
        2,
        int(
            round(
                math.sqrt(
                    observation_count
                )
            )
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    strengths: list[float] = []

    for _ in range(
        repetitions
    ):
        indices = circular_bootstrap_indices(
            observation_count,
            block_length,
            rng,
        )

        source_sample = source_values[
            indices
        ]

        target_sample = target_values[
            indices
        ]

        if (
            np.unique(
                source_sample
            ).size
            < minimum_unique_values
            or np.unique(
                target_sample
            ).size
            < minimum_unique_values
        ):
            continue

        result = spearmanr(
            source_sample,
            target_sample,
            nan_policy="omit",
        )

        strength = float(
            result.statistic
        )

        if math.isfinite(
            strength
        ):
            strengths.append(
                strength
            )

    if not strengths:
        return {
            "bootstrap_estimable_count":
                0,

            "bootstrap_ci_lower":
                None,

            "bootstrap_ci_upper":
                None,

            "bootstrap_ci_width":
                None,
        }

    alpha = (
        1.0
        - confidence_level
    )

    lower = float(
        np.quantile(
            strengths,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            strengths,
            1.0 - alpha / 2.0,
        )
    )

    return {
        "bootstrap_estimable_count":
            int(
                len(
                    strengths
                )
            ),

        "bootstrap_ci_lower":
            lower,

        "bootstrap_ci_upper":
            upper,

        "bootstrap_ci_width":
            float(
                upper
                - lower
            ),
    }


def lag1_autocorrelation(
    values: pd.Series,
    *,
    minimum_observations: int = 12,
) -> float | None:
    """Calculate a descriptive lag-1 autocorrelation."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if len(
        numeric
    ) < minimum_observations:
        return None

    lagged = numeric.shift(
        1
    )

    valid = pd.concat(
        [
            numeric,
            lagged,
        ],
        axis=1,
    ).dropna()

    if len(
        valid
    ) < minimum_observations:
        return None

    value = valid.iloc[
        :,
        0,
    ].corr(
        valid.iloc[
            :,
            1,
        ]
    )

    if value is None:
        return None

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):
        return None

    return value


def build_primary_relation_table(
    context: ID03ExperimentContext,
) -> pd.DataFrame:
    """Return the frozen 74-relation primary audit population."""
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

    if len(
        primary
    ) != 74:
        fail(
            "Overlap audit expects 74 primary relations."
        )

    required_columns = {
        "window_id",
        "relation_id",
        "source_id",
        "target_id",
        "classification_status",
        "strength",
        "sign",
    }

    missing_columns = sorted(
        required_columns
        - set(
            primary.columns
        )
    )

    if missing_columns:
        fail(
            "Primary population is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    return primary.copy()


def audit_relation_scheme(
    window_frame: pd.DataFrame,
    row: Any,
    *,
    sampling_scheme: str,
    stride_hours: int,
    timestamp_column: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one relation under one sampling scheme."""
    sampled = sample_by_stride(
        window_frame,
        timestamp_column=timestamp_column,
        stride_hours=stride_hours,
    )

    pair = prepare_pair_data(
        sampled,
        str(
            row.source_id
        ),
        str(
            row.target_id
        ),
    )

    estimate = estimate_relation(
        pair,
        str(
            row.source_id
        ),
        str(
            row.target_id
        ),
        minimum_samples=contract[
            "minimum_samples"
        ],
        minimum_unique_values=contract[
            "minimum_unique_values"
        ],
    )

    seed = deterministic_seed(
        int(
            contract[
                "random_seed"
            ]
        ),
        str(
            row.window_id
        ),
        str(
            row.relation_id
        ),
        sampling_scheme,
    )

    uncertainty = bootstrap_uncertainty(
        pair,
        str(
            row.source_id
        ),
        str(
            row.target_id
        ),
        repetitions=contract[
            "bootstrap_repetitions"
        ],
        confidence_level=contract[
            "confidence_level"
        ],
        seed=seed,
        minimum_samples=contract[
            "minimum_samples"
        ],
        minimum_unique_values=contract[
            "minimum_unique_values"
        ],
    )

    return {
        "experiment_id":
            "CGIE3_ID_03",

        "window_id":
            str(
                row.window_id
            ),

        "relation_id":
            str(
                row.relation_id
            ),

        "source_id":
            str(
                row.source_id
            ),

        "target_id":
            str(
                row.target_id
            ),

        "id02_status":
            str(
                row.classification_status
            ),

        "sampling_scheme":
            sampling_scheme,

        "stride_hours":
            int(
                stride_hours
            ),

        "sampled_row_count":
            int(
                len(
                    sampled
                )
            ),

        "paired_observation_count":
            int(
                estimate[
                    "sample_count"
                ]
            ),

        "estimability":
            estimate[
                "estimability"
            ],

        "non_estimable_reason":
            estimate[
                "non_estimable_reason"
            ],

        "strength":
            estimate[
                "strength"
            ],

        "absolute_strength":
            estimate[
                "absolute_strength"
            ],

        "sign":
            estimate[
                "sign"
            ],

        "bootstrap_estimable_count":
            uncertainty[
                "bootstrap_estimable_count"
            ],

        "bootstrap_ci_lower":
            uncertainty[
                "bootstrap_ci_lower"
            ],

        "bootstrap_ci_upper":
            uncertainty[
                "bootstrap_ci_upper"
            ],

        "bootstrap_ci_width":
            uncertainty[
                "bootstrap_ci_width"
            ],

        "lag1_autocorrelation_source":
            lag1_autocorrelation(
                sampled[
                    str(
                        row.source_id
                    )
                ]
            ),

        "lag1_autocorrelation_target":
            lag1_autocorrelation(
                sampled[
                    str(
                        row.target_id
                    )
                ]
            ),

        "bootstrap_seed":
            seed,

        "id02_status_modified":
            False,
    }


def build_overlap_long_table(
    context: ID03ExperimentContext,
    contract: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Evaluate all 74 primary relations under three schemes."""
    baseline = select_baseline(
        context,
        timestamp_column=feature_contract[
            "timestamp_column"
        ],
    )

    primary = build_primary_relation_table(
        context
    )

    records: list[dict[str, Any]] = []

    for window_id in feature_contract[
        "required_windows"
    ]:
        window_frame = baseline.loc[
            baseline[
                feature_contract[
                    "window_column"
                ]
            ]
            == window_id
        ].copy()

        if window_frame.empty:
            fail(
                f"No baseline rows exist for window {window_id}."
            )

        (
            nominal_spacing,
            irregular_fraction,
        ) = infer_nominal_spacing_hours(
            window_frame[
                feature_contract[
                    "timestamp_column"
                ]
            ]
        )

        if (
            irregular_fraction
            > contract[
                "maximum_irregular_spacing_fraction"
            ]
        ):
            fail(
                f"Window {window_id} exceeds the frozen "
                "irregular-spacing tolerance."
            )

        window_relations = primary.loc[
            primary[
                "window_id"
            ].astype(
                str
            )
            == window_id
        ]

        stride_map = require_mapping(
            contract[
                "stride_hours"
            ][
                window_id
            ],
            f"overlap_audit.stride_hours.{window_id}",
        )

        for row in window_relations.itertuples(
            index=False
        ):
            for scheme in contract[
                "sampling_schemes"
            ]:
                stride = require_positive_integer(
                    stride_map[
                        scheme
                    ],
                    (
                        "overlap_audit.stride_hours."
                        f"{window_id}.{scheme}"
                    ),
                )

                record = audit_relation_scheme(
                    window_frame,
                    row,
                    sampling_scheme=scheme,
                    stride_hours=stride,
                    timestamp_column=feature_contract[
                        "timestamp_column"
                    ],
                    contract=contract,
                )

                record[
                    "nominal_spacing_hours"
                ] = nominal_spacing

                record[
                    "irregular_spacing_fraction"
                ] = irregular_fraction

                records.append(
                    record
                )

    output = pd.DataFrame.from_records(
        records
    )

    expected_count = (
        74
        * len(
            contract[
                "sampling_schemes"
            ]
        )
    )

    if len(
        output
    ) != expected_count:
        fail(
            "Overlap long table must contain "
            f"{expected_count} rows; observed {len(output)}."
        )

    duplicate_mask = output.duplicated(
        subset=[
            "window_id",
            "relation_id",
            "sampling_scheme",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        fail(
            "Overlap long table contains duplicate keys."
        )

    return output.sort_values(
        by=[
            "window_id",
            "source_id",
            "target_id",
            "relation_id",
            "sampling_scheme",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def classify_overlap(
    rolling_strength: float | None,
    rolling_sign: int | None,
    nonoverlap_strength: float | None,
    nonoverlap_sign: int | None,
    *,
    maximum_robust_change: float,
    maximum_moderate_change: float,
) -> tuple[str, float | None, bool]:
    """Assign one frozen relation-level overlap class."""
    if (
        nonoverlap_strength is None
        or nonoverlap_sign is None
    ):
        return (
            "non_estimable_non_overlapping",
            None,
            False,
        )

    if (
        rolling_strength is None
        or rolling_sign is None
    ):
        return (
            "inconclusive_overlap_audit",
            None,
            False,
        )

    sign_preserved = bool(
        int(
            rolling_sign
        )
        == int(
            nonoverlap_sign
        )
    )

    strength_change = float(
        abs(
            float(
                nonoverlap_strength
            )
            - float(
                rolling_strength
            )
        )
    )

    if (
        sign_preserved
        and strength_change
        <= maximum_robust_change
    ):
        return (
            "overlap_robust",
            strength_change,
            True,
        )

    if (
        sign_preserved
        and strength_change
        <= maximum_moderate_change
    ):
        return (
            "moderately_overlap_sensitive",
            strength_change,
            True,
        )

    return (
        "strongly_overlap_sensitive",
        strength_change,
        sign_preserved,
    )


def build_relation_summary(
    long_table: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Create one overlap-audit record per primary relation."""
    records: list[dict[str, Any]] = []

    for (
        window_id,
        relation_id,
    ), frame in long_table.groupby(
        [
            "window_id",
            "relation_id",
        ],
        sort=True,
    ):
        rows = {
            str(
                row.sampling_scheme
            ): row
            for row in frame.itertuples(
                index=False
            )
        }

        missing_schemes = sorted(
            set(
                contract[
                    "sampling_schemes"
                ]
            )
            - set(
                rows
            )
        )

        if missing_schemes:
            fail(
                "Overlap relation summary is missing schemes: "
                + ", ".join(
                    missing_schemes
                )
            )

        rolling = rows[
            "rolling_full"
        ]

        half = rows[
            "half_window_stride"
        ]

        nonoverlap = rows[
            "non_overlapping"
        ]

        overlap_class, change, sign_preserved = (
            classify_overlap(
                rolling_strength=rolling.strength,
                rolling_sign=rolling.sign,
                nonoverlap_strength=nonoverlap.strength,
                nonoverlap_sign=nonoverlap.sign,
                maximum_robust_change=contract[
                    "maximum_robust_change"
                ],
                maximum_moderate_change=contract[
                    "maximum_moderate_change"
                ],
            )
        )

        half_change = (
            float(
                abs(
                    float(
                        half.strength
                    )
                    - float(
                        rolling.strength
                    )
                )
            )
            if (
                half.strength is not None
                and not pd.isna(
                    half.strength
                )
                and rolling.strength is not None
                and not pd.isna(
                    rolling.strength
                )
            )
            else None
        )

        effective_support_diagnostic = min(
            int(
                nonoverlap.paired_observation_count
            ),
            int(
                half.paired_observation_count
            ),
        )

        records.append(
            {
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
                        rolling.source_id
                    ),

                "target_id":
                    str(
                        rolling.target_id
                    ),

                "id02_status":
                    str(
                        rolling.id02_status
                    ),

                "nominal_sample_count":
                    int(
                        rolling.paired_observation_count
                    ),

                "half_stride_sample_count":
                    int(
                        half.paired_observation_count
                    ),

                "nonoverlap_sample_count":
                    int(
                        nonoverlap.paired_observation_count
                    ),

                "rolling_strength":
                    rolling.strength,

                "half_stride_strength":
                    half.strength,

                "nonoverlap_strength":
                    nonoverlap.strength,

                "rolling_sign":
                    rolling.sign,

                "half_stride_sign":
                    half.sign,

                "nonoverlap_sign":
                    nonoverlap.sign,

                "half_stride_absolute_strength_change":
                    half_change,

                "nonoverlap_absolute_strength_change":
                    change,

                "nonoverlap_sign_preserved":
                    sign_preserved,

                "rolling_bootstrap_ci_width":
                    rolling.bootstrap_ci_width,

                "half_stride_bootstrap_ci_width":
                    half.bootstrap_ci_width,

                "nonoverlap_bootstrap_ci_width":
                    nonoverlap.bootstrap_ci_width,

                "lag1_autocorrelation_source":
                    rolling.lag1_autocorrelation_source,

                "lag1_autocorrelation_target":
                    rolling.lag1_autocorrelation_target,

                "effective_support_diagnostic":
                    int(
                        effective_support_diagnostic
                    ),

                "overlap_class":
                    overlap_class,

                "strongly_overlap_sensitive_flag":
                    (
                        overlap_class
                        == "strongly_overlap_sensitive"
                    ),

                "nonoverlap_estimable":
                    (
                        nonoverlap.estimability
                        == "estimable"
                    ),

                "id02_status_modified":
                    False,
            }
        )

    output = pd.DataFrame.from_records(
        records
    )

    if len(
        output
    ) != 74:
        fail(
            "Overlap relation summary must contain "
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


def compute_rank_diagnostics(
    long_table: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate scale-level relation-rank correlations by scheme."""
    records: list[dict[str, Any]] = []

    for window_id, frame in long_table.groupby(
        "window_id",
        sort=True,
    ):
        pivot = frame.pivot(
            index="relation_id",
            columns="sampling_scheme",
            values="absolute_strength",
        )

        rolling = pivot[
            "rolling_full"
        ]

        for comparison_scheme in (
            "half_window_stride",
            "non_overlapping",
        ):
            comparison = pivot[
                comparison_scheme
            ]

            valid = pd.concat(
                [
                    rolling,
                    comparison,
                ],
                axis=1,
            ).dropna()

            if len(
                valid
            ) < 3:
                rank_correlation = None
            else:
                rank_result = spearmanr(
                    valid.iloc[
                        :,
                        0,
                    ],
                    valid.iloc[
                        :,
                        1,
                    ],
                    nan_policy="omit",
                )

                rank_correlation = float(
                    rank_result.statistic
                )

                if not np.isfinite(
                    rank_correlation
                ):
                    rank_correlation = None

            records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_03",

                    "window_id":
                        str(
                            window_id
                        ),

                    "reference_scheme":
                        "rolling_full",

                    "comparison_scheme":
                        comparison_scheme,

                    "paired_relation_count":
                        int(
                            len(
                                valid
                            )
                        ),

                    "rank_correlation":
                        rank_correlation,
                }
            )

    return pd.DataFrame.from_records(
        records
    ).sort_values(
        by=[
            "window_id",
            "comparison_scheme",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_overlap_summary(
    relation_summary: pd.DataFrame,
    rank_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive overlap-audit counts."""
    class_counts = (
        relation_summary[
            "overlap_class"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    by_window: dict[str, Any] = {}

    for window_id, frame in relation_summary.groupby(
        "window_id",
        sort=True,
    ):
        counts = (
            frame[
                "overlap_class"
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

            "overlap_class_counts": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in counts.items()
            },

            "median_nonoverlap_sample_count":
                float(
                    frame[
                        "nonoverlap_sample_count"
                    ].median()
                ),

            "strongly_overlap_sensitive_count":
                int(
                    frame[
                        "strongly_overlap_sensitive_flag"
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

        "overlap_class_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in class_counts.items()
        },

        "strongly_overlap_sensitive_count":
            int(
                relation_summary[
                    "strongly_overlap_sensitive_flag"
                ].sum()
            ),

        "nonoverlap_non_estimable_count":
            int(
                (
                    relation_summary[
                        "nonoverlap_estimable"
                    ]
                    == False
                ).sum()
            ),

        "by_window":
            by_window,

        "rank_diagnostics":
            rank_diagnostics.to_dict(
                orient="records"
            ),

        "id02_statuses_modified":
            False,
    }


def audit_overlap(
    context: ID03ExperimentContext,
) -> ID03ExperimentContext:
    """Execute the frozen CGIE3-ID-03 overlap audit."""
    validate_context(
        context
    )

    contract = get_overlap_contract(
        context
    )

    feature_contract = get_feature_contract(
        context
    )

    long_table = build_overlap_long_table(
        context,
        contract,
        feature_contract,
    )

    relation_summary = build_relation_summary(
        long_table,
        contract,
    )

    rank_diagnostics = compute_rank_diagnostics(
        long_table
    )

    summary = build_overlap_summary(
        relation_summary,
        rank_diagnostics,
    )

    context.register_output(
        "overlap_estimates_long",
        long_table,
    )

    context.register_output(
        "overlap_sensitivity",
        relation_summary,
    )

    context.register_output(
        "overlap_rank_diagnostics",
        rank_diagnostics,
    )

    context.register_output(
        "overlap_audit_summary",
        summary,
    )

    context.register_runtime(
        "overlap_audit_status",
        "COMPLETED",
    )

    return context
