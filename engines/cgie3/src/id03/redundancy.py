"""
CGIE3-ID-03 conditional redundancy audit.

This stage evaluates whether primary ID-03 relations retain
association after conditioning on frozen dominant activity features.

Methods:

- partial Spearman association;
- rank-linear residual Spearman association;
- deterministic transformation audit;
- decision-equivalence audit.

The stage does not:

- modify ID-02 classifications;
- remove redundant relations;
- infer causality;
- identify indispensable relations;
- select family representatives.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class RedundancyAuditError(ValueError):
    """Raised when conditional redundancy violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a normalized redundancy-audit error."""
    raise RedundancyAuditError(
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
    """Require a positive integer."""
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
        raise RedundancyAuditError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
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
        raise RedundancyAuditError(
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
    """Validate redundancy-audit prerequisites."""
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
            "null_control_audit_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Null-control audit must complete before "
            "conditional redundancy."
        )

    required_outputs = {
        "primary_population",
        "relation_dependencies",
        "multiscale_relations",
        "overlap_sensitivity",
        "null_controls",
    }

    missing_outputs = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing_outputs:
        fail(
            "Redundancy prerequisites are missing: "
            + ", ".join(
                missing_outputs
            )
        )


def get_redundancy_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate the frozen redundancy contract."""
    redundancy = require_mapping(
        context.configuration.get(
            "conditional_redundancy"
        ),
        "conditional_redundancy",
    )

    if redundancy.get(
        "enabled"
    ) is not True:
        fail(
            "Conditional redundancy must remain enabled."
        )

    methods = require_mapping(
        redundancy.get(
            "methods"
        ),
        "conditional_redundancy.methods",
    )

    partial = require_mapping(
        methods.get(
            "partial_spearman"
        ),
        "conditional_redundancy.methods.partial_spearman",
    )

    residual = require_mapping(
        methods.get(
            "residual_spearman"
        ),
        "conditional_redundancy.methods.residual_spearman",
    )

    deterministic = require_mapping(
        methods.get(
            "deterministic_transformation"
        ),
        (
            "conditional_redundancy.methods."
            "deterministic_transformation"
        ),
    )

    decision_equivalence = require_mapping(
        methods.get(
            "decision_equivalence"
        ),
        (
            "conditional_redundancy.methods."
            "decision_equivalence"
        ),
    )

    if partial.get("enabled") is not True:
        fail(
            "Partial Spearman must remain enabled."
        )

    if residual.get("enabled") is not True:
        fail(
            "Residual Spearman must remain enabled."
        )

    if residual.get(
        "residual_model"
    ) != "rank_linear":
        fail(
            "Residual model must remain rank_linear."
        )

    if deterministic.get(
        "enabled"
    ) is not True:
        fail(
            "Deterministic transformation audit "
            "must remain enabled."
        )

    if decision_equivalence.get(
        "enabled"
    ) is not True:
        fail(
            "Decision-equivalence audit must remain enabled."
        )

    residual_information = require_mapping(
        redundancy.get(
            "residual_information"
        ),
        "conditional_redundancy.residual_information",
    )

    feature_table = require_mapping(
        context.configuration.get(
            "feature_table"
        ),
        "feature_table",
    )

    return {
        "dominant_activity_features":
            require_string_list(
                redundancy.get(
                    "dominant_activity_features"
                ),
                (
                    "conditional_redundancy."
                    "dominant_activity_features"
                ),
            ),

        "conditional_controls":
            require_string_list(
                redundancy.get(
                    "conditional_controls"
                ),
                (
                    "conditional_redundancy."
                    "conditional_controls"
                ),
            ),

        "partial_minimum_observations":
            require_positive_integer(
                partial.get(
                    "minimum_complete_observations"
                ),
                (
                    "conditional_redundancy.methods."
                    "partial_spearman."
                    "minimum_complete_observations"
                ),
            ),

        "residual_minimum_observations":
            require_positive_integer(
                residual.get(
                    "minimum_complete_observations"
                ),
                (
                    "conditional_redundancy.methods."
                    "residual_spearman."
                    "minimum_complete_observations"
                ),
            ),

        "deterministic_tolerance":
            require_non_negative_float(
                deterministic.get(
                    "numerical_tolerance"
                ),
                (
                    "conditional_redundancy.methods."
                    "deterministic_transformation."
                    "numerical_tolerance"
                ),
            ),

        "minimum_absolute_partial":
            require_non_negative_float(
                residual_information.get(
                    "minimum_absolute_partial_spearman"
                ),
                (
                    "conditional_redundancy."
                    "residual_information."
                    "minimum_absolute_partial_spearman"
                ),
            ),

        "minimum_absolute_residual":
            require_non_negative_float(
                residual_information.get(
                    "minimum_absolute_residual_spearman"
                ),
                (
                    "conditional_redundancy."
                    "residual_information."
                    "minimum_absolute_residual_spearman"
                ),
            ),

        "minimum_methods_supporting":
            require_positive_integer(
                residual_information.get(
                    "minimum_methods_supporting_"
                    "residual_information"
                ),
                (
                    "conditional_redundancy."
                    "residual_information."
                    "minimum_methods_supporting_"
                    "residual_information"
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
    }


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
            "Conditional redundancy found no baseline rows."
        )

    return frame


def prepare_numeric_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Prepare a finite complete numeric table."""
    missing_columns = sorted(
        set(
            columns
        )
        - set(
            frame.columns
        )
    )

    if missing_columns:
        fail(
            "Conditional frame is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    output = frame.loc[
        :,
        columns,
    ].copy()

    for column in columns:
        output[
            column
        ] = pd.to_numeric(
            output[
                column
            ],
            errors="coerce",
        )

    return output.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna(
        subset=list(
            columns
        )
    ).reset_index(
        drop=True
    )


def rank_matrix(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Convert each column into average ranks."""
    return np.column_stack(
        [
            rankdata(
                frame[
                    column
                ].to_numpy(
                    dtype=float
                ),
                method="average",
            )
            for column in frame.columns
        ]
    )


def regress_residuals(
    response: np.ndarray,
    controls: np.ndarray,
) -> np.ndarray:
    """Return linear residuals with an intercept."""
    design = np.column_stack(
        [
            np.ones(
                len(
                    response
                ),
                dtype=float,
            ),
            controls,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        response,
        rcond=None,
    )

    fitted = design @ coefficients

    return response - fitted


def partial_spearman(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
    control_ids: tuple[str, ...],
    *,
    minimum_observations: int,
) -> dict[str, Any]:
    """Estimate partial Spearman through rank residualization."""
    columns = (
        source_id,
        target_id,
        *control_ids,
    )

    complete = prepare_numeric_frame(
        frame,
        columns,
    )

    sample_count = int(
        len(
            complete
        )
    )

    if sample_count < minimum_observations:
        return {
            "estimability":
                "non_estimable",

            "reason":
                "insufficient_complete_observations",

            "sample_count":
                sample_count,

            "strength":
                None,
        }

    ranked = rank_matrix(
        complete
    )

    source_rank = ranked[
        :,
        0,
    ]

    target_rank = ranked[
        :,
        1,
    ]

    controls = ranked[
        :,
        2:,
    ]

    if controls.shape[1] == 0:
        result = spearmanr(
            source_rank,
            target_rank,
        )

        strength = float(
            result.statistic
        )
    else:
        source_residual = regress_residuals(
            source_rank,
            controls,
        )

        target_residual = regress_residuals(
            target_rank,
            controls,
        )

        result = spearmanr(
            source_residual,
            target_residual,
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

            "reason":
                "non_finite_partial_estimate",

            "sample_count":
                sample_count,

            "strength":
                None,
        }

    return {
        "estimability":
            "estimable",

        "reason":
            None,

        "sample_count":
            sample_count,

        "strength":
            strength,
    }


def residual_spearman(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
    control_ids: tuple[str, ...],
    *,
    minimum_observations: int,
) -> dict[str, Any]:
    """Estimate rank-linear residual Spearman association."""
    columns = (
        source_id,
        target_id,
        *control_ids,
    )

    complete = prepare_numeric_frame(
        frame,
        columns,
    )

    sample_count = int(
        len(
            complete
        )
    )

    if sample_count < minimum_observations:
        return {
            "estimability":
                "non_estimable",

            "reason":
                "insufficient_complete_observations",

            "sample_count":
                sample_count,

            "strength":
                None,
        }

    ranked = rank_matrix(
        complete
    )

    source_rank = ranked[
        :,
        0,
    ]

    target_rank = ranked[
        :,
        1,
    ]

    controls = ranked[
        :,
        2:,
    ]

    if controls.shape[1] == 0:
        source_residual = source_rank
        target_residual = target_rank
    else:
        source_residual = regress_residuals(
            source_rank,
            controls,
        )

        target_residual = regress_residuals(
            target_rank,
            controls,
        )

    result = spearmanr(
        source_residual,
        target_residual,
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

            "reason":
                "non_finite_residual_estimate",

            "sample_count":
                sample_count,

            "strength":
                None,
        }

    return {
        "estimability":
            "estimable",

        "reason":
            None,

        "sample_count":
            sample_count,

        "strength":
            strength,
    }


def affine_deterministic_equivalence(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Test approximate affine deterministic equivalence."""
    pair = prepare_numeric_frame(
        frame,
        (
            source_id,
            target_id,
        ),
    )

    if len(
        pair
    ) < 3:
        return {
            "estimability":
                "non_estimable",

            "equivalent":
                False,

            "intercept":
                None,

            "slope":
                None,

            "maximum_scaled_residual":
                None,
        }

    source = pair[
        source_id
    ].to_numpy(
        dtype=float
    )

    target = pair[
        target_id
    ].to_numpy(
        dtype=float
    )

    if np.unique(
        source
    ).size < 2:
        return {
            "estimability":
                "non_estimable",

            "equivalent":
                False,

            "intercept":
                None,

            "slope":
                None,

            "maximum_scaled_residual":
                None,
        }

    design = np.column_stack(
        [
            np.ones(
                len(
                    source
                )
            ),
            source,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    fitted = design @ coefficients

    maximum_residual = float(
        np.max(
            np.abs(
                target
                - fitted
            )
        )
    )

    target_scale = max(
        1.0,
        float(
            np.max(
                np.abs(
                    target
                )
            )
        ),
    )

    scaled_residual = float(
        maximum_residual
        / target_scale
    )

    return {
        "estimability":
            "estimable",

        "equivalent":
            bool(
                scaled_residual
                <= tolerance
            ),

        "intercept":
            float(
                coefficients[
                    0
                ]
            ),

        "slope":
            float(
                coefficients[
                    1
                ]
            ),

        "maximum_scaled_residual":
            scaled_residual,
    }


def choose_control_features(
    source_id: str,
    target_id: str,
    dominant_features: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Select dominant controls excluding the relation endpoints.

    Conditioning a feature on itself would force a trivial residual
    and is therefore prohibited.
    """
    return tuple(
        feature_id
        for feature_id in dominant_features
        if feature_id
        not in {
            source_id,
            target_id,
        }
    )


def get_pair_id(
    source_id: str,
    target_id: str,
) -> str:
    """Return a canonical unordered pair identifier."""
    left, right = sorted(
        (
            str(
                source_id
            ),
            str(
                target_id
            ),
        )
    )

    return (
        f"{left}--{right}"
    )


def evaluate_decision_equivalence(
    row: Any,
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """
    Compare the relation with other scales of the same component pair.

    This is descriptive. It does not itself establish full redundancy.
    """
    multiscale = context.outputs[
        "multiscale_relations"
    ]

    if not isinstance(
        multiscale,
        pd.DataFrame,
    ):
        fail(
            "multiscale_relations must be a DataFrame."
        )

    pair_id = get_pair_id(
        str(
            row.source_id
        ),
        str(
            row.target_id
        ),
    )

    pair_rows = multiscale.loc[
        multiscale[
            "pair_id"
        ].astype(str)
        == pair_id
    ]

    if len(
        pair_rows
    ) != 1:
        return {
            "pair_id":
                pair_id,

            "multiscale_record_found":
                False,

            "supported_scale_count":
                None,

            "multiscale_class":
                None,

            "decision_equivalent_across_scales":
                False,
        }

    pair_row = pair_rows.iloc[
        0
    ]

    supported_scale_count = int(
        pair_row[
            "supported_scale_count"
        ]
    )

    multiscale_class = str(
        pair_row[
            "multiscale_class"
        ]
    )

    equivalent = bool(
        supported_scale_count >= 2
        and multiscale_class
        in {
            "multi_scale_2",
            "multi_scale_3",
            "multi_scale_4",
        }
    )

    return {
        "pair_id":
            pair_id,

        "multiscale_record_found":
            True,

        "supported_scale_count":
            supported_scale_count,

        "multiscale_class":
            multiscale_class,

        "decision_equivalent_across_scales":
            equivalent,
    }


def classify_redundancy(
    *,
    partial_strength: float | None,
    residual_strength: float | None,
    deterministic_equivalent: bool,
    minimum_absolute_partial: float,
    minimum_absolute_residual: float,
    minimum_methods_supporting: int,
) -> dict[str, Any]:
    """Assign one frozen conditional-redundancy class."""
    partial_support = bool(
        partial_strength is not None
        and abs(
            partial_strength
        )
        >= minimum_absolute_partial
    )

    residual_support = bool(
        residual_strength is not None
        and abs(
            residual_strength
        )
        >= minimum_absolute_residual
    )

    supported_method_count = int(
        partial_support
        + residual_support
    )

    estimable_method_count = int(
        partial_strength is not None
    ) + int(
        residual_strength is not None
    )

    if deterministic_equivalent:
        status = "fully_redundant"

    elif estimable_method_count == 0:
        status = (
            "conditional_test_inconclusive"
        )

    elif (
        supported_method_count
        >= minimum_methods_supporting
        and supported_method_count
        == estimable_method_count
    ):
        status = (
            "retains_residual_information"
        )

    elif (
        supported_method_count
        >= minimum_methods_supporting
    ):
        status = "partially_redundant"

    elif supported_method_count == 0:
        status = "fully_redundant"

    else:
        status = (
            "conditional_test_inconclusive"
        )

    return {
        "partial_supports_residual_information":
            partial_support,

        "residual_supports_residual_information":
            residual_support,

        "supported_method_count":
            supported_method_count,

        "estimable_method_count":
            estimable_method_count,

        "retains_residual_information":
            bool(
                status
                in {
                    "retains_residual_information",
                    "partially_redundant",
                }
            ),

        "redundancy_status":
            status,

        "fully_redundant_flag":
            bool(
                status
                == "fully_redundant"
            ),
    }


def audit_relation(
    frame: pd.DataFrame,
    row: Any,
    context: ID03ExperimentContext,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Run conditional redundancy analyses for one relation."""
    source_id = str(
        row.source_id
    )

    target_id = str(
        row.target_id
    )

    control_ids = choose_control_features(
        source_id,
        target_id,
        tuple(
            contract[
                "dominant_activity_features"
            ]
        ),
    )

    partial = partial_spearman(
        frame,
        source_id,
        target_id,
        control_ids,
        minimum_observations=contract[
            "partial_minimum_observations"
        ],
    )

    residual = residual_spearman(
        frame,
        source_id,
        target_id,
        control_ids,
        minimum_observations=contract[
            "residual_minimum_observations"
        ],
    )

    deterministic = (
        affine_deterministic_equivalence(
            frame,
            source_id,
            target_id,
            tolerance=contract[
                "deterministic_tolerance"
            ],
        )
    )

    decision = evaluate_decision_equivalence(
        row,
        context,
    )

    partial_strength = partial[
        "strength"
    ]

    residual_strength = residual[
        "strength"
    ]

    classification = classify_redundancy(
        partial_strength=partial_strength,
        residual_strength=residual_strength,
        deterministic_equivalent=bool(
            deterministic[
                "equivalent"
            ]
        ),
        minimum_absolute_partial=contract[
            "minimum_absolute_partial"
        ],
        minimum_absolute_residual=contract[
            "minimum_absolute_residual"
        ],
        minimum_methods_supporting=contract[
            "minimum_methods_supporting"
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

        "pair_id":
            decision[
                "pair_id"
            ],

        "source_id":
            source_id,

        "target_id":
            target_id,

        "id02_status":
            str(
                row.classification_status
            ),

        "control_features":
            "|".join(
                control_ids
            ),

        "control_feature_count":
            int(
                len(
                    control_ids
                )
            ),

        "partial_spearman_estimability":
            partial[
                "estimability"
            ],

        "partial_spearman_reason":
            partial[
                "reason"
            ],

        "partial_spearman_sample_count":
            int(
                partial[
                    "sample_count"
                ]
            ),

        "partial_spearman_strength":
            partial_strength,

        "residual_spearman_estimability":
            residual[
                "estimability"
            ],

        "residual_spearman_reason":
            residual[
                "reason"
            ],

        "residual_spearman_sample_count":
            int(
                residual[
                    "sample_count"
                ]
            ),

        "residual_spearman_strength":
            residual_strength,

        "deterministic_audit_estimability":
            deterministic[
                "estimability"
            ],

        "deterministic_affine_equivalent":
            bool(
                deterministic[
                    "equivalent"
                ]
            ),

        "deterministic_affine_intercept":
            deterministic[
                "intercept"
            ],

        "deterministic_affine_slope":
            deterministic[
                "slope"
            ],

        "deterministic_maximum_scaled_residual":
            deterministic[
                "maximum_scaled_residual"
            ],

        "supported_scale_count":
            decision[
                "supported_scale_count"
            ],

        "multiscale_class":
            decision[
                "multiscale_class"
            ],

        "decision_equivalent_across_scales":
            decision[
                "decision_equivalent_across_scales"
            ],

        **classification,

        "id02_status_modified":
            False,
    }


def build_redundancy_table(
    context: ID03ExperimentContext,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Audit all 74 primary relations."""
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
            "Redundancy audit expects 74 primary relations."
        )

    baseline = select_baseline(
        context,
        timestamp_column=contract[
            "timestamp_column"
        ],
    )

    records: list[dict[str, Any]] = []

    for row in primary.itertuples(
        index=False
    ):
        window_frame = baseline.loc[
            baseline[
                contract[
                    "window_column"
                ]
            ].astype(str)
            == str(
                row.window_id
            )
        ].copy()

        if window_frame.empty:
            fail(
                "No baseline rows exist for window "
                f"{row.window_id}."
            )

        records.append(
            audit_relation(
                window_frame,
                row,
                context,
                contract,
            )
        )

    output = pd.DataFrame.from_records(
        records
    )

    if len(
        output
    ) != 74:
        fail(
            "Conditional redundancy output must contain "
            f"74 rows; observed {len(output)}."
        )

    if output.duplicated(
        subset=[
            "window_id",
            "relation_id",
        ],
        keep=False,
    ).any():
        fail(
            "Conditional redundancy contains "
            "duplicate window-relation keys."
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


def build_redundancy_summary(
    redundancy: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive conditional-redundancy counts."""
    status_counts = (
        redundancy[
            "redundancy_status"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    by_window: dict[str, Any] = {}

    for window_id, frame in redundancy.groupby(
        "window_id",
        sort=True,
    ):
        counts = (
            frame[
                "redundancy_status"
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

            "status_counts": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in counts.items()
            },

            "retains_residual_information_count":
                int(
                    frame[
                        "retains_residual_information"
                    ].sum()
                ),

            "fully_redundant_count":
                int(
                    frame[
                        "fully_redundant_flag"
                    ].sum()
                ),

            "deterministic_affine_equivalent_count":
                int(
                    frame[
                        "deterministic_affine_equivalent"
                    ].sum()
                ),
        }

    return {
        "status":
            "COMPLETED",

        "primary_relation_count":
            int(
                len(
                    redundancy
                )
            ),

        "redundancy_status_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in status_counts.items()
        },

        "retains_residual_information_count":
            int(
                redundancy[
                    "retains_residual_information"
                ].sum()
            ),

        "fully_redundant_count":
            int(
                redundancy[
                    "fully_redundant_flag"
                ].sum()
            ),

        "deterministic_affine_equivalent_count":
            int(
                redundancy[
                    "deterministic_affine_equivalent"
                ].sum()
            ),

        "by_window":
            by_window,

        "id02_statuses_modified":
            False,
    }


def audit_redundancy(
    context: ID03ExperimentContext,
) -> ID03ExperimentContext:
    """Execute the frozen conditional-redundancy audit."""
    validate_context(
        context
    )

    contract = get_redundancy_contract(
        context
    )

    redundancy = build_redundancy_table(
        context,
        contract,
    )

    summary = build_redundancy_summary(
        redundancy
    )

    context.register_output(
        "conditional_redundancy",
        redundancy,
    )

    context.register_output(
        "conditional_redundancy_summary",
        summary,
    )

    context.register_runtime(
        "conditional_redundancy_status",
        "COMPLETED",
    )

    return context

