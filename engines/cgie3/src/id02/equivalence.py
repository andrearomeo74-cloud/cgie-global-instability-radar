"""
CGIE3-ID-02 equivalence and redundancy audit.

This stage detects possible duplication or deterministic equivalence
among declared components and flags relations that may be redundant.

Flags are descriptive only.

They do not automatically reject, promote or reclassify relations.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from congruity.core import ExperimentContext


class EquivalenceStageError(ValueError):
    """Raised when the equivalence audit violates its contract."""


def fail(message: str) -> None:
    """Raise an equivalence-stage error."""
    raise EquivalenceStageError(
        str(message).strip()
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like value."""
    if not isinstance(value, Mapping):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def require_fraction(
    value: Any,
    field_name: str,
) -> float:
    """Require a finite value inside [0, 1]."""
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise EquivalenceStageError(
            f"{field_name} must be numeric."
        ) from exc

    if (
        not np.isfinite(normalized)
        or not 0.0 <= normalized <= 1.0
    ):
        fail(
            f"{field_name} must lie inside [0, 1]."
        )

    return normalized


def validate_context(
    context: ExperimentContext,
) -> None:
    """Validate equivalence-stage prerequisites."""
    if not isinstance(
        context,
        ExperimentContext,
    ):
        fail(
            "context must be an ExperimentContext."
        )

    if context.experiment_id != "CGIE3_ID_02":
        fail(
            "Unexpected experiment ID: "
            f"{context.experiment_id}"
        )

    required_outputs = {
        "baseline_feature_tables",
        "components_by_window",
        "relation_classification",
    }

    missing = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing:
        fail(
            "Equivalence prerequisites are missing: "
            + ", ".join(missing)
        )


def get_equivalence_contract(
    context: ExperimentContext,
) -> dict[str, Any]:
    """Extract the frozen equivalence configuration."""
    checks = require_mapping(
        context.configuration.get(
            "equivalence_checks"
        ),
        "equivalence_checks",
    )

    duplicate_detection = require_mapping(
        checks.get(
            "duplicate_component_detection"
        ),
        "equivalence_checks."
        "duplicate_component_detection",
    )

    transformation_detection = require_mapping(
        checks.get(
            "deterministic_transformation_detection"
        ),
        "equivalence_checks."
        "deterministic_transformation_detection",
    )

    redundant_relations = require_mapping(
        checks.get(
            "redundant_relation_flagging"
        ),
        "equivalence_checks."
        "redundant_relation_flagging",
    )

    if duplicate_detection.get("enabled") is not True:
        fail(
            "Duplicate-component detection must remain enabled."
        )

    if transformation_detection.get("enabled") is not True:
        fail(
            "Deterministic-transformation detection "
            "must remain enabled."
        )

    if redundant_relations.get("enabled") is not True:
        fail(
            "Redundant-relation flagging must remain enabled."
        )

    if (
        redundant_relations.get(
            "does_not_automatically_reject"
        )
        is not True
    ):
        fail(
            "Redundancy flags must not automatically "
            "reject relations."
        )

    threshold = require_fraction(
        duplicate_detection.get(
            "absolute_spearman_threshold"
        ),
        "equivalence_checks."
        "duplicate_component_detection."
        "absolute_spearman_threshold",
    )

    return {
        "duplicate_threshold":
            threshold,
    }


def prepare_pair(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
) -> pd.DataFrame:
    """Prepare finite paired numeric observations."""
    pair = frame.loc[
        :,
        [
            source_id,
            target_id,
        ],
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


def affine_equivalence(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> tuple[bool, float | None, float | None]:
    """
    Test whether target is approximately an affine transformation
    of source: target = intercept + slope * source.
    """
    if source.size < 3:
        return False, None, None

    if np.unique(source).size < 2:
        return False, None, None

    design = np.column_stack(
        [
            np.ones(source.size),
            source,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    fitted = design @ coefficients

    residual_scale = float(
        np.max(
            np.abs(
                target - fitted
            )
        )
    )

    target_scale = max(
        1.0,
        float(
            np.max(
                np.abs(target)
            )
        ),
    )

    equivalent = bool(
        residual_scale
        <= tolerance * target_scale
    )

    return (
        equivalent,
        float(coefficients[0]),
        float(coefficients[1]),
    )


def audit_window(
    frame: pd.DataFrame,
    component_ids: tuple[str, ...],
    classification: pd.DataFrame,
    *,
    window_id: str,
    duplicate_threshold: float,
) -> pd.DataFrame:
    """Audit all candidate component pairs in one window."""
    records: list[dict[str, Any]] = []

    window_classification = classification.loc[
        classification["window_id"]
        == window_id
    ].copy()

    classification_by_relation = {
        str(row.relation_id): row
        for row in window_classification.itertuples(
            index=False
        )
    }

    for source_index, source_id in enumerate(
        component_ids
    ):
        for target_id in component_ids[
            source_index + 1:
        ]:
            pair = prepare_pair(
                frame,
                source_id,
                target_id,
            )

            sample_count = int(
                len(pair)
            )

            relation_match = (
                window_classification.loc[
                    (
                        window_classification[
                            "source_id"
                        ]
                        == source_id
                    )
                    & (
                        window_classification[
                            "target_id"
                        ]
                        == target_id
                    )
                ]
            )

            if relation_match.empty:
                relation_match = (
                    window_classification.loc[
                        (
                            window_classification[
                                "source_id"
                            ]
                            == target_id
                        )
                        & (
                            window_classification[
                                "target_id"
                            ]
                            == source_id
                        )
                    ]
                )

            if len(relation_match) != 1:
                fail(
                    "Unable to identify exactly one "
                    f"classified relation for "
                    f"{window_id}: {source_id}--{target_id}."
                )

            relation_id = str(
                relation_match.iloc[0][
                    "relation_id"
                ]
            )

            relation_row = (
                classification_by_relation[
                    relation_id
                ]
            )

            if sample_count < 3:
                spearman_strength = np.nan
                duplicate_flag = False
                affine_flag = False
                intercept = np.nan
                slope = np.nan
                audit_status = "non_estimable"
            else:
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

                result = spearmanr(
                    source_values,
                    target_values,
                    nan_policy="omit",
                )

                spearman_strength = float(
                    result.statistic
                )

                if not np.isfinite(
                    spearman_strength
                ):
                    duplicate_flag = False
                    affine_flag = False
                    intercept = np.nan
                    slope = np.nan
                    audit_status = "non_estimable"
                else:
                    duplicate_flag = bool(
                        abs(spearman_strength)
                        >= duplicate_threshold
                    )

                    (
                        affine_flag,
                        intercept_value,
                        slope_value,
                    ) = affine_equivalence(
                        source_values,
                        target_values,
                    )

                    intercept = (
                        intercept_value
                        if intercept_value
                        is not None
                        else np.nan
                    )

                    slope = (
                        slope_value
                        if slope_value
                        is not None
                        else np.nan
                    )

                    audit_status = "evaluated"

            redundancy_flag = bool(
                duplicate_flag
                or affine_flag
            )

            records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_02",
                    "window_id":
                        window_id,
                    "relation_id":
                        relation_id,
                    "source_id":
                        source_id,
                    "target_id":
                        target_id,
                    "classification_status":
                        relation_row.classification_status,
                    "paired_observation_count":
                        sample_count,
                    "equivalence_audit_status":
                        audit_status,
                    "absolute_spearman_threshold":
                        duplicate_threshold,
                    "observed_spearman_strength":
                        spearman_strength,
                    "near_duplicate_component_flag":
                        duplicate_flag,
                    "affine_transformation_flag":
                        affine_flag,
                    "affine_intercept":
                        intercept,
                    "affine_slope":
                        slope,
                    "redundant_relation_flag":
                        redundancy_flag,
                    "automatic_rejection_applied":
                        False,
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def build_equivalence_summary(
    flags: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive equivalence-audit counts."""
    by_window: dict[str, Any] = {}

    for window_id, frame in flags.groupby(
        "window_id",
        sort=True,
    ):
        by_window[str(window_id)] = {
            "relation_count":
                int(len(frame)),
            "near_duplicate_component_count":
                int(
                    frame[
                        "near_duplicate_component_flag"
                    ].sum()
                ),
            "affine_transformation_count":
                int(
                    frame[
                        "affine_transformation_flag"
                    ].sum()
                ),
            "redundant_relation_flag_count":
                int(
                    frame[
                        "redundant_relation_flag"
                    ].sum()
                ),
            "non_estimable_audit_count":
                int(
                    (
                        frame[
                            "equivalence_audit_status"
                        ]
                        == "non_estimable"
                    ).sum()
                ),
        }

    return {
        "status":
            "COMPLETED",
        "relation_count":
            int(len(flags)),
        "redundant_relation_flag_count":
            int(
                flags[
                    "redundant_relation_flag"
                ].sum()
            ),
        "automatic_rejections":
            0,
        "classification_modified":
            False,
        "by_window":
            by_window,
    }


def evaluate_equivalence(
    context: ExperimentContext,
) -> ExperimentContext:
    """Execute the frozen equivalence and redundancy audit."""
    validate_context(
        context
    )

    contract = get_equivalence_contract(
        context
    )

    baseline_tables = require_mapping(
        context.outputs[
            "baseline_feature_tables"
        ],
        "baseline_feature_tables",
    )

    components_by_window = require_mapping(
        context.outputs[
            "components_by_window"
        ],
        "components_by_window",
    )

    classification = context.outputs[
        "relation_classification"
    ]

    if not isinstance(
        classification,
        pd.DataFrame,
    ):
        fail(
            "relation_classification must be "
            "a pandas DataFrame."
        )

    audit_frames: list[pd.DataFrame] = []

    for window_id in sorted(
        baseline_tables
    ):
        audit_frame = audit_window(
            frame=baseline_tables[
                window_id
            ],
            component_ids=tuple(
                components_by_window[
                    window_id
                ]
            ),
            classification=classification,
            window_id=window_id,
            duplicate_threshold=contract[
                "duplicate_threshold"
            ],
        )

        audit_frames.append(
            audit_frame
        )

    flags = pd.concat(
        audit_frames,
        ignore_index=True,
    )

    flags = flags.sort_values(
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

    if len(flags) != 136:
        fail(
            "Equivalence audit must contain "
            f"136 relations, observed {len(flags)}."
        )

    if flags.duplicated(
        subset=[
            "window_id",
            "relation_id",
        ]
    ).any():
        fail(
            "Equivalence audit contains duplicate "
            "window-relation keys."
        )

    summary = build_equivalence_summary(
        flags
    )

    context.register_output(
        "equivalence_flags",
        flags,
    )

    context.register_output(
        "equivalence_summary",
        summary,
    )

    context.register_runtime(
        "equivalence_status",
        "COMPLETED",
    )

    return context
  
