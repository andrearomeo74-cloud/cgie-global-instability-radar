"""
CGIE3-ID-02 persistence stage.

This module evaluates candidate relations across frozen contiguous
calendar blocks of the baseline.

It measures:

- block estimability;
- sign preservation;
- minimum-strength preservation;
- relation persistence;
- leave-one-block-out robustness.

It does not classify relations as eligible, candidate or rejected.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from congruity.core import ExperimentContext
from congruity.relations import (
    RelationDiscoveryError,
    discover_spearman_relations,
    relation_estimates_to_frame,
)


class PersistenceStageError(ValueError):
    """Raised when persistence evaluation violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a persistence-stage error."""
    raise PersistenceStageError(
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
    """Require a numeric fraction inside [0, 1]."""
    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise PersistenceStageError(
            f"{field_name} must be numeric."
        ) from exc

    if not 0.0 <= normalized <= 1.0:
        fail(
            f"{field_name} must lie inside [0, 1]."
        )

    return normalized


def require_non_negative_float(
    value: Any,
    field_name: str,
) -> float:
    """Require a finite non-negative float."""
    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise PersistenceStageError(
            f"{field_name} must be numeric."
        ) from exc

    if (
        not np.isfinite(normalized)
        or normalized < 0.0
    ):
        fail(
            f"{field_name} must be finite and non-negative."
        )

    return normalized


def validate_context(
    context: ExperimentContext,
) -> None:
    """Validate persistence-stage prerequisites."""
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
        "candidate_relations",
        "candidate_relations_by_window",
    }

    missing_outputs = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing_outputs:
        fail(
            "Relation-discovery outputs are missing: "
            + ", ".join(missing_outputs)
        )


def get_persistence_contract(
    context: ExperimentContext,
) -> dict[str, Any]:
    """Extract frozen persistence and block rules."""
    partition = require_mapping(
        context.configuration.get(
            "baseline_partition"
        ),
        "baseline_partition",
    )

    persistence = require_mapping(
        context.configuration.get(
            "persistence"
        ),
        "persistence",
    )

    reproducibility = require_mapping(
        context.configuration.get(
            "relation_reproducibility"
        ),
        "relation_reproducibility",
    )

    block_estimates = require_mapping(
        reproducibility.get(
            "block_estimates"
        ),
        "relation_reproducibility.block_estimates",
    )

    robustness = require_mapping(
        context.configuration.get(
            "robustness"
        ),
        "robustness",
    )

    leave_one_out = require_mapping(
        robustness.get(
            "leave_one_block_out"
        ),
        "robustness.leave_one_block_out",
    )

    method = str(
        partition.get(
            "method",
            ""
        )
    ).strip()

    if method != "contiguous_calendar_blocks":
        fail(
            "Only contiguous_calendar_blocks is supported."
        )

    block_count = int(
        partition.get(
            "block_count"
        )
    )

    if block_count != 12:
        fail(
            "CGIE3-ID-02 requires exactly 12 baseline blocks."
        )

    if (
        partition.get(
            "preserve_temporal_order"
        )
        is not True
    ):
        fail(
            "Temporal block order must be preserved."
        )

    if (
        block_estimates.get("enabled")
        is not True
    ):
        fail(
            "Block relation estimates must remain enabled."
        )

    if (
        leave_one_out.get("enabled")
        is not True
    ):
        fail(
            "Leave-one-block-out robustness must remain enabled."
        )

    minimum_valid_blocks = int(
        partition.get(
            "minimum_valid_blocks"
        )
    )

    if not 1 <= minimum_valid_blocks <= block_count:
        fail(
            "baseline_partition.minimum_valid_blocks "
            "is outside the valid range."
        )

    minimum_samples_per_block = int(
        block_estimates.get(
            "minimum_paired_observations_per_block"
        )
    )

    if minimum_samples_per_block < 3:
        fail(
            "Block minimum paired observations "
            "must be at least 3."
        )

    return {
        "block_count":
            block_count,
        "minimum_valid_blocks":
            minimum_valid_blocks,
        "minimum_samples_per_block":
            minimum_samples_per_block,
        "minimum_block_strength":
            require_non_negative_float(
                persistence.get(
                    "minimum_block_absolute_strength"
                ),
                "persistence."
                "minimum_block_absolute_strength",
            ),
        "minimum_estimable_fraction":
            require_fraction(
                persistence.get(
                    "minimum_estimable_block_fraction"
                ),
                "persistence."
                "minimum_estimable_block_fraction",
            ),
        "minimum_sign_fraction":
            require_fraction(
                persistence.get(
                    "minimum_sign_preservation_fraction"
                ),
                "persistence."
                "minimum_sign_preservation_fraction",
            ),
        "loo_minimum_sign_fraction":
            require_fraction(
                leave_one_out.get(
                    "minimum_sign_preservation_fraction"
                ),
                "robustness.leave_one_block_out."
                "minimum_sign_preservation_fraction",
            ),
        "loo_maximum_median_change":
            require_non_negative_float(
                leave_one_out.get(
                    "maximum_median_absolute_strength_change"
                ),
                "robustness.leave_one_block_out."
                "maximum_median_absolute_strength_change",
            ),
    }


def assign_calendar_blocks(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Assign each baseline observation to its UTC calendar month.

    The frozen 2025 baseline must produce exactly 12 ordered blocks.
    """
    output = frame.copy()

    if timestamp_column not in output.columns:
        fail(
            f"Timestamp column is absent: {timestamp_column}"
        )

    output["_block_id"] = (
        output[timestamp_column]
        .dt.strftime("%Y-%m")
    )

    observed_blocks = tuple(
        sorted(
            output["_block_id"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    expected_blocks = tuple(
        f"2025-{month:02d}"
        for month in range(1, 13)
    )

    if observed_blocks != expected_blocks:
        fail(
            "Observed baseline calendar blocks differ "
            "from the frozen 2025 monthly sequence. "
            f"Observed: {observed_blocks}"
        )

    return output


def estimate_block_relations(
    frame: pd.DataFrame,
    component_ids: tuple[str, ...],
    *,
    window_id: str,
    timestamp_column: str,
    minimum_samples: int,
) -> pd.DataFrame:
    """Estimate all candidate relations independently in each month."""
    blocked_frame = assign_calendar_blocks(
        frame,
        timestamp_column=timestamp_column,
    )

    block_frames: list[pd.DataFrame] = []

    for block_id in sorted(
        blocked_frame["_block_id"].unique()
    ):
        block = blocked_frame.loc[
            blocked_frame["_block_id"]
            == block_id
        ].copy()

        timestamp_utc = (
            block[timestamp_column]
            .max()
            .isoformat()
        )

        try:
            estimates = discover_spearman_relations(
                frame=block,
                component_ids=component_ids,
                window_id=window_id,
                timestamp_utc=timestamp_utc,
                minimum_samples=minimum_samples,
                metadata={
                    "experiment_id":
                        "CGIE3_ID_02",
                    "estimate_scope":
                        "baseline_calendar_block",
                    "block_id":
                        block_id,
                    "uses_evaluation_data":
                        False,
                    "uses_target_event":
                        False,
                },
            )
        except RelationDiscoveryError as exc:
            raise PersistenceStageError(
                f"Block relation discovery failed for "
                f"{window_id}/{block_id}: {exc}"
            ) from exc

        block_relation_frame = (
            relation_estimates_to_frame(
                estimates
            )
        )

        block_relation_frame.insert(
            0,
            "block_id",
            block_id,
        )

        block_relation_frame.insert(
            0,
            "experiment_id",
            "CGIE3_ID_02",
        )

        block_frames.append(
            block_relation_frame
        )

    if not block_frames:
        fail(
            f"No block estimates were produced for {window_id}."
        )

    return pd.concat(
        block_frames,
        ignore_index=True,
    ).sort_values(
        by=[
            "window_id",
            "block_id",
            "source_id",
            "target_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def summarize_relation_persistence(
    full_relations: pd.DataFrame,
    block_relations: pd.DataFrame,
    *,
    minimum_block_strength: float,
    minimum_valid_blocks: int,
) -> pd.DataFrame:
    """Summarize block stability for every full-baseline relation."""
    records: list[dict[str, Any]] = []

    for full_row in full_relations.itertuples(
        index=False
    ):
        relation_blocks = block_relations.loc[
            block_relations["relation_id"]
            == full_row.relation_id
        ].copy()

        estimable = relation_blocks.loc[
            relation_blocks["estimability"]
            == "estimable"
        ].copy()

        valid_block_count = int(
            len(estimable)
        )

        total_block_count = int(
            len(relation_blocks)
        )

        estimable_fraction = (
            valid_block_count
            / total_block_count
            if total_block_count
            else 0.0
        )

        if (
            full_row.estimability
            != "estimable"
            or full_row.sign is None
            or pd.isna(full_row.sign)
        ):
            sign_preservation_fraction = np.nan
            strength_preservation_fraction = np.nan
            persistence_value = np.nan
        elif valid_block_count == 0:
            sign_preservation_fraction = 0.0
            strength_preservation_fraction = 0.0
            persistence_value = 0.0
        else:
            sign_preserved = (
                estimable["sign"]
                == int(full_row.sign)
            )

            strength_preserved = (
                estimable["absolute_strength"]
                >= minimum_block_strength
            )

            jointly_preserved = (
                sign_preserved
                & strength_preserved
            )

            sign_preservation_fraction = float(
                sign_preserved.mean()
            )

            strength_preservation_fraction = float(
                strength_preserved.mean()
            )

            persistence_value = float(
                jointly_preserved.sum()
                / total_block_count
            )

        block_strengths = (
            estimable["strength"]
            .dropna()
            .astype(float)
        )

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_02",
                "window_id":
                    full_row.window_id,
                "relation_id":
                    full_row.relation_id,
                "source_id":
                    full_row.source_id,
                "target_id":
                    full_row.target_id,
                "full_baseline_strength":
                    full_row.strength,
                "full_baseline_absolute_strength":
                    full_row.absolute_strength,
                "full_baseline_sign":
                    full_row.sign,
                "total_block_count":
                    total_block_count,
                "valid_block_count":
                    valid_block_count,
                "minimum_valid_blocks_met":
                    (
                        valid_block_count
                        >= minimum_valid_blocks
                    ),
                "estimable_block_fraction":
                    estimable_fraction,
                "sign_preservation_fraction":
                    sign_preservation_fraction,
                "strength_preservation_fraction":
                    strength_preservation_fraction,
                "persistence":
                    persistence_value,
                "median_block_strength":
                    (
                        float(
                            block_strengths.median()
                        )
                        if not block_strengths.empty
                        else np.nan
                    ),
                "median_block_absolute_strength":
                    (
                        float(
                            block_strengths.abs().median()
                        )
                        if not block_strengths.empty
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame.from_records(
        records
    ).sort_values(
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


def evaluate_leave_one_block_out(
    frame: pd.DataFrame,
    component_ids: tuple[str, ...],
    full_relations: pd.DataFrame,
    *,
    window_id: str,
    timestamp_column: str,
    minimum_samples: int,
    minimum_sign_fraction: float,
    maximum_median_change: float,
) -> pd.DataFrame:
    """Evaluate relation stability after removing each baseline month."""
    blocked_frame = assign_calendar_blocks(
        frame,
        timestamp_column=timestamp_column,
    )

    estimates_by_omission: list[
        pd.DataFrame
    ] = []

    for omitted_block in sorted(
        blocked_frame["_block_id"].unique()
    ):
        reduced_frame = blocked_frame.loc[
            blocked_frame["_block_id"]
            != omitted_block
        ].copy()

        timestamp_utc = (
            reduced_frame[timestamp_column]
            .max()
            .isoformat()
        )

        try:
            estimates = discover_spearman_relations(
                frame=reduced_frame,
                component_ids=component_ids,
                window_id=window_id,
                timestamp_utc=timestamp_utc,
                minimum_samples=minimum_samples,
                metadata={
                    "experiment_id":
                        "CGIE3_ID_02",
                    "estimate_scope":
                        "leave_one_block_out",
                    "omitted_block":
                        omitted_block,
                },
            )
        except RelationDiscoveryError as exc:
            raise PersistenceStageError(
                "Leave-one-block-out discovery failed "
                f"for {window_id}/{omitted_block}: {exc}"
            ) from exc

        estimate_frame = relation_estimates_to_frame(
            estimates
        )

        estimate_frame.insert(
            0,
            "omitted_block",
            omitted_block,
        )

        estimates_by_omission.append(
            estimate_frame
        )

    combined = pd.concat(
        estimates_by_omission,
        ignore_index=True,
    )

    records: list[dict[str, Any]] = []

    for full_row in full_relations.itertuples(
        index=False
    ):
        relation_rows = combined.loc[
            combined["relation_id"]
            == full_row.relation_id
        ].copy()

        estimable = relation_rows.loc[
            relation_rows["estimability"]
            == "estimable"
        ].copy()

        if (
            full_row.estimability
            != "estimable"
            or pd.isna(full_row.sign)
            or full_row.strength is None
        ):
            sign_fraction = np.nan
            median_change = np.nan
            robust = False
        elif estimable.empty:
            sign_fraction = 0.0
            median_change = np.nan
            robust = False
        else:
            sign_fraction = float(
                (
                    estimable["sign"]
                    == int(full_row.sign)
                ).mean()
            )

            median_change = float(
                (
                    estimable["strength"]
                    - float(
                        full_row.strength
                    )
                )
                .abs()
                .median()
            )

            robust = bool(
                sign_fraction
                >= minimum_sign_fraction
                and median_change
                <= maximum_median_change
            )

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_02",
                "window_id":
                    full_row.window_id,
                "relation_id":
                    full_row.relation_id,
                "source_id":
                    full_row.source_id,
                "target_id":
                    full_row.target_id,
                "loo_estimable_count":
                    int(len(estimable)),
                "loo_sign_preservation_fraction":
                    sign_fraction,
                "loo_median_absolute_strength_change":
                    median_change,
                "leave_one_block_out_robust":
                    robust,
            }
        )

    return pd.DataFrame.from_records(
        records
    ).sort_values(
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


def build_persistence_summary(
    persistence_frame: pd.DataFrame,
    loo_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive persistence-stage counts."""
    merged = persistence_frame.merge(
        loo_frame[
            [
                "window_id",
                "relation_id",
                "leave_one_block_out_robust",
            ]
        ],
        on=[
            "window_id",
            "relation_id",
        ],
        how="left",
        validate="one_to_one",
    )

    by_window: dict[str, Any] = {}

    for window_id, frame in merged.groupby(
        "window_id",
        sort=True,
    ):
        by_window[str(window_id)] = {
            "relation_count":
                int(len(frame)),
            "minimum_valid_blocks_met_count":
                int(
                    frame[
                        "minimum_valid_blocks_met"
                    ].sum()
                ),
            "leave_one_block_out_robust_count":
                int(
                    frame[
                        "leave_one_block_out_robust"
                    ]
                    .fillna(False)
                    .sum()
                ),
            "median_persistence":
                (
                    float(
                        frame["persistence"]
                        .dropna()
                        .median()
                    )
                    if frame[
                        "persistence"
                    ].notna().any()
                    else None
                ),
        }

    return {
        "status":
            "COMPLETED",
        "relation_count":
            int(len(merged)),
        "by_window":
            by_window,
    }


def evaluate_persistence(
    context: ExperimentContext,
) -> ExperimentContext:
    """Execute frozen block-persistence evaluation."""
    validate_context(
        context
    )

    contract = get_persistence_contract(
        context
    )

    feature_config = require_mapping(
        context.configuration.get(
            "feature_table"
        ),
        "feature_table",
    )

    timestamp_column = str(
        feature_config[
            "timestamp_column"
        ]
    ).strip()

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

    full_relations_by_window = require_mapping(
        context.outputs[
            "candidate_relations_by_window"
        ],
        "candidate_relations_by_window",
    )

    block_frames: list[pd.DataFrame] = []
    persistence_frames: list[pd.DataFrame] = []
    loo_frames: list[pd.DataFrame] = []

    for window_id in sorted(
        baseline_tables
    ):
        frame = baseline_tables[
            window_id
        ]

        component_ids = tuple(
            components_by_window[
                window_id
            ]
        )

        full_relations = (
            full_relations_by_window[
                window_id
            ]
        )

        block_relations = (
            estimate_block_relations(
                frame=frame,
                component_ids=component_ids,
                window_id=window_id,
                timestamp_column=(
                    timestamp_column
                ),
                minimum_samples=contract[
                    "minimum_samples_per_block"
                ],
            )
        )

        persistence_frame = (
            summarize_relation_persistence(
                full_relations,
                block_relations,
                minimum_block_strength=contract[
                    "minimum_block_strength"
                ],
                minimum_valid_blocks=contract[
                    "minimum_valid_blocks"
                ],
            )
        )

        loo_frame = (
            evaluate_leave_one_block_out(
                frame=frame,
                component_ids=component_ids,
                full_relations=full_relations,
                window_id=window_id,
                timestamp_column=(
                    timestamp_column
                ),
                minimum_samples=contract[
                    "minimum_samples_per_block"
                ],
                minimum_sign_fraction=contract[
                    "loo_minimum_sign_fraction"
                ],
                maximum_median_change=contract[
                    "loo_maximum_median_change"
                ],
            )
        )

        block_frames.append(
            block_relations
        )

        persistence_frames.append(
            persistence_frame
        )

        loo_frames.append(
            loo_frame
        )

    block_relations_all = pd.concat(
        block_frames,
        ignore_index=True,
    )

    persistence_all = pd.concat(
        persistence_frames,
        ignore_index=True,
    )

    loo_all = pd.concat(
        loo_frames,
        ignore_index=True,
    )

    if len(persistence_all) != 136:
        fail(
            "Persistence summary must contain "
            f"136 relations, observed {len(persistence_all)}."
        )

    if len(loo_all) != 136:
        fail(
            "Leave-one-block-out summary must contain "
            f"136 relations, observed {len(loo_all)}."
        )

    summary = build_persistence_summary(
        persistence_all,
        loo_all,
    )

    context.register_output(
        "block_relations",
        block_relations_all,
    )

    context.register_output(
        "relation_persistence",
        persistence_all,
    )

    context.register_output(
        "leave_one_block_out",
        loo_all,
    )

    context.register_output(
        "persistence_summary",
        summary,
    )

    context.register_runtime(
        "persistence_status",
        "COMPLETED",
    )

    return context
  
