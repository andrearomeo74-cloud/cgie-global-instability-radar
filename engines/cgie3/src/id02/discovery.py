"""
CGIE3-ID-02 relation-discovery stage.

This module applies the domain-independent Relation Core to the
preprocessed baseline feature tables stored in ExperimentContext.

It does not:

- read files directly;
- select eligible relations;
- use evaluation-period data;
- use target-event labels;
- infer causality or prediction.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from congruity.core import ExperimentContext
from congruity.relations import (
    RelationDiscoveryError,
    discover_spearman_relations,
    relation_estimates_to_frame,
)


class DiscoveryStageError(ValueError):
    """Raised when relation discovery violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a discovery-stage error."""
    raise DiscoveryStageError(
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


def require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """Require an integer greater than zero."""
    if isinstance(value, bool):
        fail(
            f"{field_name} must be an integer."
        )

    try:
        normalized = int(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise DiscoveryStageError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
        )

    return normalized


def validate_context(
    context: ExperimentContext,
) -> None:
    """Validate discovery-stage prerequisites."""
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
        "preprocessing_quality_audit",
    }

    missing_outputs = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing_outputs:
        fail(
            "Preprocessing outputs are missing: "
            + ", ".join(missing_outputs)
        )


def get_discovery_contract(
    context: ExperimentContext,
) -> dict[str, Any]:
    """Extract the frozen relation-discovery contract."""
    candidate_generation = require_mapping(
        context.configuration.get(
            "candidate_generation"
        ),
        "candidate_generation",
    )

    estimability = require_mapping(
        context.configuration.get(
            "estimability"
        ),
        "estimability",
    )

    estimator = str(
        candidate_generation.get(
            "estimator",
            ""
        )
    ).strip()

    if estimator != "spearman":
        fail(
            "CGIE3-ID-02 currently requires "
            "the frozen Spearman estimator."
        )

    if (
        candidate_generation.get(
            "generate_all_unique_pairs"
        )
        is not True
    ):
        fail(
            "All unique component pairs must be generated."
        )

    if (
        candidate_generation.get(
            "allow_self_relations"
        )
        is not False
    ):
        fail(
            "Self-relations must remain disabled."
        )

    minimum_samples = (
        require_positive_integer(
            estimability.get(
                "minimum_paired_observations"
            ),
            "estimability."
            "minimum_paired_observations",
        )
    )

    return {
        "estimator":
            estimator,
        "minimum_samples":
            minimum_samples,
        "zero_tolerance":
            1e-12,
    }


def discover_window_relations(
    frame: pd.DataFrame,
    component_ids: tuple[str, ...],
    *,
    window_id: str,
    timestamp_column: str,
    minimum_samples: int,
    zero_tolerance: float,
) -> pd.DataFrame:
    """Discover all candidate relations for one temporal window."""
    if frame.empty:
        fail(
            f"Baseline frame is empty for window {window_id}."
        )

    if timestamp_column not in frame.columns:
        fail(
            f"Timestamp column {timestamp_column} "
            f"is absent for window {window_id}."
        )

    timestamp_utc = (
        frame[timestamp_column]
        .max()
        .isoformat()
    )

    try:
        estimates = discover_spearman_relations(
            frame=frame,
            component_ids=component_ids,
            window_id=window_id,
            timestamp_utc=timestamp_utc,
            minimum_samples=minimum_samples,
            zero_tolerance=zero_tolerance,
            metadata={
                "experiment_id":
                    "CGIE3_ID_02",
                "selection_period":
                    "frozen_baseline",
                "uses_evaluation_data":
                    False,
                "uses_target_event":
                    False,
            },
        )
    except RelationDiscoveryError as exc:
        raise DiscoveryStageError(
            f"Relation discovery failed for "
            f"window {window_id}: {exc}"
        ) from exc

    relation_frame = relation_estimates_to_frame(
        estimates
    )

    expected_count = (
        len(component_ids)
        * (
            len(component_ids) - 1
        )
        // 2
    )

    if len(relation_frame) != expected_count:
        fail(
            f"Window {window_id} produced "
            f"{len(relation_frame)} relations; "
            f"expected {expected_count}."
        )

    relation_frame.insert(
        0,
        "experiment_id",
        "CGIE3_ID_02",
    )

    return relation_frame


def validate_total_relation_count(
    relation_frames: Mapping[
        str,
        pd.DataFrame,
    ],
    context: ExperimentContext,
) -> None:
    """Validate frozen per-window and total relation counts."""
    window_rules = require_mapping(
        context.configuration.get(
            "window_rules"
        ),
        "window_rules",
    )

    expected_by_window = require_mapping(
        window_rules.get(
            "expected_candidate_relation_counts"
        ),
        "window_rules."
        "expected_candidate_relation_counts",
    )

    observed_total = 0

    for window_id, frame in relation_frames.items():
        expected = int(
            expected_by_window[
                window_id
            ]
        )

        observed = int(
            len(frame)
        )

        if observed != expected:
            fail(
                f"Window {window_id} contains "
                f"{observed} relations; expected {expected}."
            )

        observed_total += observed

    expected_total = int(
        window_rules[
            "expected_total_candidate_relations"
        ]
    )

    if observed_total != expected_total:
        fail(
            "Total candidate relation count is "
            f"{observed_total}; expected {expected_total}."
        )


def build_discovery_summary(
    relation_frames: Mapping[
        str,
        pd.DataFrame,
    ],
) -> dict[str, Any]:
    """Build descriptive discovery-stage counts."""
    by_window: dict[str, Any] = {}

    total_estimable = 0
    total_non_estimable = 0

    for window_id, frame in relation_frames.items():
        estimable_count = int(
            (
                frame["estimability"]
                == "estimable"
            ).sum()
        )

        non_estimable_count = int(
            len(frame)
            - estimable_count
        )

        total_estimable += estimable_count
        total_non_estimable += non_estimable_count

        reasons = (
            frame.loc[
                frame["estimability"]
                != "estimable",
                "non_estimable_reason",
            ]
            .fillna("unspecified")
            .value_counts()
            .sort_index()
            .to_dict()
        )

        by_window[window_id] = {
            "candidate_relation_count":
                int(len(frame)),
            "estimable_count":
                estimable_count,
            "non_estimable_count":
                non_estimable_count,
            "non_estimable_reasons":
                {
                    str(key): int(value)
                    for key, value
                    in reasons.items()
                },
        }

    return {
        "status":
            "COMPLETED",
        "total_candidate_relations":
            int(
                total_estimable
                + total_non_estimable
            ),
        "total_estimable":
            int(total_estimable),
        "total_non_estimable":
            int(total_non_estimable),
        "by_window":
            by_window,
    }


def discover(
    context: ExperimentContext,
) -> ExperimentContext:
    """Execute frozen CGIE3-ID-02 candidate relation discovery."""
    validate_context(
        context
    )

    contract = get_discovery_contract(
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

    relation_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

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

        relation_frames[
            window_id
        ] = discover_window_relations(
            frame=frame,
            component_ids=component_ids,
            window_id=window_id,
            timestamp_column=(
                timestamp_column
            ),
            minimum_samples=contract[
                "minimum_samples"
            ],
            zero_tolerance=contract[
                "zero_tolerance"
            ],
        )

    validate_total_relation_count(
        relation_frames,
        context,
    )

    combined_frame = pd.concat(
        [
            relation_frames[
                window_id
            ]
            for window_id in sorted(
                relation_frames
            )
        ],
        ignore_index=True,
    )

    combined_frame = combined_frame.sort_values(
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

    summary = build_discovery_summary(
        relation_frames
    )

    context.register_output(
        "candidate_relations_by_window",
        relation_frames,
    )

    context.register_output(
        "candidate_relations",
        combined_frame,
    )

    context.register_output(
        "relation_discovery_summary",
        summary,
    )

    context.register_runtime(
        "relation_discovery_status",
        "COMPLETED",
    )

    return context
