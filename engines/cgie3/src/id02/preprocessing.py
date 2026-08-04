"""
CGIE3-ID-02 preprocessing stage.

This module validates and prepares the frozen baseline feature table
for relation discovery.

It does not:

- read files directly;
- estimate relations;
- select eligible relations;
- calculate Congruity metrics;
- use evaluation-period or target-event information.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from congruity.core import ExperimentContext


class PreprocessingError(ValueError):
    """Raised when feature preprocessing violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a preprocessing error."""
    raise PreprocessingError(
        str(message).strip()
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like configuration value."""
    if not isinstance(value, Mapping):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def require_string(
    value: Any,
    field_name: str,
) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str):
        fail(
            f"{field_name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        fail(
            f"{field_name} must not be empty."
        )

    return normalized


def require_string_list(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    """Require a list of unique non-empty strings."""
    if not isinstance(
        value,
        (list, tuple),
    ):
        fail(
            f"{field_name} must be a list."
        )

    normalized = tuple(
        require_string(
            item,
            f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )

    if not normalized:
        fail(
            f"{field_name} must not be empty."
        )

    if len(normalized) != len(
        set(normalized)
    ):
        fail(
            f"{field_name} contains duplicates."
        )

    return normalized


def parse_utc(
    value: Any,
    field_name: str,
) -> pd.Timestamp:
    """Parse a configuration timestamp as UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise PreprocessingError(
            f"{field_name} is not a valid timestamp: "
            f"{value}"
        ) from exc

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp


def validate_context(
    context: ExperimentContext,
) -> None:
    """Validate the preprocessing input contract."""
    if not isinstance(
        context,
        ExperimentContext,
    ):
        fail(
            "context must be an ExperimentContext."
        )

    if context.experiment_id != "CGIE3_ID_02":
        fail(
            "Unexpected experiment ID. "
            f"Observed: {context.experiment_id}"
        )

    if not isinstance(
        context.feature_table,
        pd.DataFrame,
    ):
        fail(
            "context.feature_table must be "
            "a pandas DataFrame."
        )

    if context.feature_table.empty:
        fail(
            "context.feature_table is empty."
        )


def get_feature_contract(
    context: ExperimentContext,
) -> dict[str, Any]:
    """Extract the frozen feature-table contract."""
    feature_config = require_mapping(
        context.configuration.get(
            "feature_table"
        ),
        "feature_table",
    )

    timestamp_column = require_string(
        feature_config.get(
            "timestamp_column"
        ),
        "feature_table.timestamp_column",
    )

    window_column = require_string(
        feature_config.get(
            "window_column"
        ),
        "feature_table.window_column",
    )

    required_windows = require_string_list(
        feature_config.get(
            "required_windows"
        ),
        "feature_table.required_windows",
    )

    required_components = (
        require_string_list(
            feature_config.get(
                "required_components"
            ),
            "feature_table.required_components",
        )
    )

    excluded_columns_raw = (
        feature_config.get(
            "excluded_columns",
            [],
        )
    )

    if not isinstance(
        excluded_columns_raw,
        (list, tuple),
    ):
        fail(
            "feature_table.excluded_columns "
            "must be a list."
        )

    excluded_columns = tuple(
        str(value).strip()
        for value in excluded_columns_raw
        if str(value).strip()
    )

    exclusions_by_window_raw = (
        feature_config.get(
            "window_specific_exclusions",
            {},
        )
    )

    exclusions_by_window = (
        require_mapping(
            exclusions_by_window_raw,
            "feature_table."
            "window_specific_exclusions",
        )
    )

    normalized_exclusions: dict[
        str,
        tuple[str, ...],
    ] = {}

    for window_id in required_windows:
        raw_values = exclusions_by_window.get(
            window_id,
            [],
        )

        if not isinstance(
            raw_values,
            (list, tuple),
        ):
            fail(
                "Window-specific exclusions for "
                f"{window_id} must be a list."
            )

        normalized_exclusions[
            window_id
        ] = tuple(
            str(value).strip()
            for value in raw_values
            if str(value).strip()
        )

    return {
        "timestamp_column":
            timestamp_column,
        "window_column":
            window_column,
        "required_windows":
            required_windows,
        "required_components":
            required_components,
        "excluded_columns":
            excluded_columns,
        "window_specific_exclusions":
            normalized_exclusions,
    }


def select_baseline_rows(
    frame: pd.DataFrame,
    context: ExperimentContext,
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
        analysis_period.get("baseline"),
        "analysis_period.baseline",
    )

    baseline_start = parse_utc(
        baseline.get("start_utc"),
        "analysis_period.baseline.start_utc",
    )

    baseline_end = parse_utc(
        baseline.get("end_utc"),
        "analysis_period.baseline.end_utc",
    )

    if baseline_start >= baseline_end:
        fail(
            "Baseline start must precede "
            "baseline end."
        )

    use_evaluation = analysis_period.get(
        "use_evaluation_for_relation_selection"
    )

    if use_evaluation is not False:
        fail(
            "Evaluation data must not be used "
            "for relation selection."
        )

    use_target = analysis_period.get(
        "use_target_event_for_relation_selection"
    )

    if use_target is not False:
        fail(
            "Target-event information must not "
            "be used for relation selection."
        )

    baseline_frame = frame.loc[
        (
            frame[timestamp_column]
            >= baseline_start
        )
        & (
            frame[timestamp_column]
            <= baseline_end
        )
    ].copy()

    if baseline_frame.empty:
        fail(
            "No feature rows fall inside "
            "the frozen baseline interval."
        )

    return baseline_frame


def prepare_window_frame(
    baseline_frame: pd.DataFrame,
    *,
    window_id: str,
    timestamp_column: str,
    window_column: str,
    required_components: tuple[str, ...],
    excluded_components: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Prepare one frozen temporal-window feature table."""
    frame = baseline_frame.loc[
        baseline_frame[window_column]
        == window_id
    ].copy()

    if frame.empty:
        fail(
            f"No baseline rows exist for window "
            f"{window_id}."
        )

    active_components = tuple(
        component
        for component in required_components
        if component
        not in set(excluded_components)
    )

    if len(active_components) < 2:
        fail(
            f"Window {window_id} has fewer than "
            "two active components."
        )

    missing_components = sorted(
        set(active_components)
        - set(frame.columns)
    )

    if missing_components:
        fail(
            f"Window {window_id} is missing "
            "active components: "
            + ", ".join(missing_components)
        )

    output_columns = (
        timestamp_column,
        window_column,
        *active_components,
    )

    frame = frame.loc[
        :,
        output_columns,
    ].copy()

    for component in active_components:
        frame[component] = pd.to_numeric(
            frame[component],
            errors="coerce",
        )

        frame[component] = frame[
            component
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    frame = frame.sort_values(
        by=timestamp_column,
        kind="stable",
    ).reset_index(drop=True)

    if not frame[
        timestamp_column
    ].is_monotonic_increasing:
        fail(
            f"Timestamps are not monotonic "
            f"for window {window_id}."
        )

    duplicate_count = int(
        frame.duplicated(
            subset=[
                timestamp_column,
                window_column,
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        fail(
            f"Window {window_id} contains "
            f"{duplicate_count} duplicate-key rows."
        )

    return frame, active_components


def validate_expected_counts(
    prepared_windows: Mapping[
        str,
        pd.DataFrame,
    ],
    components_by_window: Mapping[
        str,
        tuple[str, ...],
    ],
    context: ExperimentContext,
) -> None:
    """Validate frozen component and candidate-pair counts."""
    window_rules = require_mapping(
        context.configuration.get(
            "window_rules"
        ),
        "window_rules",
    )

    expected_components = (
        require_mapping(
            window_rules.get(
                "expected_component_counts"
            ),
            "window_rules."
            "expected_component_counts",
        )
    )

    expected_relations = (
        require_mapping(
            window_rules.get(
                "expected_candidate_relation_counts"
            ),
            "window_rules."
            "expected_candidate_relation_counts",
        )
    )

    total_relations = 0

    for window_id in prepared_windows:
        component_count = len(
            components_by_window[
                window_id
            ]
        )

        expected_component_count = int(
            expected_components[
                window_id
            ]
        )

        if (
            component_count
            != expected_component_count
        ):
            fail(
                f"Window {window_id} has "
                f"{component_count} active components; "
                f"expected {expected_component_count}."
            )

        relation_count = (
            component_count
            * (component_count - 1)
            // 2
        )

        expected_relation_count = int(
            expected_relations[
                window_id
            ]
        )

        if (
            relation_count
            != expected_relation_count
        ):
            fail(
                f"Window {window_id} produces "
                f"{relation_count} candidate relations; "
                f"expected {expected_relation_count}."
            )

        total_relations += relation_count

    expected_total = int(
        window_rules[
            "expected_total_candidate_relations"
        ]
    )

    if total_relations != expected_total:
        fail(
            "Total candidate relation count is "
            f"{total_relations}; expected "
            f"{expected_total}."
        )


def build_quality_audit(
    prepared_windows: Mapping[
        str,
        pd.DataFrame,
    ],
    components_by_window: Mapping[
        str,
        tuple[str, ...],
    ],
    *,
    timestamp_column: str,
) -> dict[str, Any]:
    """Build descriptive data-quality information."""
    windows: dict[str, Any] = {}

    for window_id, frame in (
        prepared_windows.items()
    ):
        components = components_by_window[
            window_id
        ]

        missing_fraction = {
            component: float(
                frame[component]
                .isna()
                .mean()
            )
            for component in components
        }

        unique_counts = {
            component: int(
                frame[component]
                .nunique(
                    dropna=True
                )
            )
            for component in components
        }

        windows[window_id] = {
            "row_count":
                int(len(frame)),
            "component_count":
                int(len(components)),
            "candidate_relation_count":
                int(
                    len(components)
                    * (
                        len(components) - 1
                    )
                    // 2
                ),
            "start_utc":
                frame[timestamp_column]
                .min()
                .isoformat(),
            "end_utc":
                frame[timestamp_column]
                .max()
                .isoformat(),
            "missing_fraction":
                missing_fraction,
            "unique_value_count":
                unique_counts,
        }

    return {
        "status": "VALID",
        "windows": windows,
    }


def preprocess(
    context: ExperimentContext,
) -> ExperimentContext:
    """Execute frozen CGIE3-ID-02 preprocessing."""
    validate_context(context)

    contract = get_feature_contract(
        context
    )

    timestamp_column = contract[
        "timestamp_column"
    ]

    window_column = contract[
        "window_column"
    ]

    feature_table = (
        context.feature_table.copy()
    )

    baseline_frame = select_baseline_rows(
        feature_table,
        context,
        timestamp_column=timestamp_column,
    )

    prepared_windows: dict[
        str,
        pd.DataFrame,
    ] = {}

    components_by_window: dict[
        str,
        tuple[str, ...],
    ] = {}

    for window_id in contract[
        "required_windows"
    ]:
        (
            prepared_frame,
            active_components,
        ) = prepare_window_frame(
            baseline_frame,
            window_id=window_id,
            timestamp_column=(
                timestamp_column
            ),
            window_column=window_column,
            required_components=contract[
                "required_components"
            ],
            excluded_components=contract[
                "window_specific_exclusions"
            ][window_id],
        )

        prepared_windows[
            window_id
        ] = prepared_frame

        components_by_window[
            window_id
        ] = active_components

    validate_expected_counts(
        prepared_windows,
        components_by_window,
        context,
    )

    quality_audit = build_quality_audit(
        prepared_windows,
        components_by_window,
        timestamp_column=timestamp_column,
    )

    context.register_output(
        "baseline_feature_tables",
        prepared_windows,
    )

    context.register_output(
        "components_by_window",
        components_by_window,
    )

    context.register_output(
        "preprocessing_quality_audit",
        quality_audit,
    )

    context.register_metadata(
        "baseline_row_count",
        int(len(baseline_frame)),
    )

    context.register_runtime(
        "preprocessing_status",
        "COMPLETED",
    )

    return context
