"""
CGIE3-ID-04 relational snapshot construction.

This stage reconstructs temporal relational snapshots from the frozen
74-relation ID-04 population.

For every frozen temporal scale and every admissible temporal position
the module:

- selects the corresponding rolling feature observations;
- re-estimates each frozen relation using Spearman correlation;
- preserves explicit non-estimability;
- derives relation sign only when strength is estimable;
- records one relation estimate per snapshot.

The stage does not:

- select new relations;
- remove frozen relations;
- modify ID-02 classifications;
- modify ID-03 states;
- use earthquake-event information;
- optimize temporal scales;
- infer causality or prediction.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cgie3.src.id04.loader import (
    ID04ExperimentContext,
)


class ID04SnapshotError(ValueError):
    """Raised when relational snapshot construction violates the contract."""


def fail(message: str) -> None:
    """Raise a normalized snapshot-construction error."""
    raise ID04SnapshotError(
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
        raise ID04SnapshotError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
        )

    return normalized


def validate_context(
    context: ID04ExperimentContext,
) -> None:
    """Validate prerequisites for snapshot reconstruction."""
    if not isinstance(
        context,
        ID04ExperimentContext,
    ):
        fail(
            "context must be an ID04ExperimentContext."
        )

    if context.experiment_id != "CGIE3_ID_04":
        fail(
            "Unexpected experiment ID: "
            f"{context.experiment_id}"
        )

    if (
        context.runtime.get(
            "loader_status"
        )
        != "COMPLETED"
    ):
        fail(
            "ID-04 loader must complete before snapshots."
        )

    if len(
        context.primary_population
    ) != 74:
        fail(
            "Snapshot construction requires exactly "
            "74 frozen primary relations."
        )


def parse_window_days(
    window_id: str,
) -> int:
    """
    Convert frozen window IDs such as 1d, 3d, 7d and 30d to days.

    No arbitrary scale interpretation is permitted.
    """
    normalized = str(
        window_id
    ).strip().lower()

    match = re.fullmatch(
        r"([1-9][0-9]*)d",
        normalized,
    )

    if match is None:
        fail(
            "Unsupported frozen temporal scale: "
            f"{window_id}"
        )

    return require_positive_integer(
        match.group(1),
        f"window scale {window_id}",
    )


def get_snapshot_contract(
    context: ID04ExperimentContext,
) -> dict[str, Any]:
    """Extract the frozen snapshot contract."""
    snapshot_definition = require_mapping(
        context.configuration.get(
            "snapshot_definition"
        ),
        "snapshot_definition",
    )

    estimator = require_mapping(
        snapshot_definition.get(
            "estimator"
        ),
        "snapshot_definition.estimator",
    )

    if estimator.get(
        "primary"
    ) != "spearman":
        fail(
            "ID-04 primary snapshot estimator must remain Spearman."
        )

    relation_set = require_mapping(
        snapshot_definition.get(
            "relation_set"
        ),
        "snapshot_definition.relation_set",
    )

    if relation_set.get(
        "recompute_strength_per_snapshot"
    ) is not True:
        fail(
            "Relation strengths must be re-estimated per snapshot."
        )

    non_estimable = require_mapping(
        snapshot_definition.get(
            "non_estimable_relations"
        ),
        "snapshot_definition.non_estimable_relations",
    )

    if non_estimable.get(
        "retain_explicitly"
    ) is not True:
        fail(
            "Non-estimable relations must remain explicit."
        )

    if non_estimable.get(
        "convert_to_zero"
    ) is not False:
        fail(
            "Non-estimable relations must not be converted to zero."
        )

    temporal_scales = tuple(
        str(
            value
        ).strip()
        for value in context.identity.temporal_scales
    )

    if not temporal_scales:
        fail(
            "Identity declaration contains no temporal scales."
        )

    return {
        "temporal_scales":
            temporal_scales,

        "timestamp_column":
            str(
                context.metadata[
                    "timestamp_column"
                ]
            ),

        # Spearman requires at least 3 finite paired observations
        # for a minimally meaningful rank estimate.
        "minimum_complete_observations":
            3,
    }


def prepare_features(
    context: ID04ExperimentContext,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    """Validate and normalize the frozen feature table."""
    features = context.frozen_features.copy()

    if timestamp_column not in features.columns:
        fail(
            "Frozen feature timestamp column is missing: "
            f"{timestamp_column}"
        )

    features[
        timestamp_column
    ] = pd.to_datetime(
        features[
            timestamp_column
        ],
        utc=True,
        errors="coerce",
    )

    if features[
        timestamp_column
    ].isna().any():
        fail(
            "Frozen feature table contains invalid timestamps."
        )

    if "window_id" not in features.columns:
        fail(
        "Frozen feature table is missing required window_id column."
    )

    if features.duplicated(
    subset=[
        timestamp_column,
        "window_id",
    ]
    ).any():
        fail(
        "Frozen feature table contains duplicate timestamp-window pairs."
    )

    return features.sort_values(
    by=[
        timestamp_column,
        "window_id",
    ],
    kind="stable",
    ).reset_index(
    drop=True
    )


def relation_population_by_scale(
    context: ID04ExperimentContext,
    temporal_scales: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """
    Partition the frozen primary relation population by temporal scale.

    Every frozen relation must belong to exactly one declared scale.
    """
    primary = context.primary_population.copy()

    observed_scales = set(
        primary[
            "window_id"
        ].astype(str)
    )

    declared_scales = set(
        temporal_scales
    )

    unknown_scales = sorted(
        observed_scales
        - declared_scales
    )

    if unknown_scales:
        fail(
            "Primary relations contain undeclared temporal scales: "
            + ", ".join(
                unknown_scales
            )
        )

    output: dict[
        str,
        pd.DataFrame,
    ] = {}

    for scale_id in temporal_scales:
        frame = primary.loc[
            primary[
                "window_id"
            ].astype(str)
            == scale_id
        ].copy()

        if frame.empty:
            continue

        output[
            scale_id
        ] = frame.sort_values(
            by=[
                "source_id",
                "target_id",
                "relation_id",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

    if not output:
        fail(
            "No frozen primary relations match declared temporal scales."
        )

    total = sum(
        len(
            frame
        )
        for frame in output.values()
    )

    if total != 74:
        fail(
            "Scale partition must preserve all 74 primary relations; "
            f"observed {total}."
        )

    return output


def complete_pair_values(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
) -> pd.DataFrame:
    """Return finite paired observations for one frozen relation."""
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
            "Frozen feature table is missing relation features: "
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
    ).dropna(
        subset=[
            source_id,
            target_id,
        ]
    )

    return pair.reset_index(
        drop=True
    )


def estimate_relation(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
    *,
    minimum_observations: int,
) -> dict[str, Any]:
    """Estimate one frozen relation inside one temporal snapshot."""
    pair = complete_pair_values(
        frame,
        source_id,
        target_id,
    )

    sample_count = int(
        len(
            pair
        )
    )

    if sample_count < minimum_observations:
        return {
            "estimability":
                "insufficient_observations",

            "strength":
                None,

            "sign":
                None,

            "p_value":
                None,

            "sample_count":
                sample_count,

            "non_estimable_reason":
                "insufficient_complete_observations",
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
                "non_identifiable",

            "strength":
                None,

            "sign":
                None,

            "p_value":
                None,

            "sample_count":
                sample_count,

            "non_estimable_reason":
                "constant_source_feature",
        }

    if np.unique(
        target
    ).size < 2:
        return {
            "estimability":
                "non_identifiable",

            "strength":
                None,

            "sign":
                None,

            "p_value":
                None,

            "sample_count":
                sample_count,

            "non_estimable_reason":
                "constant_target_feature",
        }

    result = spearmanr(
        source,
        target,
    )

    strength = float(
        result.statistic
    )

    p_value = float(
        result.pvalue
    )

    if not np.isfinite(
        strength
    ):
        return {
            "estimability":
                "numerical_failure",

            "strength":
                None,

            "sign":
                None,

            "p_value":
                None,

            "sample_count":
                sample_count,

            "non_estimable_reason":
                "non_finite_spearman_strength",
        }

    if strength > 0.0:
        sign = 1
    elif strength < 0.0:
        sign = -1
    else:
        sign = 0

    return {
        "estimability":
            "estimable",

        "strength":
            strength,

        "sign":
            sign,

        "p_value":
            (
                p_value
                if np.isfinite(
                    p_value
                )
                else None
            ),

        "sample_count":
            sample_count,

        "non_estimable_reason":
            None,
    }


def build_snapshot_positions(
    features: pd.DataFrame,
    *,
    timestamp_column: str,
    window_days: int,
) -> tuple[pd.Timestamp, ...]:
    """
    Define all admissible chronological snapshot endpoints.

    Every observed feature timestamp can become an endpoint once the
    requested trailing window is representable inside the observed data.
    """
    timestamps = tuple(
        pd.Timestamp(
            value
        )
        for value in features[
            timestamp_column
        ]
    )

    if not timestamps:
        fail(
            "Frozen feature table contains no timestamps."
        )

    first_timestamp = timestamps[
        0
    ]

    minimum_endpoint = (
        first_timestamp
        + pd.Timedelta(
            days=window_days
        )
    )

    endpoints = tuple(
        timestamp
        for timestamp in timestamps
        if timestamp >= minimum_endpoint
    )

    if not endpoints:
        fail(
            "No admissible snapshot endpoints exist for "
            f"{window_days}d scale."
        )

    return endpoints


def select_snapshot_window(
    features: pd.DataFrame,
    *,
    timestamp_column: str,
    endpoint: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    """Select one trailing frozen temporal window."""
    start = (
        endpoint
        - pd.Timedelta(
            days=window_days
        )
    )

    return features.loc[
        (
            features[
                timestamp_column
            ]
            > start
        )
        & (
            features[
                timestamp_column
            ]
            <= endpoint
        )
    ].copy()


def build_scale_snapshots(
    features: pd.DataFrame,
    relations: pd.DataFrame,
    *,
    scale_id: str,
    timestamp_column: str,
    minimum_observations: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Re-estimate every frozen scale-specific relation at every endpoint.

    Returns:
    - long relation-estimate table;
    - one-row-per-snapshot metadata table.
    """
    window_days = parse_window_days(
        scale_id
    )

    if "window_id" not in features.columns:
        fail(
            "Frozen feature table is missing required window_id column."
        )

    scale_features = features.loc[
        features["window_id"].astype(str).str.strip()
        == str(scale_id).strip()
    ].copy()

    if scale_features.empty:
        fail(
            f"No frozen feature rows found for temporal scale {scale_id}."
        )

    if scale_features[timestamp_column].duplicated().any():
        fail(
            f"Frozen feature table contains duplicate timestamps "
            f"within temporal scale {scale_id}."
        )

    endpoints = build_snapshot_positions(
        scale_features,
        timestamp_column=timestamp_column,
        window_days=window_days,
    )

    relation_records: list[
        dict[str, Any]
    ] = []

    snapshot_records: list[
        dict[str, Any]
    ] = []

    for snapshot_index, endpoint in enumerate(
        endpoints,
        start=1,
    ):
        snapshot_id = (
            f"CGIE3_ID_04::{scale_id}::"
            f"{endpoint.strftime('%Y%m%dT%H%M%SZ')}"
        )

        snapshot_frame = select_snapshot_window(
            scale_features,
            timestamp_column=timestamp_column,
            endpoint=endpoint,
            window_days=window_days,
        )

        snapshot_start = (
            endpoint
            - pd.Timedelta(
                days=window_days
            )
        )

        estimable_count = 0
        non_estimable_count = 0

        for row in relations.itertuples(
            index=False
        ):
            source_id = str(
                row.source_id
            )

            target_id = str(
                row.target_id
            )

            estimate = estimate_relation(
                snapshot_frame,
                source_id,
                target_id,
                minimum_observations=minimum_observations,
            )

            if (
                estimate[
                    "estimability"
                ]
                == "estimable"
            ):
                estimable_count += 1
            else:
                non_estimable_count += 1

            relation_records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_04",

                    "snapshot_id":
                        snapshot_id,

                    "snapshot_index":
                        int(
                            snapshot_index
                        ),

                    "scale_id":
                        scale_id,

                    "window_days":
                        int(
                            window_days
                        ),

                    "snapshot_start_utc":
                        snapshot_start.isoformat(),

                    "snapshot_end_utc":
                        endpoint.isoformat(),

                    "relation_id":
                        str(
                            row.relation_id
                        ),

                    "source_id":
                        source_id,

                    "target_id":
                        target_id,

                    "id02_status":
                        str(
                            row.classification_status
                        ),

                    "estimator_id":
                        "spearman",

                    "estimability":
                        estimate[
                            "estimability"
                        ],

                    "strength":
                        estimate[
                            "strength"
                        ],

                    "absolute_strength":
                        (
                            abs(
                                float(
                                    estimate[
                                        "strength"
                                    ]
                                )
                            )
                            if estimate[
                                "strength"
                            ]
                            is not None
                            else None
                        ),

                    "sign":
                        estimate[
                            "sign"
                        ],

                    "p_value":
                        estimate[
                            "p_value"
                        ],

                    "sample_count":
                        int(
                            estimate[
                                "sample_count"
                            ]
                        ),

                    "non_estimable_reason":
                        estimate[
                            "non_estimable_reason"
                        ],

                    "relation_selected_post_hoc":
                        False,

                    "id02_status_modified":
                        False,

                    "id03_state_modified":
                        False,
                }
            )

        snapshot_records.append(
            {
                "experiment_id":
                    "CGIE3_ID_04",

                "snapshot_id":
                    snapshot_id,

                "snapshot_index":
                    int(
                        snapshot_index
                    ),

                "scale_id":
                    scale_id,

                "window_days":
                    int(
                        window_days
                    ),

                "snapshot_start_utc":
                    snapshot_start.isoformat(),

                "snapshot_end_utc":
                    endpoint.isoformat(),

                "feature_row_count":
                    int(
                        len(
                            snapshot_frame
                        )
                    ),

                "frozen_relation_count":
                    int(
                        len(
                            relations
                        )
                    ),

                "estimable_relation_count":
                    int(
                        estimable_count
                    ),

                "non_estimable_relation_count":
                    int(
                        non_estimable_count
                    ),

                "estimable_relation_fraction":
                    float(
                        estimable_count
                        / len(
                            relations
                        )
                    ),
            }
        )

    return (
        pd.DataFrame.from_records(
            relation_records
        ),
        pd.DataFrame.from_records(
            snapshot_records
        ),
    )


def validate_snapshot_outputs(
    relations: pd.DataFrame,
    snapshots: pd.DataFrame,
    scale_relations: Mapping[
        str,
        pd.DataFrame,
    ],
) -> None:
    """Validate structural completeness of generated snapshots."""
    if relations.empty:
        fail(
            "Snapshot relation table is empty."
        )

    if snapshots.empty:
        fail(
            "Snapshot metadata table is empty."
        )

    duplicate_relation_keys = relations.duplicated(
        subset=[
            "snapshot_id",
            "relation_id",
        ],
        keep=False,
    )

    if duplicate_relation_keys.any():
        fail(
            "Snapshot relation table contains duplicate "
            "snapshot-relation keys."
        )

    duplicate_snapshot_ids = snapshots[
        "snapshot_id"
    ].duplicated(
        keep=False
    )

    if duplicate_snapshot_ids.any():
        fail(
            "Snapshot metadata contains duplicate snapshot IDs."
        )

    for scale_id, relation_frame in scale_relations.items():
        expected_relation_count = int(
            len(
                relation_frame
            )
        )

        observed = (
            relations.loc[
                relations[
                    "scale_id"
                ]
                == scale_id
            ]
            .groupby(
                "snapshot_id",
                sort=False,
            )
            .size()
        )

        if observed.empty:
            fail(
                f"No snapshot relations produced for {scale_id}."
            )

        if (
            observed
            != expected_relation_count
        ).any():
            fail(
                "Snapshot construction failed to preserve "
                f"all frozen {scale_id} relations."
            )

    if relations[
        "relation_selected_post_hoc"
    ].any():
        fail(
            "Post-hoc relation selection detected."
        )

    if relations[
        "id02_status_modified"
    ].any():
        fail(
            "ID-02 status mutation detected."
        )

    if relations[
        "id03_state_modified"
    ].any():
        fail(
            "ID-03 state mutation detected."
        )


def build_snapshot_summary(
    relations: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive snapshot-stage summary."""
    by_scale: dict[
        str,
        Any,
    ] = {}

    for scale_id, frame in snapshots.groupby(
        "scale_id",
        sort=True,
    ):
        by_scale[
            str(
                scale_id
            )
        ] = {
            "snapshot_count":
                int(
                    len(
                        frame
                    )
                ),

            "minimum_feature_rows":
                int(
                    frame[
                        "feature_row_count"
                    ].min()
                ),

            "maximum_feature_rows":
                int(
                    frame[
                        "feature_row_count"
                    ].max()
                ),

            "mean_estimable_relation_fraction":
                float(
                    frame[
                        "estimable_relation_fraction"
                    ].mean()
                ),
        }

    estimability_counts = (
        relations[
            "estimability"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "status":
            "COMPLETED",

        "snapshot_count":
            int(
                len(
                    snapshots
                )
            ),

        "snapshot_relation_row_count":
            int(
                len(
                    relations
                )
            ),

        "scale_count":
            int(
                snapshots[
                    "scale_id"
                ].nunique()
            ),

        "estimability_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in estimability_counts.items()
        },

        "by_scale":
            by_scale,

        "relation_selection_modified":
            False,

        "id02_statuses_modified":
            False,

        "id03_states_modified":
            False,

        "earthquake_event_information_used":
            False,
    }


def build_snapshots(
    context: ID04ExperimentContext,
) -> ID04ExperimentContext:
    """Execute frozen CGIE3-ID-04 relational snapshot construction."""
    validate_context(
        context
    )

    contract = get_snapshot_contract(
        context
    )

    features = prepare_features(
        context,
        timestamp_column=contract[
            "timestamp_column"
        ],
    )

    scale_relations = relation_population_by_scale(
        context,
        contract[
            "temporal_scales"
        ],
    )

    relation_frames: list[
        pd.DataFrame
    ] = []

    snapshot_frames: list[
        pd.DataFrame
    ] = []

    for scale_id in contract[
        "temporal_scales"
    ]:
        if scale_id not in scale_relations:
            continue

        (
            relations,
            snapshots,
        ) = build_scale_snapshots(
            features,
            scale_relations[
                scale_id
            ],
            scale_id=scale_id,
            timestamp_column=contract[
                "timestamp_column"
            ],
            minimum_observations=contract[
                "minimum_complete_observations"
            ],
        )

        relation_frames.append(
            relations
        )

        snapshot_frames.append(
            snapshots
        )

    all_relations = pd.concat(
        relation_frames,
        ignore_index=True,
    )

    all_snapshots = pd.concat(
        snapshot_frames,
        ignore_index=True,
    )

    all_relations = all_relations.sort_values(
        by=[
            "scale_id",
            "snapshot_index",
            "source_id",
            "target_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    all_snapshots = all_snapshots.sort_values(
        by=[
            "scale_id",
            "snapshot_index",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    validate_snapshot_outputs(
        all_relations,
        all_snapshots,
        scale_relations,
    )

    summary = build_snapshot_summary(
        all_relations,
        all_snapshots,
    )

    context.register_output(
        "snapshot_relations",
        all_relations,
    )

    context.register_output(
        "snapshots",
        all_snapshots,
    )

    context.register_output(
        "snapshot_summary",
        summary,
    )

    context.register_runtime(
        "snapshot_status",
        "COMPLETED",
    )

    return context
