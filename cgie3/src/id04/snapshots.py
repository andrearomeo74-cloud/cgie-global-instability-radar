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

Implementation note
-------------------
The frozen scientific contract is unchanged. The execution path is
optimized by coercing feature columns once per temporal scale and by
operating on NumPy slices rather than rebuilding pandas frames for every
relation at every snapshot.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as student_t

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

    features[
        "window_id"
    ] = (
        features[
            "window_id"
        ]
        .astype(str)
        .str.strip()
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
            "window_id",
            timestamp_column,
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


def _spearman_from_arrays(
    source: np.ndarray,
    target: np.ndarray,
    *,
    minimum_observations: int,
) -> dict[str, Any]:
    """
    Compute Spearman from already-coerced NumPy vectors.

    This is mathematically equivalent to ranking the two complete vectors
    and computing their Pearson correlation. P-values use the same
    t-transformation used by scipy.stats.spearmanr for the two-variable case.
    """
    complete = (
        np.isfinite(
            source
        )
        & np.isfinite(
            target
        )
    )

    x = source[
        complete
    ]

    y = target[
        complete
    ]

    sample_count = int(
        x.size
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

    if np.unique(
        x
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
        y
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

    ranked_x = rankdata(
        x,
        method="average",
    ).astype(
        float,
        copy=False,
    )

    ranked_y = rankdata(
        y,
        method="average",
    ).astype(
        float,
        copy=False,
    )

    ranked_x = (
        ranked_x
        - ranked_x.mean()
    )

    ranked_y = (
        ranked_y
        - ranked_y.mean()
    )

    denominator = float(
        np.sqrt(
            np.dot(
                ranked_x,
                ranked_x,
            )
            * np.dot(
                ranked_y,
                ranked_y,
            )
        )
    )

    if (
        denominator <= 0.0
        or not np.isfinite(
            denominator
        )
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

    strength = float(
        np.dot(
            ranked_x,
            ranked_y,
        )
        / denominator
    )

    # Protect against tiny floating-point excursions beyond [-1, 1].
    strength = float(
        np.clip(
            strength,
            -1.0,
            1.0,
        )
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

    degrees_of_freedom = (
        sample_count
        - 2
    )

    if abs(
        strength
    ) >= 1.0:
        p_value = 0.0
    else:
        denominator_term = (
            (1.0 + strength)
            * (1.0 - strength)
        )

        if denominator_term <= 0.0:
            p_value = 0.0
        else:
            t_statistic = (
                strength
                * np.sqrt(
                    degrees_of_freedom
                    / denominator_term
                )
            )

            p_value = float(
                2.0
                * student_t.sf(
                    abs(
                        t_statistic
                    ),
                    degrees_of_freedom,
                )
            )

            if not np.isfinite(
                p_value
            ):
                p_value = None

    return {
        "estimability":
            "estimable",

        "strength":
            strength,

        "sign":
            sign,

        "p_value":
            p_value,

        "sample_count":
            sample_count,

        "non_estimable_reason":
            None,
    }


def estimate_relation(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
    *,
    minimum_observations: int,
) -> dict[str, Any]:
    """
    Estimate one frozen relation inside one temporal snapshot.

    Retained as a public stage helper. The main execution path uses the
    equivalent NumPy implementation directly to avoid repeated dataframe
    conversion overhead.
    """
    pair = complete_pair_values(
        frame,
        source_id,
        target_id,
    )

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

    return _spearman_from_arrays(
        source,
        target,
        minimum_observations=minimum_observations,
    )


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

    The scientific operation is unchanged from the original implementation.
    The execution is accelerated by pre-coercing relation feature columns
    once and slicing NumPy arrays by integer positions.
    """
    window_days = parse_window_days(
        scale_id
    )

    if "window_id" not in features.columns:
        fail(
            "Frozen feature table is missing required window_id column."
        )

    scale_features = features.loc[
        (
            features[
                "window_id"
            ]
            .astype(str)
            .str.strip()
        )
        == str(
            scale_id
        ).strip()
    ].copy()

    if scale_features.empty:
        fail(
            f"No frozen feature rows found for temporal scale {scale_id}."
        )

    scale_features = scale_features.sort_values(
        by=timestamp_column,
        kind="stable",
    ).reset_index(
        drop=True
    )

    if scale_features[
        timestamp_column
    ].duplicated().any():
        fail(
            f"Frozen feature table contains duplicate timestamps "
            f"within temporal scale {scale_id}."
        )

    required_feature_columns = sorted(
        set(
            relations[
                "source_id"
            ].astype(str)
        )
        | set(
            relations[
                "target_id"
            ].astype(str)
        )
    )

    missing_feature_columns = sorted(
        set(
            required_feature_columns
        )
        - set(
            scale_features.columns
        )
    )

    if missing_feature_columns:
        fail(
            "Frozen feature table is missing relation features: "
            + ", ".join(
                missing_feature_columns
            )
        )

    feature_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for column in required_feature_columns:
        values = pd.to_numeric(
            scale_features[
                column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            copy=True,
        )

        values[
            ~np.isfinite(
                values
            )
        ] = np.nan

        feature_arrays[
            column
        ] = values

    relation_specs: list[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = []

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

    Optimized implementation:
    - preserves the frozen relation population;
    - preserves every admissible endpoint;
    - preserves Spearman estimation;
    - preserves explicit non-estimability;
    - avoids repeated DataFrame filtering inside every relation/snapshot pair;
    - avoids post-hoc relation selection.

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
        features[
            "window_id"
        ].astype(str).str.strip()
        == str(
            scale_id
        ).strip()
    ].copy()

    if scale_features.empty:
        fail(
            f"No frozen feature rows found for temporal scale {scale_id}."
        )

    scale_features = scale_features.sort_values(
        by=timestamp_column,
        kind="stable",
    ).reset_index(
        drop=True
    )

    if scale_features[
        timestamp_column
    ].duplicated().any():
        fail(
            "Frozen feature table contains duplicate timestamps "
            f"within temporal scale {scale_id}."
        )

    required_feature_columns = (
        set(
            relations[
                "source_id"
            ].astype(str)
        )
        |
        set(
            relations[
                "target_id"
            ].astype(str)
        )
    )

    missing_feature_columns = sorted(
        required_feature_columns
        - set(
            scale_features.columns
        )
    )

    if missing_feature_columns:
        fail(
            "Frozen feature table is missing relation features: "
            + ", ".join(
                missing_feature_columns
            )
        )

    timestamps = pd.DatetimeIndex(
        scale_features[
            timestamp_column
        ]
    )

    if len(
        timestamps
    ) == 0:
        fail(
            "Frozen feature table contains no timestamps."
        )

    minimum_endpoint = (
        timestamps[
            0
        ]
        + pd.Timedelta(
            days=window_days
        )
    )

    first_endpoint_index = int(
        timestamps.searchsorted(
            minimum_endpoint,
            side="left",
        )
    )

    if first_endpoint_index >= len(
        timestamps
    ):
        fail(
            "No admissible snapshot endpoints exist for "
            f"{window_days}d scale."
        )

    endpoint_indices = np.arange(
        first_endpoint_index,
        len(
            timestamps
        ),
        dtype=np.int64,
    )

    feature_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for column in required_feature_columns:
        values = pd.to_numeric(
            scale_features[
                column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            copy=True,
        )

        values[
            ~np.isfinite(
                values
            )
        ] = np.nan

        feature_arrays[
            column
        ] = values

    relation_specs: list[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = []

    for row in relations.itertuples(
        index=False
    ):
        relation_specs.append(
            (
                str(
                    row.relation_id
                ),
                str(
                    row.source_id
                ),
                str(
                    row.target_id
                ),
                str(
                    row.classification_status
                ),
            )
        )

    relation_records: list[
        dict[str, Any]
    ] = []

    snapshot_records: list[
        dict[str, Any]
    ] = []

    total_endpoints = int(
        len(
            endpoint_indices
        )
    )

    relation_count = int(
        len(
            relation_specs
        )
    )

    print(
        f"[ID04] START scale={scale_id} "
        f"endpoints={total_endpoints} "
        f"relations={relation_count}",
        flush=True,
    )

    window_delta = pd.Timedelta(
        days=window_days
    )

    for snapshot_index, endpoint_position in enumerate(
        endpoint_indices,
        start=1,
    ):
        endpoint = timestamps[
            endpoint_position
        ]

        start_time = (
            endpoint
            - window_delta
        )

        start_position = int(
            timestamps.searchsorted(
                start_time,
                side="right",
            )
        )

        stop_position = int(
            endpoint_position
            + 1
        )

        feature_row_count = int(
            stop_position
            - start_position
        )

        if (
            snapshot_index == 1
            or snapshot_index % 500 == 0
            or snapshot_index == total_endpoints
        ):
            print(
                f"[ID04] scale={scale_id} "
                f"snapshot={snapshot_index}/{total_endpoints} "
                f"endpoint={endpoint.isoformat()}",
                flush=True,
            )

        snapshot_id = (
            f"CGIE3_ID_04::{scale_id}::"
            f"{endpoint.strftime('%Y%m%dT%H%M%SZ')}"
        )

        estimable_count = 0
        non_estimable_count = 0

        for (
            relation_id,
            source_id,
            target_id,
            classification_status,
        ) in relation_specs:

            source = feature_arrays[
                source_id
            ][
                start_position:
                stop_position
            ]

            target = feature_arrays[
                target_id
            ][
                start_position:
                stop_position
            ]

            finite_mask = (
                np.isfinite(
                    source
                )
                &
                np.isfinite(
                    target
                )
            )

            pair_source = source[
                finite_mask
            ]

            pair_target = target[
                finite_mask
            ]

            sample_count = int(
                pair_source.size
            )

            estimability: str
            strength: float | None
            sign: int | None
            p_value: float | None
            non_estimable_reason: str | None

            if sample_count < minimum_observations:
                estimability = "insufficient_observations"
                strength = None
                sign = None
                p_value = None
                non_estimable_reason = (
                    "insufficient_complete_observations"
                )

            elif np.unique(
                pair_source
            ).size < 2:
                estimability = "non_identifiable"
                strength = None
                sign = None
                p_value = None
                non_estimable_reason = (
                    "constant_source_feature"
                )

            elif np.unique(
                pair_target
            ).size < 2:
                estimability = "non_identifiable"
                strength = None
                sign = None
                p_value = None
                non_estimable_reason = (
                    "constant_target_feature"
                )

            else:
                result = spearmanr(
                    pair_source,
                    pair_target,
                )

                candidate_strength = float(
                    result.statistic
                )

                candidate_p_value = float(
                    result.pvalue
                )

                if not np.isfinite(
                    candidate_strength
                ):
                    estimability = "numerical_failure"
                    strength = None
                    sign = None
                    p_value = None
                    non_estimable_reason = (
                        "non_finite_spearman_strength"
                    )

                else:
                    estimability = "estimable"
                    strength = candidate_strength

                    if strength > 0.0:
                        sign = 1
                    elif strength < 0.0:
                        sign = -1
                    else:
                        sign = 0

                    p_value = (
                        candidate_p_value
                        if np.isfinite(
                            candidate_p_value
                        )
                        else None
                    )

                    non_estimable_reason = None

            if estimability == "estimable":
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
                        start_time.isoformat(),

                    "snapshot_end_utc":
                        endpoint.isoformat(),

                    "relation_id":
                        relation_id,

                    "source_id":
                        source_id,

                    "target_id":
                        target_id,

                    "id02_status":
                        classification_status,

                    "estimator_id":
                        "spearman",

                    "estimability":
                        estimability,

                    "strength":
                        strength,

                    "absolute_strength":
                        (
                            abs(
                                float(
                                    strength
                                )
                            )
                            if strength is not None
                            else None
                        ),

                    "sign":
                        sign,

                    "p_value":
                        p_value,

                    "sample_count":
                        sample_count,

                    "non_estimable_reason":
                        non_estimable_reason,

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
                    start_time.isoformat(),

                "snapshot_end_utc":
                    endpoint.isoformat(),

                "feature_row_count":
                    feature_row_count,

                "frozen_relation_count":
                    relation_count,

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
                        / relation_count
                    ),
            }
        )

    print(
        f"[ID04] DONE scale={scale_id} "
        f"snapshots={total_endpoints} "
        f"relations_per_snapshot={relation_count}",
        flush=True,
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

        "estimability_counts":
            {
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

    if not relation_frames:
        fail(
            "No snapshot relation frames were produced."
        )

    if not snapshot_frames:
        fail(
            "No snapshot metadata frames were produced."
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
