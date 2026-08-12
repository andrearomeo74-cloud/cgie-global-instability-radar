"""
CGIE3-ID-04 relational continuity metrics.

Implements the four frozen continuity components:

EC — Edge Continuity
WC — Weight Continuity
SC — Sign Continuity
TC — Topological Continuity

and the composite:

RCS — Relational Continuity Score

Only consecutive snapshots within the same frozen temporal scale
are compared.

This module does not perform null testing, event alignment,
threshold optimization, prediction, or causal inference.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cgie3.src.id04.loader import (
    ID04ExperimentContext,
)


class ID04ContinuityError(ValueError):
    """Raised when the frozen continuity contract is violated."""


def fail(message: str) -> None:
    """Raise normalized ID-04 continuity error."""
    raise ID04ContinuityError(
        str(message).strip()
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like object."""
    if not isinstance(
        value,
        Mapping,
    ):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def validate_context(
    context: ID04ExperimentContext,
) -> None:
    """Validate prerequisites for continuity analysis."""
    if not isinstance(
        context,
        ID04ExperimentContext,
    ):
        fail(
            "context must be an ID04ExperimentContext."
        )

    if (
        context.runtime.get(
            "snapshot_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Snapshot construction must complete "
            "before continuity analysis."
        )

    if "snapshot_relations" not in context.outputs:
        fail(
            "snapshot_relations output is missing."
        )

    if "snapshots" not in context.outputs:
        fail(
            "snapshots output is missing."
        )


def get_contract(
    context: ID04ExperimentContext,
) -> dict[str, Any]:
    """Load and validate frozen continuity parameters."""
    components = require_mapping(
        context.configuration.get(
            "continuity_components"
        ),
        "continuity_components",
    )

    rcs = require_mapping(
        context.configuration.get(
            "relational_continuity_score"
        ),
        "relational_continuity_score",
    )

    transition_definition = require_mapping(
        context.configuration.get(
            "transition_definition"
        ),
        "transition_definition",
    )

    if transition_definition.get(
        "ordering"
    ) != "chronological":
        fail(
            "Transition ordering must remain chronological."
        )

    comparison = require_mapping(
        transition_definition.get(
            "comparison"
        ),
        "transition_definition.comparison",
    )

    if comparison.get(
        "consecutive_snapshots_only"
    ) is not True:
        fail(
            "Primary continuity analysis must compare "
            "consecutive snapshots only."
        )

    component_ids = (
        "edge_continuity",
        "weight_continuity",
        "sign_continuity",
        "topological_continuity",
    )

    normalized_components: dict[
        str,
        dict[str, Any],
    ] = {}

    total_weight = 0.0

    for component_name in component_ids:
        component = require_mapping(
            components.get(
                component_name
            ),
            (
                "continuity_components."
                f"{component_name}"
            ),
        )

        if component.get(
            "enabled"
        ) is not True:
            fail(
                f"{component_name} must remain enabled."
            )

        weight = float(
            component.get(
                "weight"
            )
        )

        if not np.isfinite(
            weight
        ):
            fail(
                f"{component_name} weight must be finite."
            )

        if weight <= 0.0:
            fail(
                f"{component_name} weight must be positive."
            )

        total_weight += weight

        normalized_components[
            component_name
        ] = dict(
            component
        )

    if not np.isclose(
        total_weight,
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        fail(
            "Frozen continuity component weights must sum to 1."
        )

    if rcs.get(
        "combine_only_estimable_components"
    ) is not True:
        fail(
            "RCS must combine only estimable components."
        )

    if rcs.get(
        "missing_component_policy"
    ) != "renormalize_available_weights":
        fail(
            "Unexpected RCS missing-component policy."
        )

    minimum_estimable = int(
        rcs.get(
            "minimum_estimable_components"
        )
    )

    if minimum_estimable != 2:
        fail(
            "Frozen RCS minimum estimable component count "
            "must remain 2."
        )

    if rcs.get(
        "weights_fixed"
    ) is not True:
        fail(
            "RCS weights must remain frozen."
        )

    if rcs.get(
        "optimize_weights"
    ) is not False:
        fail(
            "RCS weight optimization is prohibited."
        )

    return {
        "components":
            normalized_components,

        "minimum_estimable_components":
            minimum_estimable,
    }


def estimable_relations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return relations with a finite estimated strength."""
    output = frame.loc[
        frame[
            "estimability"
        ]
        == "estimable"
    ].copy()

    output[
        "strength"
    ] = pd.to_numeric(
        output[
            "strength"
        ],
        errors="coerce",
    )

    output = output.loc[
        np.isfinite(
            output[
                "strength"
            ].to_numpy(
                dtype=float
            )
        )
    ].copy()

    return output.reset_index(
        drop=True
    )


def relation_key_set(
    frame: pd.DataFrame,
) -> set[str]:
    """Return the estimable relation-ID set."""
    return set(
        frame[
            "relation_id"
        ].astype(str)
    )


def edge_continuity(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> dict[str, Any]:
    """Compute frozen Jaccard Edge Continuity."""
    left_estimable = estimable_relations(
        left
    )

    right_estimable = estimable_relations(
        right
    )

    left_edges = relation_key_set(
        left_estimable
    )

    right_edges = relation_key_set(
        right_estimable
    )

    union = (
        left_edges
        | right_edges
    )

    intersection = (
        left_edges
        & right_edges
    )

    if not union:
        return {
            "value":
                None,

            "estimable":
                False,

            "reason":
                "no_estimable_edges_in_either_snapshot",

            "left_edge_count":
                0,

            "right_edge_count":
                0,

            "common_edge_count":
                0,

            "union_edge_count":
                0,
        }

    value = float(
        len(
            intersection
        )
        / len(
            union
        )
    )

    return {
        "value":
            value,

        "estimable":
            True,

        "reason":
            None,

        "left_edge_count":
            int(
                len(
                    left_edges
                )
            ),

        "right_edge_count":
            int(
                len(
                    right_edges
                )
            ),

        "common_edge_count":
            int(
                len(
                    intersection
                )
            ),

        "union_edge_count":
            int(
                len(
                    union
                )
            ),
    }


def common_relation_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """Join estimable relations present in both snapshots."""
    left_estimable = estimable_relations(
        left
    )

    right_estimable = estimable_relations(
        right
    )

    left_columns = left_estimable[
        [
            "relation_id",
            "source_id",
            "target_id",
            "strength",
            "sign",
        ]
    ].rename(
        columns={
            "strength":
                "strength_left",

            "sign":
                "sign_left",
        }
    )

    right_columns = right_estimable[
        [
            "relation_id",
            "source_id",
            "target_id",
            "strength",
            "sign",
        ]
    ].rename(
        columns={
            "strength":
                "strength_right",

            "sign":
                "sign_right",
        }
    )

    common = left_columns.merge(
        right_columns,
        on=[
            "relation_id",
            "source_id",
            "target_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    return common.sort_values(
        by=[
            "source_id",
            "target_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def safe_spearman(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    """Return finite Spearman correlation or explicit non-estimability."""
    if left.size != right.size:
        fail(
            "Spearman input vectors have different lengths."
        )

    if left.size == 0:
        return None

    if np.unique(
        left
    ).size < 2:
        return None

    if np.unique(
        right
    ).size < 2:
        return None

    result = spearmanr(
        left,
        right,
    )

    value = float(
        result.statistic
    )

    if not np.isfinite(
        value
    ):
        return None

    return value


def weight_continuity(
    common: pd.DataFrame,
    *,
    minimum_common_edges: int,
) -> dict[str, Any]:
    """Compute frozen rank-based Weight Continuity."""
    common_count = int(
        len(
            common
        )
    )

    if common_count < minimum_common_edges:
        return {
            "value":
                None,

            "estimable":
                False,

            "reason":
                "insufficient_common_edges",

            "common_edge_count":
                common_count,
        }

    left = common[
        "strength_left"
    ].to_numpy(
        dtype=float
    )

    right = common[
        "strength_right"
    ].to_numpy(
        dtype=float
    )

    value = safe_spearman(
        left,
        right,
    )

    if value is None:
        return {
            "value":
                None,

            "estimable":
                False,

            "reason":
                "non_identifiable_rank_order",

            "common_edge_count":
                common_count,
        }

    return {
        "value":
            float(
                value
            ),

        "estimable":
            True,

        "reason":
            None,

        "common_edge_count":
            common_count,
    }


def sign_continuity(
    common: pd.DataFrame,
    *,
    minimum_common_signed_edges: int,
) -> dict[str, Any]:
    """Compute frozen Sign Continuity."""
    signed = common.copy()

    signed[
        "sign_left"
    ] = pd.to_numeric(
        signed[
            "sign_left"
        ],
        errors="coerce",
    )

    signed[
        "sign_right"
    ] = pd.to_numeric(
        signed[
            "sign_right"
        ],
        errors="coerce",
    )

    finite = (
        np.isfinite(
            signed[
                "sign_left"
            ].to_numpy(
                dtype=float
            )
        )
        & np.isfinite(
            signed[
                "sign_right"
            ].to_numpy(
                dtype=float
            )
        )
    )

    signed = signed.loc[
        finite
    ].copy()

    common_signed_count = int(
        len(
            signed
        )
    )

    if (
        common_signed_count
        < minimum_common_signed_edges
    ):
        return {
            "value":
                None,

            "estimable":
                False,

            "reason":
                "insufficient_common_signed_edges",

            "common_signed_edge_count":
                common_signed_count,

            "same_sign_count":
                0,

            "sign_reversal_count":
                0,
        }

    same_sign = (
        signed[
            "sign_left"
        ].to_numpy(
            dtype=float
        )
        == signed[
            "sign_right"
        ].to_numpy(
            dtype=float
        )
    )

    same_sign_count = int(
        same_sign.sum()
    )

    sign_reversal_count = int(
        common_signed_count
        - same_sign_count
    )

    value = float(
        same_sign_count
        / common_signed_count
    )

    return {
        "value":
            value,

        "estimable":
            True,

        "reason":
            None,

        "common_signed_edge_count":
            common_signed_count,

        "same_sign_count":
            same_sign_count,

        "sign_reversal_count":
            sign_reversal_count,
    }


def node_strength_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute weighted node strength from estimable relations.

    Absolute relation strength is used so that node role represents
    amount of relational participation rather than cancellation of
    positive and negative edges.
    """
    estimable = estimable_relations(
        frame
    )

    if estimable.empty:
        return pd.DataFrame(
            columns=[
                "node_id",
                "node_strength",
            ]
        )

    records: list[
        tuple[str, float]
    ] = []

    for row in estimable.itertuples(
        index=False
    ):
        weight = abs(
            float(
                row.strength
            )
        )

        records.append(
            (
                str(
                    row.source_id
                ),
                weight,
            )
        )

        records.append(
            (
                str(
                    row.target_id
                ),
                weight,
            )
        )

    node_frame = pd.DataFrame(
        records,
        columns=[
            "node_id",
            "weight",
        ],
    )

    output = (
        node_frame.groupby(
            "node_id",
            as_index=False,
            sort=True,
        )[
            "weight"
        ]
        .sum()
        .rename(
            columns={
                "weight":
                    "node_strength"
            }
        )
    )

    return output


def topological_continuity(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    minimum_common_nodes: int,
) -> dict[str, Any]:
    """Compute frozen node-role Topological Continuity."""
    left_nodes = node_strength_table(
        left
    ).rename(
        columns={
            "node_strength":
                "strength_left"
        }
    )

    right_nodes = node_strength_table(
        right
    ).rename(
        columns={
            "node_strength":
                "strength_right"
        }
    )

    common = left_nodes.merge(
        right_nodes,
        on="node_id",
        how="inner",
        validate="one_to_one",
    )

    common_node_count = int(
        len(
            common
        )
    )

    if common_node_count < minimum_common_nodes:
        return {
            "value":
                None,

            "estimable":
                False,

            "reason":
                "insufficient_common_nodes",

            "common_node_count":
                common_node_count,
        }

    value = safe_spearman(
        common[
            "strength_left"
        ].to_numpy(
            dtype=float
        ),
        common[
            "strength_right"
        ].to_numpy(
            dtype=float
        ),
    )

    if value is None:
        return {
            "value":
                None,

            "estimable":
                False,

            "reason":
                "non_identifiable_node_role_ranking",

            "common_node_count":
                common_node_count,
        }

    return {
        "value":
            float(
                value
            ),

        "estimable":
            True,

        "reason":
            None,

        "common_node_count":
            common_node_count,
    }


def normalize_for_rcs(
    component_id: str,
    value: float,
) -> float:
    """
    Map every estimable continuity component to [0, 1].

    EC and SC already lie in [0, 1].

    WC and TC are Spearman correlations in [-1, 1], so they are
    linearly mapped using (rho + 1) / 2 before composition.

    Raw values remain separately preserved in official outputs.
    """
    if component_id in {
        "EC",
        "SC",
    }:
        normalized = float(
            value
        )

    elif component_id in {
        "WC",
        "TC",
    }:
        normalized = float(
            (
                value
                + 1.0
            )
            / 2.0
        )

    else:
        fail(
            "Unknown continuity component ID: "
            f"{component_id}"
        )

    if not np.isfinite(
        normalized
    ):
        fail(
            f"{component_id} normalization produced "
            "a non-finite value."
        )

    tolerance = 1e-12

    if (
        normalized
        < -tolerance
        or normalized
        > 1.0 + tolerance
    ):
        fail(
            f"{component_id} normalized value is outside [0, 1]."
        )

    return float(
        np.clip(
            normalized,
            0.0,
            1.0,
        )
    )


def relational_continuity_score(
    component_results: Mapping[
        str,
        Mapping[str, Any],
    ],
    component_weights: Mapping[
        str,
        float,
    ],
    *,
    minimum_estimable_components: int,
) -> dict[str, Any]:
    """Combine estimable components using frozen renormalized weights."""
    estimable: list[
        tuple[
            str,
            float,
            float,
        ]
    ] = []

    for component_id in (
        "EC",
        "WC",
        "SC",
        "TC",
    ):
        result = component_results[
            component_id
        ]

        if result.get(
            "estimable"
        ) is not True:
            continue

        raw_value = result.get(
            "value"
        )

        if raw_value is None:
            continue

        normalized = normalize_for_rcs(
            component_id,
            float(
                raw_value
            ),
        )

        estimable.append(
            (
                component_id,
                normalized,
                float(
                    component_weights[
                        component_id
                    ]
                ),
            )
        )

    component_count = int(
        len(
            estimable
        )
    )

    if (
        component_count
        < minimum_estimable_components
    ):
        return {
            "value":
                None,

            "estimable":
                False,

            "reason":
                "insufficient_estimable_components",

            "estimable_component_count":
                component_count,

            "effective_weight_sum":
                float(
                    sum(
                        weight
                        for _, _, weight
                        in estimable
                    )
                ),
        }

    weight_sum = float(
        sum(
            weight
            for _, _, weight
            in estimable
        )
    )

    if weight_sum <= 0.0:
        fail(
            "Estimable RCS component weight sum must be positive."
        )

    value = float(
        sum(
            normalized
            * weight
            for _, normalized, weight
            in estimable
        )
        / weight_sum
    )

    return {
        "value":
            value,

        "estimable":
            True,

        "reason":
            None,

        "estimable_component_count":
            component_count,

        "effective_weight_sum":
            weight_sum,
    }


def build_transition(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_snapshot: pd.Series,
    right_snapshot: pd.Series,
    *,
    contract: Mapping[str, Any],
    transition_index: int,
) -> dict[str, Any]:
    """Calculate all frozen metrics for one consecutive transition."""
    components = contract[
        "components"
    ]

    ec = edge_continuity(
        left,
        right,
    )

    common = common_relation_table(
        left,
        right,
    )

    wc_contract = components[
        "weight_continuity"
    ]

    wc = weight_continuity(
        common,
        minimum_common_edges=int(
            wc_contract[
                "minimum_common_edges"
            ]
        ),
    )

    sc_contract = components[
        "sign_continuity"
    ]

    sc = sign_continuity(
        common,
        minimum_common_signed_edges=int(
            sc_contract[
                "minimum_common_signed_edges"
            ]
        ),
    )

    tc_contract = components[
        "topological_continuity"
    ]

    tc = topological_continuity(
        left,
        right,
        minimum_common_nodes=int(
            tc_contract[
                "minimum_common_nodes"
            ]
        ),
    )

    component_results = {
        "EC":
            ec,

        "WC":
            wc,

        "SC":
            sc,

        "TC":
            tc,
    }

    component_weights = {
        "EC":
            float(
                components[
                    "edge_continuity"
                ][
                    "weight"
                ]
            ),

        "WC":
            float(
                components[
                    "weight_continuity"
                ][
                    "weight"
                ]
            ),

        "SC":
            float(
                components[
                    "sign_continuity"
                ][
                    "weight"
                ]
            ),

        "TC":
            float(
                components[
                    "topological_continuity"
                ][
                    "weight"
                ]
            ),
    }

    rcs = relational_continuity_score(
        component_results,
        component_weights,
        minimum_estimable_components=contract[
            "minimum_estimable_components"
        ],
    )

    transition_id = (
        "CGIE3_ID_04::"
        f"{left_snapshot['scale_id']}::"
        f"T{transition_index:06d}"
    )

    return {
        "experiment_id":
            "CGIE3_ID_04",

        "transition_id":
            transition_id,

        "transition_index":
            int(
                transition_index
            ),

        "scale_id":
            str(
                left_snapshot[
                    "scale_id"
                ]
            ),

        "left_snapshot_id":
            str(
                left_snapshot[
                    "snapshot_id"
                ]
            ),

        "right_snapshot_id":
            str(
                right_snapshot[
                    "snapshot_id"
                ]
            ),

        "left_snapshot_end_utc":
            str(
                left_snapshot[
                    "snapshot_end_utc"
                ]
            ),

        "right_snapshot_end_utc":
            str(
                right_snapshot[
                    "snapshot_end_utc"
                ]
            ),

        "EC":
            ec[
                "value"
            ],

        "EC_estimable":
            bool(
                ec[
                    "estimable"
                ]
            ),

        "EC_reason":
            ec[
                "reason"
            ],

        "WC":
            wc[
                "value"
            ],

        "WC_estimable":
            bool(
                wc[
                    "estimable"
                ]
            ),

        "WC_reason":
            wc[
                "reason"
            ],

        "SC":
            sc[
                "value"
            ],

        "SC_estimable":
            bool(
                sc[
                    "estimable"
                ]
            ),

        "SC_reason":
            sc[
                "reason"
            ],

        "TC":
            tc[
                "value"
            ],

        "TC_estimable":
            bool(
                tc[
                    "estimable"
                ]
            ),

        "TC_reason":
            tc[
                "reason"
            ],

        "RCS":
            rcs[
                "value"
            ],

        "RCS_estimable":
            bool(
                rcs[
                    "estimable"
                ]
            ),

        "RCS_reason":
            rcs[
                "reason"
            ],

        "estimable_component_count":
            int(
                rcs[
                    "estimable_component_count"
                ]
            ),

        "effective_weight_sum":
            float(
                rcs[
                    "effective_weight_sum"
                ]
            ),

        "left_estimable_edge_count":
            int(
                ec[
                    "left_edge_count"
                ]
            ),

        "right_estimable_edge_count":
            int(
                ec[
                    "right_edge_count"
                ]
            ),

        "common_edge_count":
            int(
                ec[
                    "common_edge_count"
                ]
            ),

        "union_edge_count":
            int(
                ec[
                    "union_edge_count"
                ]
            ),

        "common_signed_edge_count":
            int(
                sc[
                    "common_signed_edge_count"
                ]
            ),

        "sign_reversal_count":
            int(
                sc[
                    "sign_reversal_count"
                ]
            ),

        "common_node_count":
            int(
                tc[
                    "common_node_count"
                ]
            ),

        "event_information_used":
            False,

        "post_hoc_parameter_selection":
            False,
    }


def validate_transition_table(
    transitions: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> None:
    """Validate chronology and structural completeness."""
    if transitions.empty:
        fail(
            "Continuity analysis produced no transitions."
        )

    if transitions[
        "transition_id"
    ].duplicated().any():
        fail(
            "Duplicate transition IDs detected."
        )

    if transitions[
        "event_information_used"
    ].any():
        fail(
            "Event information entered the ID-04 continuity stage."
        )

    if transitions[
        "post_hoc_parameter_selection"
    ].any():
        fail(
            "Post-hoc parameter selection detected."
        )

    for scale_id, scale_snapshots in snapshots.groupby(
        "scale_id",
        sort=True,
    ):
        scale_transitions = transitions.loc[
            transitions[
                "scale_id"
            ]
            == scale_id
        ]

        expected = max(
            int(
                len(
                    scale_snapshots
                )
            )
            - 1,
            0,
        )

        observed = int(
            len(
                scale_transitions
            )
        )

        if observed != expected:
            fail(
                f"Scale {scale_id} requires {expected} "
                f"consecutive transitions; observed {observed}."
            )

        ordered = scale_transitions.sort_values(
            by="transition_index",
            kind="stable",
        )

        left_times = pd.to_datetime(
            ordered[
                "left_snapshot_end_utc"
            ],
            utc=True,
            errors="coerce",
        )

        right_times = pd.to_datetime(
            ordered[
                "right_snapshot_end_utc"
            ],
            utc=True,
            errors="coerce",
        )

        if (
            left_times.isna().any()
            or right_times.isna().any()
        ):
            fail(
                f"Invalid transition timestamps in scale {scale_id}."
            )

        if not (
            right_times.to_numpy()
            > left_times.to_numpy()
        ).all():
            fail(
                f"Non-chronological transition detected in {scale_id}."
            )


def build_continuity_summary(
    transitions: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive continuity-stage summary without inference."""
    by_scale: dict[
        str,
        Any,
    ] = {}

    for scale_id, frame in transitions.groupby(
        "scale_id",
        sort=True,
    ):
        estimable = frame.loc[
            frame[
                "RCS_estimable"
            ]
        ].copy()

        if estimable.empty:
            mean_rcs = None
            median_rcs = None
            minimum_rcs = None
            maximum_rcs = None
        else:
            mean_rcs = float(
                estimable[
                    "RCS"
                ].mean()
            )

            median_rcs = float(
                estimable[
                    "RCS"
                ].median()
            )

            minimum_rcs = float(
                estimable[
                    "RCS"
                ].min()
            )

            maximum_rcs = float(
                estimable[
                    "RCS"
                ].max()
            )

        by_scale[
            str(
                scale_id
            )
        ] = {
            "transition_count":
                int(
                    len(
                        frame
                    )
                ),

            "estimable_transition_count":
                int(
                    len(
                        estimable
                    )
                ),

            "non_estimable_transition_count":
                int(
                    len(
                        frame
                    )
                    - len(
                        estimable
                    )
                ),

            "mean_RCS":
                mean_rcs,

            "median_RCS":
                median_rcs,

            "minimum_RCS":
                minimum_rcs,

            "maximum_RCS":
                maximum_rcs,
        }

    return {
        "status":
            "COMPLETED",

        "transition_count":
            int(
                len(
                    transitions
                )
            ),

        "estimable_transition_count":
            int(
                transitions[
                    "RCS_estimable"
                ].sum()
            ),

        "scale_count":
            int(
                transitions[
                    "scale_id"
                ].nunique()
            ),

        "by_scale":
            by_scale,

        "null_testing_performed":
            False,

        "scientific_outcome_assigned":
            False,

        "event_information_used":
            False,
    }


def compute_continuity(
    context: ID04ExperimentContext,
) -> ID04ExperimentContext:
    """Execute frozen consecutive-snapshot continuity analysis."""
    validate_context(
        context
    )

    contract = get_contract(
        context
    )

    relations = context.outputs[
        "snapshot_relations"
    ].copy()

    snapshots = context.outputs[
        "snapshots"
    ].copy()

    records: list[
        dict[str, Any]
    ] = []

    for scale_id, scale_snapshots in snapshots.groupby(
        "scale_id",
        sort=True,
    ):
        ordered_snapshots = scale_snapshots.sort_values(
            by="snapshot_index",
            kind="stable",
        ).reset_index(
            drop=True
        )

        if len(
            ordered_snapshots
        ) < 2:
            continue

        for position in range(
            len(
                ordered_snapshots
            )
            - 1
        ):
            left_snapshot = ordered_snapshots.iloc[
                position
            ]

            right_snapshot = ordered_snapshots.iloc[
                position + 1
            ]

            left = relations.loc[
                relations[
                    "snapshot_id"
                ]
                == left_snapshot[
                    "snapshot_id"
                ]
            ].copy()

            right = relations.loc[
                relations[
                    "snapshot_id"
                ]
                == right_snapshot[
                    "snapshot_id"
                ]
            ].copy()

            if left.empty:
                fail(
                    "Missing left snapshot relations: "
                    f"{left_snapshot['snapshot_id']}"
                )

            if right.empty:
                fail(
                    "Missing right snapshot relations: "
                    f"{right_snapshot['snapshot_id']}"
                )

            records.append(
                build_transition(
                    left,
                    right,
                    left_snapshot,
                    right_snapshot,
                    contract=contract,
                    transition_index=position + 1,
                )
            )

    transitions = pd.DataFrame.from_records(
        records
    )

    transitions = transitions.sort_values(
        by=[
            "scale_id",
            "transition_index",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    validate_transition_table(
        transitions,
        snapshots,
    )

    summary = build_continuity_summary(
        transitions
    )

    context.register_output(
        "transition_continuity",
        transitions,
    )

    context.register_output(
        "continuity_summary",
        summary,
    )

    context.register_runtime(
        "continuity_status",
        "COMPLETED",
    )

    return context
    
