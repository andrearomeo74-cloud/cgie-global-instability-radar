"""
CGIE3-ID-04 frozen null controls.

Implements the three preregistered null procedures:

NULL-1 — temporal snapshot permutation
NULL-2 — relation-label permutation
NULL-3 — constrained weight/sign surrogate

The stage compares observed scale-level mean RCS against null
distributions generated under the frozen ID-04 protocol.

This module does not:

- change the observed snapshot population;
- optimize thresholds or scales;
- use earthquake-event information;
- assign the final scientific outcome;
- infer causality or prediction.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from engines.cgie3.src.id04.continuity import (
    edge_continuity,
    common_relation_table,
    weight_continuity,
    sign_continuity,
    topological_continuity,
    relational_continuity_score,
)

from engines.cgie3.src.id04.loader import (
    ID04ExperimentContext,
)


class ID04NullControlError(ValueError):
    """Raised when frozen ID-04 null testing violates the contract."""


def fail(message: str) -> None:
    """Raise a normalized null-control error."""
    raise ID04NullControlError(
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
        raise ID04NullControlError(
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
    """Validate prerequisites for frozen null controls."""
    if not isinstance(
        context,
        ID04ExperimentContext,
    ):
        fail(
            "context must be an ID04ExperimentContext."
        )

    if (
        context.runtime.get(
            "continuity_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Continuity analysis must complete before null testing."
        )

    required_outputs = {
        "snapshot_relations",
        "snapshots",
        "transition_continuity",
    }

    missing = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing:
        fail(
            "Null-control prerequisites are missing: "
            + ", ".join(
                missing
            )
        )


def get_contract(
    context: ID04ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate frozen null-control parameters."""
    null_controls = require_mapping(
        context.configuration.get(
            "null_controls"
        ),
        "null_controls",
    )

    if null_controls.get(
        "enabled"
    ) is not True:
        fail(
            "ID-04 null controls must remain enabled."
        )

    repetitions = require_positive_integer(
        null_controls.get(
            "repetitions"
        ),
        "null_controls.repetitions",
    )

    if repetitions != 1000:
        fail(
            "Frozen ID-04 null repetitions must remain 1000."
        )

    random_seed = require_positive_integer(
        null_controls.get(
            "random_seed"
        ),
        "null_controls.random_seed",
    )

    primary_nulls = null_controls.get(
        "primary_nulls"
    )

    if not isinstance(
        primary_nulls,
        list,
    ):
        fail(
            "null_controls.primary_nulls must be a list."
        )

    expected_nulls = (
        "temporal_snapshot_permutation",
        "relation_label_permutation",
        "constrained_weight_sign_surrogate",
    )

    if tuple(
        str(value)
        for value in primary_nulls
    ) != expected_nulls:
        fail(
            "Primary null order differs from the frozen contract."
        )

    significance = require_mapping(
        context.configuration.get(
            "empirical_significance"
        ),
        "empirical_significance",
    )

    if significance.get(
        "upper_tail"
    ) is not True:
        fail(
            "ID-04 empirical significance must remain upper-tail."
        )

    if significance.get(
        "finite_sample_correction"
    ) != "plus_one":
        fail(
            "ID-04 empirical p-values must use plus-one correction."
        )

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

    return {
        "repetitions":
            repetitions,

        "random_seed":
            random_seed,

        "primary_nulls":
            expected_nulls,

        "components":
            components,

        "minimum_estimable_components":
            int(
                rcs[
                    "minimum_estimable_components"
                ]
            ),
    }


def deterministic_seed(
    base_seed: int,
    null_id: str,
    scale_id: str,
    repetition: int,
) -> int:
    """Derive one stable deterministic seed."""
    token = (
        f"{base_seed}::{null_id}::{scale_id}::{repetition}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        token
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def component_weights(
    contract: Mapping[str, Any],
) -> dict[str, float]:
    """Return frozen component weights."""
    components = contract[
        "components"
    ]

    return {
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


def transition_rcs_from_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
) -> float | None:
    """Recompute one RCS from two supplied snapshot relation frames."""
    ec = edge_continuity(
        left,
        right,
    )

    common = common_relation_table(
        left,
        right,
    )

    wc_contract = contract[
        "components"
    ][
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

    sc_contract = contract[
        "components"
    ][
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

    tc_contract = contract[
        "components"
    ][
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

    rcs = relational_continuity_score(
        {
            "EC":
                ec,

            "WC":
                wc,

            "SC":
                sc,

            "TC":
                tc,
        },
        component_weights(
            contract
        ),
        minimum_estimable_components=contract[
            "minimum_estimable_components"
        ],
    )

    if rcs[
        "estimable"
    ] is not True:
        return None

    return float(
        rcs[
            "value"
        ]
    )


def observed_scale_mean_rcs(
    transitions: pd.DataFrame,
    scale_id: str,
) -> tuple[
    float | None,
    int,
]:
    """Return observed mean RCS and estimable transition count."""
    frame = transitions.loc[
        (
            transitions[
                "scale_id"
            ]
            == scale_id
        )
        & transitions[
            "RCS_estimable"
        ]
    ].copy()

    if frame.empty:
        return (
            None,
            0,
        )

    values = pd.to_numeric(
        frame[
            "RCS"
        ],
        errors="coerce",
    )

    values = values.loc[
        np.isfinite(
            values.to_numpy(
                dtype=float
            )
        )
    ]

    if values.empty:
        return (
            None,
            0,
        )

    return (
        float(
            values.mean()
        ),
        int(
            len(
                values
            )
        ),
    )


def snapshot_frames_by_scale(
    relations: pd.DataFrame,
    snapshots: pd.DataFrame,
    scale_id: str,
) -> list[
    tuple[
        str,
        pd.DataFrame,
    ]
]:
    """Return chronologically ordered snapshot relation frames."""
    scale_snapshots = snapshots.loc[
        snapshots[
            "scale_id"
        ]
        == scale_id
    ].sort_values(
        by="snapshot_index",
        kind="stable",
    )

    output: list[
        tuple[
            str,
            pd.DataFrame,
        ]
    ] = []

    for row in scale_snapshots.itertuples(
        index=False
    ):
        snapshot_id = str(
            row.snapshot_id
        )

        frame = relations.loc[
            relations[
                "snapshot_id"
            ]
            == snapshot_id
        ].copy()

        if frame.empty:
            fail(
                "Snapshot relation frame missing for "
                f"{snapshot_id}."
            )

        output.append(
            (
                snapshot_id,
                frame,
            )
        )

    return output


def mean_rcs_for_sequence(
    sequence: list[
        pd.DataFrame
    ],
    *,
    contract: Mapping[str, Any],
) -> tuple[
    float | None,
    int,
]:
    """Calculate mean RCS across consecutive frames in one sequence."""
    values: list[
        float
    ] = []

    for index in range(
        len(
            sequence
        )
        - 1
    ):
        value = transition_rcs_from_frames(
            sequence[
                index
            ],
            sequence[
                index + 1
            ],
            contract=contract,
        )

        if value is not None:
            values.append(
                value
            )

    if not values:
        return (
            None,
            0,
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        int(
            len(
                values
            )
        ),
    )


def temporal_snapshot_permutation(
    frames: list[
        pd.DataFrame
    ],
    *,
    rng: np.random.Generator,
    contract: Mapping[str, Any],
) -> tuple[
    float | None,
    int,
]:
    """NULL-1: destroy observed temporal ordering only."""
    if len(
        frames
    ) < 2:
        return (
            None,
            0,
        )

    order = rng.permutation(
        len(
            frames
        )
    )

    sequence = [
        frames[
            int(
                index
            )
        ]
        for index in order
    ]

    return mean_rcs_for_sequence(
        sequence,
        contract=contract,
    )


def permute_relation_labels_in_frame(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Permute relation identity while preserving observed values."""
    output = frame.copy()

    relation_identity = output[
        [
            "relation_id",
            "source_id",
            "target_id",
        ]
    ].copy()

    permutation = rng.permutation(
        len(
            relation_identity
        )
    )

    permuted = relation_identity.iloc[
        permutation
    ].reset_index(
        drop=True
    )

    output = output.reset_index(
        drop=True
    )

    output[
        [
            "relation_id",
            "source_id",
            "target_id",
        ]
    ] = permuted[
        [
            "relation_id",
            "source_id",
            "target_id",
        ]
    ]

    return output


def relation_label_permutation(
    frames: list[
        pd.DataFrame
    ],
    *,
    rng: np.random.Generator,
    contract: Mapping[str, Any],
) -> tuple[
    float | None,
    int,
]:
    """NULL-2: preserve values but destroy relation identity."""
    permuted_frames = [
        permute_relation_labels_in_frame(
            frame,
            rng=rng,
        )
        for frame in frames
    ]

    return mean_rcs_for_sequence(
        permuted_frames,
        contract=contract,
    )


def constrained_surrogate_frame(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build one constrained weight/sign surrogate.

    Snapshot row count and estimability states remain fixed.

    Among estimable rows:
    - absolute strengths are permuted;
    - signs are permuted independently;
    - relation identities are permuted.

    This preserves the marginal observed absolute-strength and sign
    distributions while breaking their original relational assignment.
    """
    output = frame.copy().reset_index(
        drop=True
    )

    estimable_mask = (
        output[
            "estimability"
        ]
        == "estimable"
    )

    estimable_indices = np.flatnonzero(
        estimable_mask.to_numpy()
    )

    if estimable_indices.size == 0:
        return output

    strengths = pd.to_numeric(
        output.loc[
            estimable_mask,
            "strength",
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    signs = pd.to_numeric(
        output.loc[
            estimable_mask,
            "sign",
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite_strengths = np.isfinite(
        strengths
    )

    if not finite_strengths.all():
        fail(
            "Estimable surrogate rows contain non-finite strengths."
        )

    absolute_strengths = np.abs(
        strengths
    )

    shuffled_absolute = rng.permutation(
        absolute_strengths
    )

    shuffled_signs = rng.permutation(
        signs
    )

    surrogate_strengths = (
        shuffled_absolute
        * shuffled_signs
    )

    output.loc[
        estimable_mask,
        "strength",
    ] = surrogate_strengths

    output.loc[
        estimable_mask,
        "absolute_strength",
    ] = shuffled_absolute

    output.loc[
        estimable_mask,
        "sign",
    ] = shuffled_signs

    identity = output.loc[
        estimable_mask,
        [
            "relation_id",
            "source_id",
            "target_id",
        ],
    ].copy()

    identity_permutation = rng.permutation(
        len(
            identity
        )
    )

    permuted_identity = identity.iloc[
        identity_permutation
    ].reset_index(
        drop=True
    )

    output.loc[
        estimable_mask,
        [
            "relation_id",
            "source_id",
            "target_id",
        ],
    ] = permuted_identity.to_numpy()

    return output


def constrained_weight_sign_surrogate(
    frames: list[
        pd.DataFrame
    ],
    *,
    rng: np.random.Generator,
    contract: Mapping[str, Any],
) -> tuple[
    float | None,
    int,
]:
    """NULL-3: constrained marginal weight/sign surrogate."""
    surrogate_frames = [
        constrained_surrogate_frame(
            frame,
            rng=rng,
        )
        for frame in frames
    ]

    return mean_rcs_for_sequence(
        surrogate_frames,
        contract=contract,
    )


def execute_null_repetition(
    null_id: str,
    frames: list[
        pd.DataFrame
    ],
    *,
    seed: int,
    contract: Mapping[str, Any],
) -> tuple[
    float | None,
    int,
]:
    """Execute one frozen null repetition."""
    rng = np.random.default_rng(
        seed
    )

    if (
        null_id
        == "temporal_snapshot_permutation"
    ):
        return temporal_snapshot_permutation(
            frames,
            rng=rng,
            contract=contract,
        )

    if (
        null_id
        == "relation_label_permutation"
    ):
        return relation_label_permutation(
            frames,
            rng=rng,
            contract=contract,
        )

    if (
        null_id
        == "constrained_weight_sign_surrogate"
    ):
        return constrained_weight_sign_surrogate(
            frames,
            rng=rng,
            contract=contract,
        )

    fail(
        f"Unknown frozen null ID: {null_id}"
    )


def empirical_upper_p_value(
    observed: float,
    null_values: np.ndarray,
) -> float:
    """Calculate plus-one corrected upper-tail empirical p-value."""
    if null_values.size == 0:
        fail(
            "Empirical p-value requires at least one null value."
        )

    exceedance_count = int(
        np.sum(
            null_values
            >= observed
        )
    )

    return float(
        (
            exceedance_count
            + 1
        )
        / (
            null_values.size
            + 1
        )
    )


def build_null_outputs(
    context: ID04ExperimentContext,
    contract: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Execute every frozen null repetition and summarize by scale/null."""
    relations = context.outputs[
        "snapshot_relations"
    ]

    snapshots = context.outputs[
        "snapshots"
    ]

    transitions = context.outputs[
        "transition_continuity"
    ]

    replication_records: list[
        dict[str, Any]
    ] = []

    summary_records: list[
        dict[str, Any]
    ] = []

    for scale_id in sorted(
        snapshots[
            "scale_id"
        ].astype(str).unique()
    ):
        observed_mean, observed_count = (
            observed_scale_mean_rcs(
                transitions,
                scale_id,
            )
        )

        frame_pairs = snapshot_frames_by_scale(
            relations,
            snapshots,
            scale_id,
        )

        frames = [
            frame
            for _, frame
            in frame_pairs
        ]

        for null_id in contract[
            "primary_nulls"
        ]:
            null_values: list[
                float
            ] = []

            estimable_transition_counts: list[
                int
            ] = []

            for repetition in range(
                1,
                contract[
                    "repetitions"
                ]
                + 1,
            ):
                seed = deterministic_seed(
                    contract[
                        "random_seed"
                    ],
                    null_id,
                    scale_id,
                    repetition,
                )

                (
                    null_mean,
                    estimable_transition_count,
                ) = execute_null_repetition(
                    null_id,
                    frames,
                    seed=seed,
                    contract=contract,
                )

                replication_records.append(
                    {
                        "experiment_id":
                            "CGIE3_ID_04",

                        "scale_id":
                            scale_id,

                        "null_id":
                            null_id,

                        "repetition":
                            int(
                                repetition
                            ),

                        "seed":
                            int(
                                seed
                            ),

                        "null_mean_RCS":
                            null_mean,

                        "estimable_transition_count":
                            int(
                                estimable_transition_count
                            ),

                        "event_information_used":
                            False,
                    }
                )

                if null_mean is not None:
                    null_values.append(
                        float(
                            null_mean
                        )
                    )

                    estimable_transition_counts.append(
                        int(
                            estimable_transition_count
                        )
                    )

            null_array = np.asarray(
                null_values,
                dtype=float,
            )

            if null_array.size == 0:
                null_median = None
                null_mean_value = None
                null_sd = None
                p_value = None
                effect = None

            else:
                null_median = float(
                    np.median(
                        null_array
                    )
                )

                null_mean_value = float(
                    np.mean(
                        null_array
                    )
                )

                null_sd = float(
                    np.std(
                        null_array,
                        ddof=1,
                    )
                ) if null_array.size > 1 else 0.0

                if observed_mean is None:
                    p_value = None
                    effect = None
                else:
                    p_value = (
                        empirical_upper_p_value(
                            float(
                                observed_mean
                            ),
                            null_array,
                        )
                    )

                    effect = float(
                        observed_mean
                        - null_median
                    )

            summary_records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_04",

                    "scale_id":
                        scale_id,

                    "null_id":
                        null_id,

                    "requested_repetitions":
                        int(
                            contract[
                                "repetitions"
                            ]
                        ),

                    "estimable_null_repetitions":
                        int(
                            null_array.size
                        ),

                    "observed_mean_RCS":
                        observed_mean,

                    "observed_estimable_transition_count":
                        int(
                            observed_count
                        ),

                    "null_mean_RCS":
                        null_mean_value,

                    "null_median_RCS":
                        null_median,

                    "null_sd_RCS":
                        null_sd,

                    "empirical_p_value":
                        p_value,

                    "effect_observed_minus_null_median":
                        effect,

                    "positive_effect":
                        (
                            bool(
                                effect > 0.0
                            )
                            if effect is not None
                            else False
                        ),

                    "event_information_used":
                        False,
                }
            )

    replications = pd.DataFrame.from_records(
        replication_records
    )

    summaries = pd.DataFrame.from_records(
        summary_records
    )

    if replications.empty:
        fail(
            "Null-control replication table is empty."
        )

    if summaries.empty:
        fail(
            "Null-control summary table is empty."
        )

    expected_replications = (
        snapshots[
            "scale_id"
        ].nunique()
        * len(
            contract[
                "primary_nulls"
            ]
        )
        * contract[
            "repetitions"
        ]
    )

    if len(
        replications
    ) != expected_replications:
        fail(
            "Unexpected null replication count: "
            f"expected {expected_replications}, "
            f"observed {len(replications)}."
        )

    return (
        replications.sort_values(
            by=[
                "scale_id",
                "null_id",
                "repetition",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
        summaries.sort_values(
            by=[
                "scale_id",
                "null_id",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
    )


def build_null_summary(
    replications: pd.DataFrame,
    summaries: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive null-control stage summary."""
    return {
        "status":
            "COMPLETED",

        "null_replication_count":
            int(
                len(
                    replications
                )
            ),

        "null_summary_count":
            int(
                len(
                    summaries
                )
            ),

        "scale_count":
            int(
                summaries[
                    "scale_id"
                ].nunique()
            ),

        "null_type_count":
            int(
                summaries[
                    "null_id"
                ].nunique()
            ),

        "all_event_information_flags_false":
            bool(
                not replications[
                    "event_information_used"
                ].any()
                and not summaries[
                    "event_information_used"
                ].any()
            ),

        "multiple_testing_correction_applied":
            False,

        "final_scientific_outcome_assigned":
            False,
    }


def run_null_controls(
    context: ID04ExperimentContext,
) -> ID04ExperimentContext:
    """Execute all frozen CGIE3-ID-04 null controls."""
    validate_context(
        context
    )

    contract = get_contract(
        context
    )

    (
        replications,
        summaries,
    ) = build_null_outputs(
        context,
        contract,
    )

    summary = build_null_summary(
        replications,
        summaries,
    )

    context.register_output(
        "null_controls",
        replications,
    )

    context.register_output(
        "null_summaries",
        summaries,
    )

    context.register_output(
        "null_control_summary",
        summary,
    )

    context.register_runtime(
        "null_control_status",
        "COMPLETED",
    )

    return context
