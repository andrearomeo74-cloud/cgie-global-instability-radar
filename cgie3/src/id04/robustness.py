"""
CGIE3-ID-04 robustness and scientific decision stage.

This stage applies:

- Benjamini-Hochberg FDR across temporal scales;
- frozen positive-effect requirements;
- leave-one-transition-out robustness;
- final preregistered scientific outcome assignment.

Allowed outcomes:

- CONTINUITY_SUPPORTED
- PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE
- CONTINUITY_NOT_SUPPORTED
- NON_IDENTIFIABLE

This module does not:

- change temporal scales;
- change thresholds;
- optimize weights;
- use earthquake-event information;
- infer causality;
- establish prediction.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from engines.cgie3.src.id04.loader import (
    ID04ExperimentContext,
)


class ID04RobustnessError(ValueError):
    """Raised when frozen robustness rules are violated."""


def fail(message: str) -> None:
    """Raise normalized robustness error."""
    raise ID04RobustnessError(
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
    """Validate prerequisites for robustness and decision."""
    if not isinstance(
        context,
        ID04ExperimentContext,
    ):
        fail(
            "context must be an ID04ExperimentContext."
        )

    if (
        context.runtime.get(
            "null_control_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Null controls must complete before robustness."
        )

    required_outputs = {
        "transition_continuity",
        "null_summaries",
    }

    missing = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing:
        fail(
            "Robustness prerequisites are missing: "
            + ", ".join(
                missing
            )
        )


def get_contract(
    context: ID04ExperimentContext,
) -> dict[str, Any]:
    """Extract frozen scientific decision rules."""
    significance = require_mapping(
        context.configuration.get(
            "empirical_significance"
        ),
        "empirical_significance",
    )

    multiple_testing = require_mapping(
        significance.get(
            "multiple_testing"
        ),
        "empirical_significance.multiple_testing",
    )

    if (
        multiple_testing.get(
            "method"
        )
        != "benjamini_hochberg"
    ):
        fail(
            "Multiple-testing method must remain "
            "Benjamini-Hochberg."
        )

    fdr = float(
        multiple_testing.get(
            "fdr"
        )
    )

    if not np.isclose(
        fdr,
        0.05,
        atol=1e-12,
    ):
        fail(
            "Frozen FDR threshold must remain 0.05."
        )

    robustness = require_mapping(
        context.configuration.get(
            "robustness"
        ),
        "robustness",
    )

    leave_one = require_mapping(
        robustness.get(
            "leave_one_transition_out"
        ),
        "robustness.leave_one_transition_out",
    )

    if leave_one.get(
        "enabled"
    ) is not True:
        fail(
            "Leave-one-transition-out must remain enabled."
        )

    minimum_retention = float(
        robustness.get(
            "minimum_conclusion_retention_fraction"
        )
    )

    if not np.isclose(
        minimum_retention,
        0.80,
        atol=1e-12,
    ):
        fail(
            "Frozen robustness threshold must remain 0.80."
        )

    outcomes = require_mapping(
        context.configuration.get(
            "outcomes"
        ),
        "outcomes",
    )

    allowed = tuple(
        str(
            value
        )
        for value in outcomes.get(
            "allowed",
            [],
        )
    )

    expected_allowed = (
        "CONTINUITY_SUPPORTED",
        "PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE",
        "CONTINUITY_NOT_SUPPORTED",
        "NON_IDENTIFIABLE",
    )

    if allowed != expected_allowed:
        fail(
            "Allowed scientific outcomes differ from "
            "the frozen contract."
        )

    supported = require_mapping(
        outcomes.get(
            "continuity_supported"
        ),
        "outcomes.continuity_supported",
    )

    partial = require_mapping(
        outcomes.get(
            "partial_or_scale_specific_evidence"
        ),
        (
            "outcomes."
            "partial_or_scale_specific_evidence"
        ),
    )

    not_supported = require_mapping(
        outcomes.get(
            "continuity_not_supported"
        ),
        "outcomes.continuity_not_supported",
    )

    non_identifiable = require_mapping(
        outcomes.get(
            "non_identifiable"
        ),
        "outcomes.non_identifiable",
    )

    estimability = require_mapping(
        context.configuration.get(
            "estimability"
        ),
        "estimability",
    )

    return {
        "fdr":
            fdr,

        "minimum_retention":
            minimum_retention,

        "single_transition_dependency_prohibited":
            bool(
                robustness.get(
                    "single_transition_dependency_prohibited"
                )
            ),

        "minimum_significant_scales_full":
            int(
                supported[
                    "minimum_significant_scales"
                ]
            ),

        "require_positive_effect_full":
            bool(
                supported[
                    "require_positive_effect_on_all_supporting_scales"
                ]
            ),

        "required_robustness_full":
            float(
                supported[
                    "require_robustness_fraction"
                ]
            ),

        "minimum_significant_scales_partial":
            int(
                partial[
                    "minimum_significant_scales"
                ]
            ),

        "not_supported_requires_no_scale":
            bool(
                not_supported[
                    "no_scale_meets_corrected_significance"
                ]
            ),

        "non_identifiable_requires_no_estimable_scales":
            bool(
                non_identifiable[
                    "no_estimable_scales"
                ]
            ),

        "minimum_estimable_transitions_per_scale":
            int(
                estimability[
                    "minimum_estimable_transition_count_per_scale"
                ]
            ),

        "allowed_outcomes":
            expected_allowed,
    }


def benjamini_hochberg(
    p_values: Mapping[str, float],
    *,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    """
    Apply Benjamini-Hochberg correction.

    Returns corrected q-values and significance flags for each scale.
    """
    if not p_values:
        return {}

    ordered = sorted(
        (
            (
                str(
                    scale_id
                ),
                float(
                    p_value
                ),
            )
            for scale_id, p_value
            in p_values.items()
        ),
        key=lambda item: (
            item[1],
            item[0],
        ),
    )

    m = len(
        ordered
    )

    raw_q: list[
        tuple[
            str,
            float,
            float,
        ]
    ] = []

    for rank, (
        scale_id,
        p_value,
    ) in enumerate(
        ordered,
        start=1,
    ):
        q_value = (
            p_value
            * m
            / rank
        )

        raw_q.append(
            (
                scale_id,
                p_value,
                float(
                    min(
                        q_value,
                        1.0,
                    )
                ),
            )
        )

    adjusted: dict[
        str,
        dict[str, Any],
    ] = {}

    running_min = 1.0

    for scale_id, p_value, raw in reversed(
        raw_q
    ):
        running_min = min(
            running_min,
            raw,
        )

        adjusted[
            scale_id
        ] = {
            "p_value":
                float(
                    p_value
                ),

            "q_value":
                float(
                    running_min
                ),

            "significant":
                bool(
                    running_min
                    <= alpha
                ),
        }

    return adjusted


def combine_null_evidence_by_scale(
    null_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse the three frozen null tests into one conservative scale test.

    A scale only supports continuity when it exceeds every primary null.

    Therefore the conservative scale-level empirical p-value is the
    maximum of the three null-specific empirical p-values.
    """
    records: list[
        dict[str, Any]
    ] = []

    for scale_id, frame in null_summaries.groupby(
        "scale_id",
        sort=True,
    ):
        p_values = pd.to_numeric(
            frame[
                "empirical_p_value"
            ],
            errors="coerce",
        )

        effects = pd.to_numeric(
            frame[
                "effect_observed_minus_null_median"
            ],
            errors="coerce",
        )

        valid_p = p_values.loc[
            np.isfinite(
                p_values.to_numpy(
                    dtype=float
                )
            )
        ]

        valid_effect = effects.loc[
            np.isfinite(
                effects.to_numpy(
                    dtype=float
                )
            )
        ]

        estimable = bool(
            len(
                frame
            )
            == 3
            and len(
                valid_p
            )
            == 3
            and len(
                valid_effect
            )
            == 3
        )

        if estimable:
            conservative_p = float(
                valid_p.max()
            )

            minimum_effect = float(
                valid_effect.min()
            )

            all_positive = bool(
                (
                    valid_effect
                    > 0.0
                ).all()
            )

        else:
            conservative_p = None
            minimum_effect = None
            all_positive = False

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_04",

                "scale_id":
                    str(
                        scale_id
                    ),

                "null_test_count":
                    int(
                        len(
                            frame
                        )
                    ),

                "all_nulls_estimable":
                    estimable,

                "conservative_empirical_p":
                    conservative_p,

                "minimum_effect_across_nulls":
                    minimum_effect,

                "positive_effect_against_all_nulls":
                    all_positive,
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def add_fdr_decisions(
    scale_table: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Apply BH correction across estimable temporal scales."""
    output = scale_table.copy()

    p_map = {
        str(
            row.scale_id
        ):
            float(
                row.conservative_empirical_p
            )
        for row in output.itertuples(
            index=False
        )
        if (
            bool(
                row.all_nulls_estimable
            )
            and row.conservative_empirical_p
            is not None
            and np.isfinite(
                float(
                    row.conservative_empirical_p
                )
            )
        )
    }

    corrected = benjamini_hochberg(
        p_map,
        alpha=alpha,
    )

    output[
        "fdr_q_value"
    ] = np.nan

    output[
        "fdr_significant"
    ] = False

    for index, row in output.iterrows():
        scale_id = str(
            row[
                "scale_id"
            ]
        )

        if scale_id not in corrected:
            continue

        output.loc[
            index,
            "fdr_q_value",
        ] = corrected[
            scale_id
        ][
            "q_value"
        ]

        output.loc[
            index,
            "fdr_significant",
        ] = corrected[
            scale_id
        ][
            "significant"
        ]

    output[
        "scale_support_before_robustness"
    ] = (
        output[
            "all_nulls_estimable"
        ].astype(
            bool
        )
        & output[
            "fdr_significant"
        ].astype(
            bool
        )
        & output[
            "positive_effect_against_all_nulls"
        ].astype(
            bool
        )
    )

    return output


def leave_one_transition_out(
    transitions: pd.DataFrame,
    scale_table: pd.DataFrame,
    *,
    minimum_estimable_transitions: int,
) -> pd.DataFrame:
    """
    Perform frozen leave-one-transition-out robustness audit.

    This stage does not re-run the null simulations.

    It asks whether the observed scale mean remains above each frozen
    null median after removing one observed transition at a time.
    """
    records: list[
        dict[str, Any]
    ] = []

    for scale_row in scale_table.itertuples(
        index=False
    ):
        scale_id = str(
            scale_row.scale_id
        )

        scale_transitions = transitions.loc[
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

        if (
            len(
                scale_transitions
            )
            < minimum_estimable_transitions
        ):
            continue

        null_reference = float(
            scale_row.minimum_effect_across_nulls
        ) if (
            scale_row.minimum_effect_across_nulls
            is not None
            and np.isfinite(
                float(
                    scale_row.minimum_effect_across_nulls
                )
            )
        ) else None

        observed_values = pd.to_numeric(
            scale_transitions[
                "RCS"
            ],
            errors="coerce",
        )

        observed_values = observed_values.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            observed_values
        ).all():
            fail(
                f"Scale {scale_id} contains non-finite "
                "estimable RCS values."
            )

        full_mean = float(
            np.mean(
                observed_values
            )
        )

        for local_index, row in enumerate(
            scale_transitions.itertuples(
                index=False
            )
        ):
            reduced = np.delete(
                observed_values,
                local_index,
            )

            if reduced.size == 0:
                retained = False
                reduced_mean = None

            else:
                reduced_mean = float(
                    np.mean(
                        reduced
                    )
                )

                if null_reference is None:
                    retained = False
                else:
                    # minimum_effect_across_nulls =
                    # observed full mean - largest relevant null median.
                    largest_null_median = (
                        full_mean
                        - null_reference
                    )

                    retained = bool(
                        reduced_mean
                        > largest_null_median
                    )

            records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_04",

                    "scale_id":
                        scale_id,

                    "removed_transition_id":
                        str(
                            row.transition_id
                        ),

                    "full_mean_RCS":
                        full_mean,

                    "reduced_mean_RCS":
                        reduced_mean,

                    "retains_positive_effect_against_all_nulls":
                        retained,

                    "event_information_used":
                        False,
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def attach_robustness(
    scale_table: pd.DataFrame,
    leave_one: pd.DataFrame,
    *,
    minimum_retention: float,
) -> pd.DataFrame:
    """Attach leave-one-transition-out retention fractions."""
    output = scale_table.copy()

    output[
        "robustness_fraction"
    ] = np.nan

    output[
        "robustness_pass"
    ] = False

    output[
        "single_transition_dependency"
    ] = False

    for index, row in output.iterrows():
        scale_id = str(
            row[
                "scale_id"
            ]
        )

        frame = leave_one.loc[
            leave_one[
                "scale_id"
            ]
            == scale_id
        ]

        if frame.empty:
            continue

        retention = float(
            frame[
                "retains_positive_effect_against_all_nulls"
            ]
            .astype(
                bool
            )
            .mean()
        )

        output.loc[
            index,
            "robustness_fraction",
        ] = retention

        output.loc[
            index,
            "robustness_pass",
        ] = bool(
            retention
            >= minimum_retention
        )

        output.loc[
            index,
            "single_transition_dependency",
        ] = bool(
            retention
            < 1.0
        )

    output[
        "scale_support_final"
    ] = (
        output[
            "scale_support_before_robustness"
        ].astype(
            bool
        )
        & output[
            "robustness_pass"
        ].astype(
            bool
        )
    )

    return output


def assign_scientific_outcome(
    scale_table: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
) -> tuple[
    str,
    dict[str, Any],
]:
    """Assign exactly one preregistered ID-04 scientific outcome."""
    estimable_scales = scale_table.loc[
        scale_table[
            "all_nulls_estimable"
        ]
        == True
    ].copy()

    supporting_before_robustness = scale_table.loc[
        scale_table[
            "scale_support_before_robustness"
        ]
        == True
    ].copy()

    supporting_final = scale_table.loc[
        scale_table[
            "scale_support_final"
        ]
        == True
    ].copy()

    estimable_count = int(
        len(
            estimable_scales
        )
    )

    support_before_count = int(
        len(
            supporting_before_robustness
        )
    )

    support_final_count = int(
        len(
            supporting_final
        )
    )

    if estimable_count == 0:
        outcome = (
            "NON_IDENTIFIABLE"
        )

        reason = (
            "no_temporal_scale_has_estimable_full_null_evidence"
        )

    elif (
        support_final_count
        >= contract[
            "minimum_significant_scales_full"
        ]
    ):
        outcome = (
            "CONTINUITY_SUPPORTED"
        )

        reason = (
            "minimum_cross_scale_support_and_robustness_met"
        )

    elif support_before_count >= (
        contract[
            "minimum_significant_scales_partial"
        ]
    ):
        outcome = (
            "PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE"
        )

        reason = (
            "some_scale_support_exists_but_full_cross_scale_"
            "robustness_requirement_not_met"
        )

    else:
        outcome = (
            "CONTINUITY_NOT_SUPPORTED"
        )

        reason = (
            "no_estimable_scale_meets_frozen_corrected_support_rule"
        )

    if outcome not in contract[
        "allowed_outcomes"
    ]:
        fail(
            "Outcome assignment produced an undeclared state."
        )

    details = {
        "scientific_outcome":
            outcome,

        "outcome_reason":
            reason,

        "estimable_scale_count":
            estimable_count,

        "supporting_scale_count_before_robustness":
            support_before_count,

        "supporting_scale_count_final":
            support_final_count,

        "supporting_scales":
            sorted(
                supporting_final[
                    "scale_id"
                ].astype(str)
            ),

        "event_information_used":
            False,

        "causality_established":
            False,

        "prediction_established":
            False,

        "earthquake_prediction_established":
            False,

        "minimum_identity_core_established":
            False,

        "indispensable_relations_established":
            False,

        "universal_transferability_established":
            False,
    }

    return (
        outcome,
        details,
    )


def build_robustness_summary(
    scale_table: pd.DataFrame,
    leave_one: pd.DataFrame,
    outcome_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build official robustness-stage summary."""
    return {
        "status":
            "COMPLETED",

        "scale_count":
            int(
                len(
                    scale_table
                )
            ),

        "estimable_scale_count":
            int(
                outcome_details[
                    "estimable_scale_count"
                ]
            ),

        "supporting_scale_count_before_robustness":
            int(
                outcome_details[
                    "supporting_scale_count_before_robustness"
                ]
            ),

        "supporting_scale_count_final":
            int(
                outcome_details[
                    "supporting_scale_count_final"
                ]
            ),

        "leave_one_transition_out_row_count":
            int(
                len(
                    leave_one
                )
            ),

        "scientific_outcome":
            str(
                outcome_details[
                    "scientific_outcome"
                ]
            ),

        "outcome_reason":
            str(
                outcome_details[
                    "outcome_reason"
                ]
            ),

        "event_information_used":
            False,

        "thresholds_modified_post_hoc":
            False,

        "temporal_scales_modified_post_hoc":
            False,

        "weights_modified_post_hoc":
            False,
    }


def run_robustness(
    context: ID04ExperimentContext,
) -> ID04ExperimentContext:
    """Execute frozen FDR, robustness and final outcome assignment."""
    validate_context(
        context
    )

    contract = get_contract(
        context
    )

    null_summaries = context.outputs[
        "null_summaries"
    ].copy()

    transitions = context.outputs[
        "transition_continuity"
    ].copy()

    scale_table = (
        combine_null_evidence_by_scale(
            null_summaries
        )
    )

    scale_table = add_fdr_decisions(
        scale_table,
        alpha=contract[
            "fdr"
        ],
    )

    leave_one = leave_one_transition_out(
        transitions,
        scale_table,
        minimum_estimable_transitions=contract[
            "minimum_estimable_transitions_per_scale"
        ],
    )

    scale_table = attach_robustness(
        scale_table,
        leave_one,
        minimum_retention=contract[
            "minimum_retention"
        ],
    )

    (
        scientific_outcome,
        outcome_details,
    ) = assign_scientific_outcome(
        scale_table,
        contract=contract,
    )

    summary = build_robustness_summary(
        scale_table,
        leave_one,
        outcome_details,
    )

    context.register_output(
        "multiscale_continuity",
        scale_table,
    )

    context.register_output(
        "leave_one_transition_out",
        leave_one,
    )

    context.register_output(
        "scientific_outcome_details",
        outcome_details,
    )

    context.register_output(
        "robustness_summary",
        summary,
    )

    context.register_runtime(
        "scientific_outcome",
        scientific_outcome,
    )

    context.register_runtime(
        "robustness_status",
        "COMPLETED",
    )

    return context
