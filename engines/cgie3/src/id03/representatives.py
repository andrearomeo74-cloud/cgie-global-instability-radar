"""
CGIE3-ID-03 representative-candidate selection.

This stage combines the frozen evidence from:

- feature dependency audit;
- multiscale audit;
- overlap audit;
- null-control audit;
- conditional redundancy audit;
- relational-family audit.

It assigns exactly one final ID-03 state to every primary relation
and selects at most one family-representative candidate for each
reproducible family.

It does not:

- modify ID-02 classifications;
- call any relation primary, essential or indispensable;
- infer causality;
- establish predictive capability;
- establish earthquake prediction.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class RepresentativeSelectionError(ValueError):
    """Raised when representative selection violates the contract."""


def fail(message: str) -> None:
    """Raise a normalized representative-selection error."""
    raise RepresentativeSelectionError(
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
        raise RepresentativeSelectionError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
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


def require_dataframe(
    value: Any,
    field_name: str,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Require a non-empty DataFrame and optional exact row count."""
    if not isinstance(
        value,
        pd.DataFrame,
    ):
        fail(
            f"{field_name} must be a pandas DataFrame."
        )

    if value.empty:
        fail(
            f"{field_name} must not be empty."
        )

    if (
        expected_rows is not None
        and len(
            value
        )
        != expected_rows
    ):
        fail(
            f"{field_name} must contain "
            f"{expected_rows} rows; observed "
            f"{len(value)}."
        )

    return value


def validate_context(
    context: ID03ExperimentContext,
) -> None:
    """Validate representative-selection prerequisites."""
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
            "family_audit_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Family audit must complete before "
            "representative selection."
        )

    required_outputs = {
        "primary_population",
        "relation_dependencies",
        "multiscale_relations",
        "overlap_sensitivity",
        "null_controls",
        "conditional_redundancy",
        "relation_families",
        "family_membership",
    }

    missing_outputs = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing_outputs:
        fail(
            "Representative-selection prerequisites "
            "are missing: "
            + ", ".join(
                missing_outputs
            )
        )


def get_selection_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate frozen selection and state rules."""
    selection = require_mapping(
        context.configuration.get(
            "representative_selection"
        ),
        "representative_selection",
    )

    if selection.get(
        "enabled"
    ) is not True:
        fail(
            "Representative selection must remain enabled."
        )

    mandatory = require_mapping(
        selection.get(
            "mandatory_conditions"
        ),
        (
            "representative_selection."
            "mandatory_conditions"
        ),
    )

    allowed_id02_statuses = require_string_list(
        mandatory.get(
            "id02_status_in"
        ),
        (
            "representative_selection."
            "mandatory_conditions.id02_status_in"
        ),
    )

    if set(
        allowed_id02_statuses
    ) != {
        "eligible",
        "candidate",
    }:
        fail(
            "Representative selection must accept only "
            "eligible and candidate ID-02 statuses."
        )

    tie_break_order = require_string_list(
        selection.get(
            "tie_break_order"
        ),
        "representative_selection.tie_break_order",
    )

    expected_tie_break_order = (
        "supported_scale_count_descending",
        "overlap_robustness_descending",
        "primary_null_margin_descending",
        "residual_information_descending",
        "id02_persistence_descending",
        "bootstrap_ci_width_ascending",
        "relation_id_lexicographic",
    )

    if tie_break_order != expected_tie_break_order:
        fail(
            "Representative tie-break order differs "
            "from the frozen contract."
        )

    state_contract = require_mapping(
        context.configuration.get(
            "id03_states"
        ),
        "id03_states",
    )

    allowed_states = require_string_list(
        state_contract.get(
            "allowed"
        ),
        "id03_states.allowed",
    )

    precedence = require_string_list(
        state_contract.get(
            "precedence"
        ),
        "id03_states.precedence",
    )

    expected_states = {
        "family_representative_candidate",
        "supporting_relation",
        "definitionally_constrained",
        "redundant_relation",
        "overlap_sensitive",
        "scale_inconsistent",
        "insufficient_evidence",
    }

    if set(
        allowed_states
    ) != expected_states:
        fail(
            "Allowed ID-03 states differ from "
            "the frozen contract."
        )

    expected_precedence = (
        "scale_inconsistent",
        "overlap_sensitive",
        "definitionally_constrained",
        "redundant_relation",
        "family_representative_candidate",
        "supporting_relation",
        "insufficient_evidence",
    )

    if precedence != expected_precedence:
        fail(
            "ID-03 state precedence differs from "
            "the frozen contract."
        )

    unresolved = require_mapping(
        selection.get(
            "unresolved_residual_information"
        ),
        (
            "representative_selection."
            "unresolved_residual_information"
        ),
    )

    if (
        unresolved.get(
            "allow_representative_selection"
        )
        is not False
    ):
        fail(
            "Relations with unresolved residual information "
            "must not become representatives."
        )

    return {
        "allowed_id02_statuses":
            set(
                allowed_id02_statuses
            ),

        "family_membership_required":
            bool(
                mandatory.get(
                    "family_membership_required"
                )
            ),

        "directly_derived_prohibited":
            bool(
                mandatory.get(
                    "directly_derived_prohibited"
                )
            ),

        "strongly_overlap_sensitive_prohibited":
            bool(
                mandatory.get(
                    "strongly_overlap_sensitive_prohibited"
                )
            ),

        "sign_preserved_under_reduced_overlap":
            bool(
                mandatory.get(
                    "sign_preserved_under_reduced_overlap"
                )
            ),

        "minimum_supported_scales":
            require_positive_integer(
                mandatory.get(
                    "minimum_supported_scales"
                ),
                (
                    "representative_selection."
                    "mandatory_conditions."
                    "minimum_supported_scales"
                ),
            ),

        "minimum_primary_nulls_exceeded":
            require_positive_integer(
                mandatory.get(
                    "minimum_primary_nulls_exceeded"
                ),
                (
                    "representative_selection."
                    "mandatory_conditions."
                    "minimum_primary_nulls_exceeded"
                ),
            ),

        "residual_information_required":
            bool(
                mandatory.get(
                    "residual_information_required"
                )
            ),

        "fully_redundant_prohibited":
            bool(
                mandatory.get(
                    "fully_redundant_prohibited"
                )
            ),

        "maximum_representatives_per_family":
            require_positive_integer(
                selection.get(
                    "maximum_representatives_per_family"
                ),
                (
                    "representative_selection."
                    "maximum_representatives_per_family"
                ),
            ),

        "allowed_states":
            set(
                allowed_states
            ),

        "precedence":
            precedence,
    }


def safe_bool(
    value: Any,
) -> bool:
    """Convert missing values to False and valid values to bool."""
    if value is None:
        return False

    try:
        if pd.isna(
            value
        ):
            return False
    except (
        TypeError,
        ValueError,
    ):
        pass

    return bool(
        value
    )


def safe_float(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    """Convert a finite value to float or return a default."""
    try:
        normalized = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return float(
            default
        )

    if not np.isfinite(
        normalized
    ):
        return float(
            default
        )

    return normalized


def canonical_pair_id(
    source_id: str,
    target_id: str,
) -> str:
    """Return one canonical unordered pair ID."""
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


def overlap_rank(
    overlap_class: Any,
) -> int:
    """Map overlap classes to deterministic descending ranks."""
    mapping = {
        "overlap_robust":
            2,

        "moderately_overlap_sensitive":
            1,

        "strongly_overlap_sensitive":
            0,

        "non_estimable_non_overlapping":
            0,

        "inconclusive_overlap_audit":
            0,
    }

    return int(
        mapping.get(
            str(
                overlap_class
            ),
            0,
        )
    )


def residual_information_rank(
    redundancy_status: Any,
) -> int:
    """Map residual-information classes to deterministic ranks."""
    mapping = {
        "retains_residual_information":
            2,

        "partially_redundant":
            1,

        "fully_redundant":
            0,

        "conditional_test_inconclusive":
            0,
    }

    return int(
        mapping.get(
            str(
                redundancy_status
            ),
            0,
        )
    )


def build_selection_evidence(
    context: ID03ExperimentContext,
) -> pd.DataFrame:
    """Merge all frozen evidence for the 74 primary relations."""
    primary = require_dataframe(
        context.outputs[
            "primary_population"
        ],
        "primary_population",
        expected_rows=74,
    ).copy()

    dependencies = require_dataframe(
        context.outputs[
            "relation_dependencies"
        ],
        "relation_dependencies",
        expected_rows=136,
    )

    overlap = require_dataframe(
        context.outputs[
            "overlap_sensitivity"
        ],
        "overlap_sensitivity",
        expected_rows=74,
    )

    null_controls = require_dataframe(
        context.outputs[
            "null_controls"
        ],
        "null_controls",
        expected_rows=74,
    )

    redundancy = require_dataframe(
        context.outputs[
            "conditional_redundancy"
        ],
        "conditional_redundancy",
        expected_rows=74,
    )

    membership = require_dataframe(
        context.outputs[
            "family_membership"
        ],
        "family_membership",
        expected_rows=74,
    )

    families = require_dataframe(
        context.outputs[
            "relation_families"
        ],
        "relation_families",
    )

    multiscale = require_dataframe(
        context.outputs[
            "multiscale_relations"
        ],
        "multiscale_relations",
    )

    keys = [
        "window_id",
        "relation_id",
    ]

    evidence = primary.copy()

    evidence = evidence.merge(
        dependencies[
            [
                *keys,
                "dependency_status",
                "directly_derived_flag",
                "definitionally_constrained_flag",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    evidence = evidence.merge(
        overlap[
            [
                *keys,
                "overlap_class",
                "strongly_overlap_sensitive_flag",
                "nonoverlap_sign_preserved",
                "nonoverlap_estimable",
                "nonoverlap_absolute_strength_change",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    evidence = evidence.merge(
        null_controls[
            [
                *keys,
                "null_outcome",
                "primary_null_margin",
                "primary_nulls_passed_alpha_0_05",
                "minimum_one_primary_null_exceeded",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    evidence = evidence.merge(
        redundancy[
            [
                *keys,
                "redundancy_status",
                "retains_residual_information",
                "fully_redundant_flag",
                "partial_spearman_strength",
                "residual_spearman_strength",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    evidence = evidence.merge(
        membership[
            [
                *keys,
                "family_id",
                "family_reproducible",
                "edge_weight",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    family_reproducibility = families[
        [
            "family_id",
            "reproducible_family",
            "family_stability",
        ]
    ].copy()

    family_reproducibility = (
        family_reproducibility.rename(
            columns={
                "reproducible_family":
                    "family_table_reproducible",
            }
        )
    )

    evidence = evidence.merge(
        family_reproducibility,
        on="family_id",
        how="left",
        validate="many_to_one",
    )

    evidence[
        "pair_id"
    ] = [
        canonical_pair_id(
            source_id,
            target_id,
        )
        for source_id, target_id
        in zip(
            evidence[
                "source_id"
            ],
            evidence[
                "target_id"
            ],
        )
    ]

    evidence = evidence.merge(
        multiscale[
            [
                "pair_id",
                "supported_scale_count",
                "multiscale_class",
                "sign_consistent",
                "scale_inconsistent",
                "median_supported_bootstrap_ci_width",
            ]
        ],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )

    if len(
        evidence
    ) != 74:
        fail(
            "Representative evidence must contain "
            f"74 rows; observed {len(evidence)}."
        )

    if evidence.duplicated(
        subset=keys,
        keep=False,
    ).any():
        fail(
            "Representative evidence contains duplicate "
            "window-relation keys."
        )

    return evidence


def evaluate_mandatory_conditions(
    evidence: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Evaluate all representative conditions without selecting yet."""
    output = evidence.copy()

    output[
        "condition_id02_status_allowed"
    ] = output[
        "classification_status"
    ].astype(str).isin(
        contract[
            "allowed_id02_statuses"
        ]
    )

    output[
        "condition_family_membership"
    ] = (
        output[
            "family_id"
        ].notna()
        if contract[
            "family_membership_required"
        ]
        else True
    )

    output[
        "condition_family_reproducible"
    ] = (
        output[
            "family_reproducible"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        & output[
            "family_table_reproducible"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    output[
        "condition_not_directly_derived"
    ] = (
        ~output[
            "directly_derived_flag"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        if contract[
            "directly_derived_prohibited"
        ]
        else True
    )

    output[
        "condition_not_strongly_overlap_sensitive"
    ] = (
        ~output[
            "strongly_overlap_sensitive_flag"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        if contract[
            "strongly_overlap_sensitive_prohibited"
        ]
        else True
    )

    output[
        "condition_reduced_overlap_sign_preserved"
    ] = (
        output[
            "nonoverlap_sign_preserved"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        if contract[
            "sign_preserved_under_reduced_overlap"
        ]
        else True
    )

    output[
        "condition_minimum_supported_scales"
    ] = (
        pd.to_numeric(
            output[
                "supported_scale_count"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        >= contract[
            "minimum_supported_scales"
        ]
    )

    output[
        "condition_minimum_primary_nulls_exceeded"
    ] = (
        pd.to_numeric(
            output[
                "primary_nulls_passed_alpha_0_05"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        >= contract[
            "minimum_primary_nulls_exceeded"
        ]
    )

    output[
        "condition_residual_information"
    ] = (
        output[
            "retains_residual_information"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        if contract[
            "residual_information_required"
        ]
        else True
    )

    output[
        "condition_not_fully_redundant"
    ] = (
        ~output[
            "fully_redundant_flag"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        if contract[
            "fully_redundant_prohibited"
        ]
        else True
    )

    mandatory_columns = [
        "condition_id02_status_allowed",
        "condition_family_membership",
        "condition_family_reproducible",
        "condition_not_directly_derived",
        "condition_not_strongly_overlap_sensitive",
        "condition_reduced_overlap_sign_preserved",
        "condition_minimum_supported_scales",
        "condition_minimum_primary_nulls_exceeded",
        "condition_residual_information",
        "condition_not_fully_redundant",
    ]

    output[
        "representative_condition_count_met"
    ] = output[
        mandatory_columns
    ].sum(
        axis=1
    ).astype(
        int
    )

    output[
        "representative_condition_count_total"
    ] = int(
        len(
            mandatory_columns
        )
    )

    output[
        "all_representative_conditions_met"
    ] = output[
        mandatory_columns
    ].all(
        axis=1
    )

    output[
        "failed_representative_conditions"
    ] = [
        "|".join(
            column.replace(
                "condition_",
                "",
                1,
            )
            for column in mandatory_columns
            if not bool(
                row[
                    column
                ]
            )
        )
        for _, row in output.iterrows()
    ]

    output[
        "tie_break_supported_scale_count"
    ] = (
        pd.to_numeric(
            output[
                "supported_scale_count"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    output[
        "tie_break_overlap_robustness"
    ] = output[
        "overlap_class"
    ].map(
        overlap_rank
    )

    output[
        "tie_break_primary_null_margin"
    ] = pd.to_numeric(
        output[
            "primary_null_margin"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "tie_break_residual_information"
    ] = output[
        "redundancy_status"
    ].map(
        residual_information_rank
    )

    output[
        "tie_break_id02_persistence"
    ] = pd.to_numeric(
        output[
            "persistence"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "tie_break_bootstrap_ci_width"
    ] = pd.to_numeric(
        output[
            "bootstrap_ci_width"
        ],
        errors="coerce",
    ).fillna(
        np.inf
    )

    return output


def select_family_representatives(
    evidence: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Select at most the frozen maximum per reproducible family."""
    output = evidence.copy()

    output[
        "selected_family_representative"
    ] = False

    output[
        "family_selection_rank"
    ] = pd.Series(
        [pd.NA] * len(
            output
        ),
        dtype="Int64",
    )

    output[
        "family_selection_reason"
    ] = "not_selected"

    eligible = output.loc[
        output[
            "all_representative_conditions_met"
        ]
        == True
    ].copy()

    for family_id, family_frame in eligible.groupby(
        "family_id",
        sort=True,
    ):
        ranked = family_frame.sort_values(
            by=[
                "tie_break_supported_scale_count",
                "tie_break_overlap_robustness",
                "tie_break_primary_null_margin",
                "tie_break_residual_information",
                "tie_break_id02_persistence",
                "tie_break_bootstrap_ci_width",
                "relation_id",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )

        ranked_indices = list(
            ranked.index
        )

        for rank_position, index in enumerate(
            ranked_indices,
            start=1,
        ):
            output.loc[
                index,
                "family_selection_rank",
            ] = rank_position

        selected_indices = ranked_indices[
            : contract[
                "maximum_representatives_per_family"
            ]
        ]

        output.loc[
            selected_indices,
            "selected_family_representative",
        ] = True

        output.loc[
            selected_indices,
            "family_selection_reason",
        ] = (
            "highest_frozen_tie_break_rank_"
            "within_reproducible_family"
        )

        non_selected_indices = ranked_indices[
            contract[
                "maximum_representatives_per_family"
            ]:
        ]

        output.loc[
            non_selected_indices,
            "family_selection_reason",
        ] = (
            "mandatory_conditions_met_but_lower_"
            "family_tie_break_rank"
        )

    return output


def assign_id03_state(
    row: Any,
) -> tuple[str, str]:
    """Apply the frozen ID-03 state precedence."""
    if safe_bool(
        row.scale_inconsistent
    ):
        return (
            "scale_inconsistent",
            "multiscale_sign_or_strong_scale_inconsistency",
        )

    if str(
        row.overlap_class
    ) == "strongly_overlap_sensitive":
        return (
            "overlap_sensitive",
            "strongly_overlap_sensitive",
        )

    if str(
        row.dependency_status
    ) == "directly_derived":
        return (
            "definitionally_constrained",
            "directly_derived_dependency",
        )

    if str(
        row.redundancy_status
    ) == "fully_redundant":
        return (
            "redundant_relation",
            "fully_redundant_after_conditioning",
        )

    if safe_bool(
        row.selected_family_representative
    ):
        return (
            "family_representative_candidate",
            "all_mandatory_conditions_and_family_rank_met",
        )

    if (
        row.family_id is not None
        and not pd.isna(
            row.family_id
        )
        and safe_bool(
            row.family_reproducible
        )
    ):
        return (
            "supporting_relation",
            "member_of_reproducible_family",
        )

    return (
        "insufficient_evidence",
        "no_higher_precedence_state_applies",
    )


def classify_id03_states(
    evidence: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Assign exactly one final state to every primary relation."""
    records: list[dict[str, str]] = []

    for row in evidence.itertuples(
        index=False
    ):
        state, reason = assign_id03_state(
            row
        )

        records.append(
            {
                "id03_state":
                    state,

                "id03_state_reason":
                    reason,
            }
        )

    state_frame = pd.DataFrame.from_records(
        records
    )

    output = pd.concat(
        [
            evidence.reset_index(
                drop=True
            ),
            state_frame.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    observed_states = set(
        output[
            "id03_state"
        ].astype(str)
    )

    unexpected_states = sorted(
        observed_states
        - contract[
            "allowed_states"
        ]
    )

    if unexpected_states:
        fail(
            "Unexpected ID-03 states: "
            + ", ".join(
                unexpected_states
            )
        )

    if len(
        output
    ) != 74:
        fail(
            "ID-03 final relation states must contain "
            f"74 rows; observed {len(output)}."
        )

    if output[
        [
            "window_id",
            "relation_id",
        ]
    ].duplicated(
        keep=False
    ).any():
        fail(
            "ID-03 final states contain duplicate "
            "window-relation keys."
        )

    representative_counts = (
        output.loc[
            output[
                "id03_state"
            ]
            == "family_representative_candidate"
        ]
        .groupby(
            "family_id",
            sort=True,
        )
        .size()
    )

    if (
        representative_counts
        > contract[
            "maximum_representatives_per_family"
        ]
    ).any():
        fail(
            "A family exceeds the frozen maximum "
            "representative count."
        )

    output[
        "id02_status_modified"
    ] = False

    output[
        "primary_relation_established"
    ] = False

    output[
        "indispensable_relation_established"
    ] = False

    output[
        "causality_established"
    ] = False

    output[
        "predictive_capability_established"
    ] = False

    output[
        "earthquake_prediction_established"
    ] = False

    return output.sort_values(
        by=[
            "family_id",
            "source_id",
            "target_id",
            "window_id",
            "relation_id",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(
        drop=True
    )


def build_representative_candidates(
    final_states: pd.DataFrame,
) -> pd.DataFrame:
    """Build the official representative-candidate table."""
    representatives = final_states.loc[
        final_states[
            "id03_state"
        ]
        == "family_representative_candidate"
    ].copy()

    selected_columns = [
        "experiment_id",
        "family_id",
        "family_selection_rank",
        "window_id",
        "relation_id",
        "pair_id",
        "source_id",
        "target_id",
        "classification_status",
        "id03_state",
        "id03_state_reason",
        "strength",
        "absolute_strength",
        "persistence",
        "bootstrap_ci_width",
        "supported_scale_count",
        "multiscale_class",
        "overlap_class",
        "nonoverlap_sign_preserved",
        "null_outcome",
        "primary_null_margin",
        "primary_nulls_passed_alpha_0_05",
        "dependency_status",
        "redundancy_status",
        "partial_spearman_strength",
        "residual_spearman_strength",
        "family_stability",
        "edge_weight",
        "failed_representative_conditions",
        "primary_relation_established",
        "indispensable_relation_established",
        "causality_established",
        "predictive_capability_established",
        "earthquake_prediction_established",
    ]

    missing_columns = sorted(
        set(
            selected_columns
        )
        - set(
            representatives.columns
        )
    )

    if missing_columns:
        fail(
            "Representative output is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    return representatives.loc[
        :,
        selected_columns,
    ].sort_values(
        by=[
            "family_id",
            "family_selection_rank",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_selection_summary(
    final_states: pd.DataFrame,
    representatives: pd.DataFrame,
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Build the final ID-03 state and representative summary."""
    state_counts = (
        final_states[
            "id03_state"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    families = require_dataframe(
        context.outputs[
            "relation_families"
        ],
        "relation_families",
    )

    reproducible_family_count = int(
        families[
            "reproducible_family"
        ].sum()
    )

    representative_count = int(
        len(
            representatives
        )
    )

    if (
        reproducible_family_count >= 1
        and representative_count >= 1
    ):
        scientific_outcome = (
            "families_and_representatives_identified"
        )

        scientific_positive_result = True

    elif (
        reproducible_family_count >= 1
        and representative_count == 0
    ):
        scientific_outcome = (
            "families_identified_without_representatives"
        )

        scientific_positive_result = False

    elif (
        int(
            (
                final_states[
                    "id03_state"
                ]
                == "definitionally_constrained"
            ).sum()
        )
        == len(
            final_states
        )
    ):
        scientific_outcome = (
            "all_relations_definitionally_constrained"
        )

        scientific_positive_result = False

    elif (
        int(
            (
                final_states[
                    "id03_state"
                ]
                == "overlap_sensitive"
            ).sum()
        )
        == len(
            final_states
        )
    ):
        scientific_outcome = (
            "all_relations_overlap_sensitive"
        )

        scientific_positive_result = False

    elif (
        context.outputs[
            "family_audit_summary"
        ][
            "reproducible_family_count"
        ]
        == 0
    ):
        scientific_outcome = (
            "no_reproducible_families"
        )

        scientific_positive_result = False

    else:
        scientific_outcome = (
            "insufficient_evidence"
        )

        scientific_positive_result = False

    by_family: dict[str, Any] = {}

    for family_id, frame in final_states.groupby(
        "family_id",
        sort=True,
        dropna=False,
    ):
        normalized_family_id = (
            str(
                family_id
            )
            if not pd.isna(
                family_id
            )
            else "UNASSIGNED"
        )

        counts = (
            frame[
                "id03_state"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        by_family[
            normalized_family_id
        ] = {
            "relation_count":
                int(
                    len(
                        frame
                    )
                ),

            "state_counts": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in counts.items()
            },

            "representative_candidate_count":
                int(
                    (
                        frame[
                            "id03_state"
                        ]
                        == "family_representative_candidate"
                    ).sum()
                ),
        }

    return {
        "status":
            "COMPLETED",

        "technical_success":
            True,

        "primary_relation_count":
            int(
                len(
                    final_states
                )
            ),

        "id03_state_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in state_counts.items()
        },

        "reproducible_family_count":
            reproducible_family_count,

        "representative_candidate_count":
            representative_count,

        "by_family":
            by_family,

        "scientific_outcome":
            scientific_outcome,

        "scientific_positive_result":
            scientific_positive_result,

        "id02_statuses_modified":
            False,

        "primary_relations_established":
            False,

        "indispensable_relations_established":
            False,

        "minimum_identity_core_established":
            False,

        "causality_established":
            False,

        "predictive_capability_established":
            False,

        "earthquake_prediction_established":
            False,

        "universal_transferability_established":
            False,
    }


def select_representatives(
    context: ID03ExperimentContext,
) -> ID03ExperimentContext:
    """Execute frozen state assignment and representative selection."""
    validate_context(
        context
    )

    contract = get_selection_contract(
        context
    )

    evidence = build_selection_evidence(
        context
    )

    conditioned = evaluate_mandatory_conditions(
        evidence,
        contract,
    )

    selected = select_family_representatives(
        conditioned,
        contract,
    )

    final_states = classify_id03_states(
        selected,
        contract,
    )

    representatives = (
        build_representative_candidates(
            final_states
        )
    )

    summary = build_selection_summary(
        final_states,
        representatives,
        context,
    )

    context.register_output(
        "representative_selection_evidence",
        conditioned,
    )

    context.register_output(
        "id03_relation_states",
        final_states,
    )

    context.register_output(
        "representative_candidates",
        representatives,
    )

    context.register_output(
        "representative_selection_summary",
        summary,
    )

    context.register_runtime(
        "representative_selection_status",
        "COMPLETED",
    )

    return context
           
