"""
CGIE3-ID-03 feature-dependency audit.

This stage assigns frozen provenance information to every declared
feature and one pairwise dependency class to every ID-02 relation.

It does not:

- recompute correlations;
- modify ID-02 classifications;
- infer causality;
- select relational families;
- select representative candidates.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class DependencyAuditError(ValueError):
    """Raised when the dependency audit violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a normalized dependency-audit error."""
    raise DependencyAuditError(
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


def require_string(
    value: Any,
    field_name: str,
) -> str:
    """Require one non-empty string."""
    if not isinstance(
        value,
        str,
    ):
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
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Require an ordered list of unique strings."""
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

    if (
        not allow_empty
        and not normalized
    ):
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


def validate_context(
    context: ID03ExperimentContext,
) -> None:
    """Validate dependency-audit prerequisites."""
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
            "loader_status"
        )
        != "COMPLETED"
    ):
        fail(
            "ID-03 loader must complete before "
            "the dependency audit."
        )

    if not isinstance(
        context.relation_classification,
        pd.DataFrame,
    ):
        fail(
            "relation_classification must be "
            "a pandas DataFrame."
        )

    if len(
        context.relation_classification
    ) != 136:
        fail(
            "Dependency audit expects 136 "
            "ID-02 relations."
        )


def get_dependency_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract the frozen feature-dependency contract."""
    registry = require_mapping(
        context.configuration.get(
            "feature_dependency_registry"
        ),
        "feature_dependency_registry",
    )

    if (
        registry.get("status")
        != "FROZEN"
    ):
        fail(
            "Feature dependency registry must "
            "have status FROZEN."
        )

    allowed_provenance_classes = set(
        require_string_list(
            registry.get(
                "allowed_provenance_classes"
            ),
            (
                "feature_dependency_registry."
                "allowed_provenance_classes"
            ),
        )
    )

    allowed_dependency_classes = set(
        require_string_list(
            registry.get(
                "allowed_pair_dependency_classes"
            ),
            (
                "feature_dependency_registry."
                "allowed_pair_dependency_classes"
            ),
        )
    )

    feature_lineage = require_mapping(
        registry.get(
            "feature_lineage"
        ),
        "feature_dependency_registry.feature_lineage",
    )

    pair_rules = require_mapping(
        registry.get(
            "pair_dependency_rules"
        ),
        (
            "feature_dependency_registry."
            "pair_dependency_rules"
        ),
    )

    declared_components = set(
        context.identity.component_ids
    )

    lineage_components = set(
        feature_lineage
    )

    if (
        lineage_components
        != declared_components
    ):
        missing = sorted(
            declared_components
            - lineage_components
        )

        unexpected = sorted(
            lineage_components
            - declared_components
        )

        fail(
            "Feature-lineage registry differs from "
            "the identity declaration. "
            f"Missing: {missing}. "
            f"Unexpected: {unexpected}."
        )

    return {
        "registry":
            registry,

        "allowed_provenance_classes":
            allowed_provenance_classes,

        "allowed_dependency_classes":
            allowed_dependency_classes,

        "feature_lineage":
            feature_lineage,

        "pair_rules":
            pair_rules,
    }


def normalize_feature_lineage(
    feature_id: str,
    payload: Mapping[str, Any],
    *,
    allowed_provenance_classes: set[str],
) -> dict[str, Any]:
    """Normalize one frozen feature-lineage declaration."""
    provenance_class = require_string(
        payload.get(
            "provenance_class"
        ),
        f"feature_lineage.{feature_id}.provenance_class",
    )

    if (
        provenance_class
        not in allowed_provenance_classes
    ):
        fail(
            f"Feature {feature_id} uses an invalid "
            f"provenance class: {provenance_class}"
        )

    source_family = require_string(
        payload.get(
            "source_family"
        ),
        f"feature_lineage.{feature_id}.source_family",
    )

    source_variables = require_string_list(
        payload.get(
            "source_variables"
        ),
        f"feature_lineage.{feature_id}.source_variables",
    )

    transformation = require_string(
        payload.get(
            "transformation"
        ),
        f"feature_lineage.{feature_id}.transformation",
    )

    directly_derived_from = (
        require_string_list(
            payload.get(
                "directly_derived_from",
                [],
            ),
            (
                f"feature_lineage.{feature_id}."
                "directly_derived_from"
            ),
            allow_empty=True,
        )
    )

    potential_direct_dependency_with = (
        require_string_list(
            payload.get(
                "potential_direct_dependency_with",
                [],
            ),
            (
                f"feature_lineage.{feature_id}."
                "potential_direct_dependency_with"
            ),
            allow_empty=True,
        )
    )

    return {
        "feature_id":
            feature_id,

        "provenance_class":
            provenance_class,

        "source_family":
            source_family,

        "source_variables":
            source_variables,

        "transformation":
            transformation,

        "directly_derived_from":
            directly_derived_from,

        "potential_direct_dependency_with":
            potential_direct_dependency_with,
    }


def build_feature_dependency_table(
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Create the official feature-lineage audit table."""
    feature_lineage = require_mapping(
        contract[
            "feature_lineage"
        ],
        "feature_lineage",
    )

    records: list[dict[str, Any]] = []

    for feature_id in sorted(
        feature_lineage
    ):
        normalized = normalize_feature_lineage(
            feature_id,
            require_mapping(
                feature_lineage[
                    feature_id
                ],
                f"feature_lineage.{feature_id}",
            ),
            allowed_provenance_classes=set(
                contract[
                    "allowed_provenance_classes"
                ]
            ),
        )

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_03",

                "feature_id":
                    normalized[
                        "feature_id"
                    ],

                "provenance_class":
                    normalized[
                        "provenance_class"
                    ],

                "source_family":
                    normalized[
                        "source_family"
                    ],

                "source_variables":
                    "|".join(
                        normalized[
                            "source_variables"
                        ]
                    ),

                "transformation":
                    normalized[
                        "transformation"
                    ],

                "directly_derived_from":
                    "|".join(
                        normalized[
                            "directly_derived_from"
                        ]
                    ),

                "potential_direct_dependency_with":
                    "|".join(
                        normalized[
                            "potential_direct_dependency_with"
                        ]
                    ),

                "registry_status":
                    "FROZEN",
            }
        )

    output = pd.DataFrame.from_records(
        records
    )

    if len(output) != 9:
        fail(
            "Feature dependency table must contain "
            f"9 features; observed {len(output)}."
        )

    return output.sort_values(
        by=[
            "feature_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def normalized_lineage_by_feature(
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create normalized lineage records indexed by feature ID."""
    feature_lineage = require_mapping(
        contract[
            "feature_lineage"
        ],
        "feature_lineage",
    )

    return {
        feature_id:
            normalize_feature_lineage(
                feature_id,
                require_mapping(
                    payload,
                    f"feature_lineage.{feature_id}",
                ),
                allowed_provenance_classes=set(
                    contract[
                        "allowed_provenance_classes"
                    ]
                ),
            )
        for feature_id, payload
        in feature_lineage.items()
    }


def shares_any_value(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    """Return whether two string collections overlap."""
    return bool(
        set(left)
        & set(right)
    )


def classify_pair_dependency(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[str, str]:
    """
    Assign exactly one frozen dependency class to a feature pair.

    Precedence:

    1. directly derived;
    2. explicit potential direct dependency;
    3. shared temporal source;
    4. shared spatial source;
    5. shared event source;
    6. independent source;
    7. unknown dependency.
    """
    source_id = str(
        source[
            "feature_id"
        ]
    )

    target_id = str(
        target[
            "feature_id"
        ]
    )

    source_direct = set(
        source[
            "directly_derived_from"
        ]
    )

    target_direct = set(
        target[
            "directly_derived_from"
        ]
    )

    if (
        source_id in target_direct
        or target_id in source_direct
    ):
        return (
            "directly_derived",
            "one_feature_explicitly_derived_from_other",
        )

    source_potential = set(
        source[
            "potential_direct_dependency_with"
        ]
    )

    target_potential = set(
        target[
            "potential_direct_dependency_with"
        ]
    )

    if (
        target_id in source_potential
        or source_id in target_potential
    ):
        return (
            "partially_derived",
            "explicit_potential_direct_dependency",
        )

    source_family = str(
        source[
            "source_family"
        ]
    )

    target_family = str(
        target[
            "source_family"
        ]
    )

    if (
        source_family
        == "event_timing"
        and target_family
        == "event_timing"
    ):
        return (
            "shared_temporal_source",
            "same_temporal_source_family",
        )

    if (
        source_family
        == "spatial_location"
        and target_family
        == "spatial_location"
    ):
        return (
            "shared_spatial_source",
            "same_spatial_source_family",
        )

    if shares_any_value(
        tuple(
            source[
                "source_variables"
            ]
        ),
        tuple(
            target[
                "source_variables"
            ]
        ),
    ):
        return (
            "shared_event_source",
            "same_direct_source",
        )

    if (
        source_family
        and target_family
        and source_family
        != target_family
    ):
        return (
            "independent_source",
            "different_declared_source_families",
        )

    return (
        "unknown_dependency",
        "unresolved",
    )


def build_relation_dependency_table(
    context: ID03ExperimentContext,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Assign dependency classes to all 136 ID-02 relations."""
    lineage = normalized_lineage_by_feature(
        contract
    )

    classification = (
        context.relation_classification.copy()
    )

    required_columns = {
        "window_id",
        "relation_id",
        "source_id",
        "target_id",
        "classification_status",
    }

    missing_columns = sorted(
        required_columns
        - set(
            classification.columns
        )
    )

    if missing_columns:
        fail(
            "ID-02 classification is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    records: list[dict[str, Any]] = []

    allowed_dependency_classes = set(
        contract[
            "allowed_dependency_classes"
        ]
    )

    for row in classification.itertuples(
        index=False
    ):
        source_id = str(
            row.source_id
        )

        target_id = str(
            row.target_id
        )

        if source_id not in lineage:
            fail(
                "Missing feature lineage for source: "
                f"{source_id}"
            )

        if target_id not in lineage:
            fail(
                "Missing feature lineage for target: "
                f"{target_id}"
            )

        dependency_status, rule_id = (
            classify_pair_dependency(
                lineage[
                    source_id
                ],
                lineage[
                    target_id
                ],
            )
        )

        if (
            dependency_status
            not in allowed_dependency_classes
        ):
            fail(
                "Invalid dependency status generated: "
                f"{dependency_status}"
            )

        source_variables = set(
            lineage[
                source_id
            ][
                "source_variables"
            ]
        )

        target_variables = set(
            lineage[
                target_id
            ][
                "source_variables"
            ]
        )

        shared_source_variables = sorted(
            source_variables
            & target_variables
        )

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_03",

                "window_id":
                    str(
                        row.window_id
                    ),

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

                "source_provenance_class":
                    lineage[
                        source_id
                    ][
                        "provenance_class"
                    ],

                "target_provenance_class":
                    lineage[
                        target_id
                    ][
                        "provenance_class"
                    ],

                "source_family":
                    lineage[
                        source_id
                    ][
                        "source_family"
                    ],

                "target_family":
                    lineage[
                        target_id
                    ][
                        "source_family"
                    ],

                "shared_source_variables":
                    "|".join(
                        shared_source_variables
                    ),

                "dependency_status":
                    dependency_status,

                "dependency_rule":
                    rule_id,

                "directly_derived_flag":
                    (
                        dependency_status
                        == "directly_derived"
                    ),

                "definitionally_constrained_flag":
                    (
                        dependency_status
                        in {
                            "directly_derived",
                            "partially_derived",
                            "shared_temporal_source",
                            "shared_spatial_source",
                            "shared_event_source",
                        }
                    ),

                "id02_status_modified":
                    False,
            }
        )

    output = pd.DataFrame.from_records(
        records
    )

    if len(output) != 136:
        fail(
            "Relation dependency audit must contain "
            f"136 rows; observed {len(output)}."
        )

    duplicate_mask = output.duplicated(
        subset=[
            "window_id",
            "relation_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        fail(
            "Relation dependency audit contains "
            "duplicate window-relation keys."
        )

    return output.sort_values(
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


def build_dependency_summary(
    feature_dependencies: pd.DataFrame,
    relation_dependencies: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive dependency-audit counts."""
    dependency_counts = (
        relation_dependencies[
            "dependency_status"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    primary = relation_dependencies.loc[
        relation_dependencies[
            "id02_status"
        ].isin(
            [
                "eligible",
                "candidate",
            ]
        )
    ]

    primary_dependency_counts = (
        primary[
            "dependency_status"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "status":
            "COMPLETED",

        "feature_count":
            int(
                len(
                    feature_dependencies
                )
            ),

        "relation_count":
            int(
                len(
                    relation_dependencies
                )
            ),

        "primary_audit_relation_count":
            int(
                len(
                    primary
                )
            ),

        "dependency_counts_all_relations": {
            str(key):
                int(value)
            for key, value
            in dependency_counts.items()
        },

        "dependency_counts_primary_population": {
            str(key):
                int(value)
            for key, value
            in primary_dependency_counts.items()
        },

        "directly_derived_relation_count":
            int(
                relation_dependencies[
                    "directly_derived_flag"
                ].sum()
            ),

        "definitionally_constrained_relation_count":
            int(
                relation_dependencies[
                    "definitionally_constrained_flag"
                ].sum()
            ),

        "id02_statuses_modified":
            False,
    }


def audit_dependencies(
    context: ID03ExperimentContext,
) -> ID03ExperimentContext:
    """Execute the frozen CGIE3-ID-03 dependency audit."""
    validate_context(
        context
    )

    contract = get_dependency_contract(
        context
    )

    feature_dependencies = (
        build_feature_dependency_table(
            contract
        )
    )

    relation_dependencies = (
        build_relation_dependency_table(
            context,
            contract,
        )
    )

    summary = build_dependency_summary(
        feature_dependencies,
        relation_dependencies,
    )

    context.register_output(
        "feature_dependencies",
        feature_dependencies,
    )

    context.register_output(
        "relation_dependencies",
        relation_dependencies,
    )

    context.register_output(
        "dependency_audit_summary",
        summary,
    )

    context.register_runtime(
        "dependency_audit_status",
        "COMPLETED",
    )

    return context

