"""CGIE3-ID-03 relational-family audit."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class FamilyAuditError(ValueError):
    """Raised when the family audit violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a normalized family-audit error."""
    raise FamilyAuditError(
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
        raise FamilyAuditError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
        )

    return normalized


def require_fraction(
    value: Any,
    field_name: str,
) -> float:
    """Require a finite value inside [0, 1]."""
    try:
        normalized = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise FamilyAuditError(
            f"{field_name} must be numeric."
        ) from exc

    if (
        not np.isfinite(normalized)
        or not 0.0 <= normalized <= 1.0
    ):
        fail(
            f"{field_name} must lie inside [0, 1]."
        )

    return normalized


def require_non_negative_float(
    value: Any,
    field_name: str,
) -> float:
    """Require a finite non-negative value."""
    try:
        normalized = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise FamilyAuditError(
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
    context: ID03ExperimentContext,
) -> None:
    """Validate family-audit prerequisites."""
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
            "conditional_redundancy_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Conditional redundancy must complete "
            "before the family audit."
        )

    required_outputs = {
        "primary_population",
        "relation_dependencies",
        "multiscale_relations",
        "overlap_sensitivity",
        "null_controls",
        "conditional_redundancy",
    }

    missing_outputs = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing_outputs:
        fail(
            "Family-audit prerequisites are missing: "
            + ", ".join(
                missing_outputs
            )
        )


def get_family_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate the frozen family-audit contract."""
    family_audit = require_mapping(
        context.configuration.get(
            "family_audit"
        ),
        "family_audit",
    )

    if family_audit.get(
        "enabled"
    ) is not True:
        fail(
            "Family audit must remain enabled."
        )

    graph = require_mapping(
        family_audit.get(
            "graph"
        ),
        "family_audit.graph",
    )

    if graph.get(
        "node_type"
    ) != "feature":
        fail(
            "Family graph nodes must represent features."
        )

    if graph.get(
        "edge_type"
    ) != "relation":
        fail(
            "Family graph edges must represent relations."
        )

    if graph.get(
        "undirected"
    ) is not True:
        fail(
            "Family graph must remain undirected."
        )

    edge_weight = require_mapping(
        family_audit.get(
            "edge_weight"
        ),
        "family_audit.edge_weight",
    )

    weight_components = require_mapping(
        edge_weight.get(
            "components"
        ),
        "family_audit.edge_weight.components",
    )

    expected_components = {
        "absolute_strength",
        "persistence",
        "multiscale_support_fraction",
        "overlap_robustness",
        "null_support",
        "residual_information",
    }

    if set(
        weight_components
    ) != expected_components:
        fail(
            "Unexpected family edge-weight components."
        )

    weights: dict[str, float] = {}

    for component_id in sorted(
        expected_components
    ):
        component = require_mapping(
            weight_components[
                component_id
            ],
            (
                "family_audit.edge_weight.components."
                f"{component_id}"
            ),
        )

        weights[
            component_id
        ] = require_non_negative_float(
            component.get(
                "weight"
            ),
            (
                "family_audit.edge_weight.components."
                f"{component_id}.weight"
            ),
        )

    if not np.isclose(
        sum(
            weights.values()
        ),
        1.0,
        atol=1e-12,
    ):
        fail(
            "Family edge weights must sum to 1.0."
        )

    community_detection = require_mapping(
        family_audit.get(
            "community_detection"
        ),
        "family_audit.community_detection",
    )

    if (
        community_detection.get(
            "primary_method"
        )
        != "greedy_modularity"
    ):
        fail(
            "Primary community method must be "
            "greedy_modularity."
        )

    if (
        community_detection.get(
            "secondary_method"
        )
        != "label_propagation"
    ):
        fail(
            "Secondary community method must be "
            "label_propagation."
        )

    stability = require_mapping(
        family_audit.get(
            "stability"
        ),
        "family_audit.stability",
    )

    minimum_family_size = require_mapping(
        family_audit.get(
            "minimum_family_size"
        ),
        "family_audit.minimum_family_size",
    )

    preliminary_labels = require_mapping(
        family_audit.get(
            "preliminary_labels"
        ),
        "family_audit.preliminary_labels",
    )

    if (
        preliminary_labels.get(
            "use_only_after_data_driven_detection"
        )
        is not True
    ):
        fail(
            "Preliminary labels must not determine "
            "community detection."
        )

    return {
        "weights":
            weights,

        "missing_component_value":
            require_fraction(
                edge_weight.get(
                    "missing_component_value"
                ),
                (
                    "family_audit.edge_weight."
                    "missing_component_value"
                ),
            ),

        "random_seed":
            require_positive_integer(
                community_detection.get(
                    "deterministic_seed"
                ),
                (
                    "family_audit.community_detection."
                    "deterministic_seed"
                ),
            ),

        "bootstrap_repetitions":
            require_positive_integer(
                stability.get(
                    "bootstrap_repetitions"
                ),
                (
                    "family_audit.stability."
                    "bootstrap_repetitions"
                ),
            ),

        "minimum_coassignment_fraction":
            require_fraction(
                stability.get(
                    "minimum_pairwise_"
                    "coassignment_fraction"
                ),
                (
                    "family_audit.stability."
                    "minimum_pairwise_"
                    "coassignment_fraction"
                ),
            ),

        "minimum_family_stability":
            require_fraction(
                stability.get(
                    "minimum_family_stability"
                ),
                (
                    "family_audit.stability."
                    "minimum_family_stability"
                ),
            ),

        "minimum_family_nodes":
            require_positive_integer(
                minimum_family_size.get(
                    "nodes"
                ),
                "family_audit.minimum_family_size.nodes",
            ),

        "minimum_family_relations":
            require_positive_integer(
                minimum_family_size.get(
                    "relations"
                ),
                (
                    "family_audit.minimum_family_size."
                    "relations"
                ),
            ),

        "family_id_format":
            str(
                family_audit.get(
                    "family_id_format"
                )
            ).strip(),

        "preliminary_label_candidates":
            require_mapping(
                preliminary_labels.get(
                    "candidates"
                ),
                (
                    "family_audit.preliminary_labels."
                    "candidates"
                ),
            ),
    }


def require_dataframe(
    value: Any,
    field_name: str,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Require a non-empty DataFrame and optional row count."""
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


def canonical_pair_id(
    source_id: str,
    target_id: str,
) -> str:
    """Return one canonical unordered pair identifier."""
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


def overlap_score(
    overlap_class: Any,
) -> float:
    """Map the frozen overlap class into a unit score."""
    mapping = {
        "overlap_robust":
            1.0,

        "moderately_overlap_sensitive":
            0.5,

        "strongly_overlap_sensitive":
            0.0,

        "non_estimable_non_overlapping":
            0.0,

        "inconclusive_overlap_audit":
            0.0,
    }

    return float(
        mapping.get(
            str(
                overlap_class
            ),
            0.0,
        )
    )


def null_score(
    null_outcome: Any,
) -> float:
    """Map the frozen null outcome into a unit score."""
    mapping = {
        "exceeds_null":
            1.0,

        "partially_exceeds_null":
            0.5,

        "equivalent_to_null":
            0.0,

        "null_test_inconclusive":
            0.0,

        "null_test_not_admissible":
            0.0,
    }

    return float(
        mapping.get(
            str(
                null_outcome
            ),
            0.0,
        )
    )


def residual_information_score(
    redundancy_status: Any,
) -> float:
    """Map conditional redundancy into a unit score."""
    mapping = {
        "retains_residual_information":
            1.0,

        "partially_redundant":
            0.5,

        "fully_redundant":
            0.0,

        "conditional_test_inconclusive":
            0.0,
    }

    return float(
        mapping.get(
            str(
                redundancy_status
            ),
            0.0,
        )
)

def build_edge_evidence(
    context: ID03ExperimentContext,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Merge every frozen audit into one evidence table used for
    graph construction.
    """

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

    evidence["pair_id"] = [
        canonical_pair_id(
            s,
            t,
        )
        for s, t in zip(
            evidence["source_id"],
            evidence["target_id"],
        )
    ]

    pair_support = multiscale[
        [
            "pair_id",
            "supported_scale_count",
            "multiscale_class",
            "sign_consistent",
            "scale_inconsistent",
        ]
    ].copy()

    evidence = evidence.merge(
        pair_support,
        on="pair_id",
        how="left",
        validate="many_to_one",
    )

    weights = contract["weights"]

    evidence["edge_weight"] = (
        evidence["absolute_strength"].fillna(0.0)
        * weights["absolute_strength"]
        + evidence["persistence"].fillna(0.0)
        * weights["persistence"]
        + (
            evidence["supported_scale_count"]
            .fillna(0)
            .clip(0, 4)
            / 4.0
        )
        * weights["multiscale_support_fraction"]
        + evidence["overlap_class"].map(
            overlap_score
        )
        * weights["overlap_robustness"]
        + evidence["null_outcome"].map(
            null_score
        )
        * weights["null_support"]
        + evidence["redundancy_status"].map(
            residual_information_score
        )
        * weights["residual_information"]
    )

    return evidence.sort_values(
        by=[
            "source_id",
            "target_id",
            "window_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True,
    )

def aggregate_pair_edges(
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate scale-specific relations into one graph edge
    for each unordered feature pair.
    """
    records: list[dict[str, Any]] = []

    for pair_id, frame in evidence.groupby(
        "pair_id",
        sort=True,
    ):
        first = frame.iloc[0]

        source_id, target_id = sorted(
            (
                str(first["source_id"]),
                str(first["target_id"]),
            )
        )

        supported_scale_values = pd.to_numeric(
            frame["supported_scale_count"],
            errors="coerce",
        ).dropna()

        records.append(
            {
                "pair_id":
                    str(pair_id),

                "source_id":
                    source_id,

                "target_id":
                    target_id,

                "scale_relation_count":
                    int(len(frame)),

                "eligible_scale_count":
                    int(
                        (
                            frame["classification_status"]
                            == "eligible"
                        ).sum()
                    ),

                "candidate_scale_count":
                    int(
                        (
                            frame["classification_status"]
                            == "candidate"
                        ).sum()
                    ),

                "mean_edge_weight":
                    float(
                        frame["edge_weight"].mean()
                    ),

                "maximum_edge_weight":
                    float(
                        frame["edge_weight"].max()
                    ),

                "graph_edge_weight":
                    float(
                        frame["edge_weight"].mean()
                    ),

                "mean_absolute_strength":
                    float(
                        pd.to_numeric(
                            frame["absolute_strength"],
                            errors="coerce",
                        ).mean()
                    ),

                "mean_persistence":
                    float(
                        pd.to_numeric(
                            frame["persistence"],
                            errors="coerce",
                        ).mean()
                    ),

                "supported_scale_count":
                    (
                        int(
                            supported_scale_values.max()
                        )
                        if not supported_scale_values.empty
                        else 0
                    ),

                "overlap_robust_scale_count":
                    int(
                        (
                            frame["overlap_class"]
                            == "overlap_robust"
                        ).sum()
                    ),

                "null_exceeds_scale_count":
                    int(
                        (
                            frame["null_outcome"]
                            == "exceeds_null"
                        ).sum()
                    ),

                "residual_information_scale_count":
                    int(
                        frame[
                            "retains_residual_information"
                        ]
                        .fillna(False)
                        .sum()
                    ),

                "dependency_statuses":
                    "|".join(
                        sorted(
                            set(
                                frame[
                                    "dependency_status"
                                ].astype(str)
                            )
                        )
                    ),

                "window_ids":
                    "|".join(
                        sorted(
                            set(
                                frame[
                                    "window_id"
                                ].astype(str)
                            )
                        )
                    ),

                "relation_ids":
                    "|".join(
                        sorted(
                            frame[
                                "relation_id"
                            ].astype(str)
                        )
                    ),
            }
        )

    output = pd.DataFrame.from_records(
        records
    )

    if output.empty:
        fail(
            "Pair-edge aggregation produced no graph edges."
        )

    if output["pair_id"].duplicated().any():
        fail(
            "Pair-edge aggregation contains duplicate pair IDs."
        )

    return output.sort_values(
        by=[
            "source_id",
            "target_id",
            "pair_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_graph(
    context: ID03ExperimentContext,
    pair_edges: pd.DataFrame,
) -> nx.Graph:
    """Build the weighted undirected feature graph."""
    graph = nx.Graph()

    for component_id in sorted(
        context.identity.component_ids
    ):
        graph.add_node(
            str(component_id)
        )

    for row in pair_edges.itertuples(
        index=False
    ):
        graph.add_edge(
            str(row.source_id),
            str(row.target_id),
            pair_id=
                str(row.pair_id),
            weight=
                float(row.graph_edge_weight),
            scale_relation_count=
                int(row.scale_relation_count),
        )

    if graph.number_of_edges() == 0:
        fail(
            "Family graph contains no edges."
        )

    return graph


def canonicalize_communities(
    communities: Any,
) -> tuple[tuple[str, ...], ...]:
    """
    Convert community collections into deterministic,
    lexicographically ordered tuples.
    """
    normalized = [
        tuple(
            sorted(
                str(node)
                for node in community
            )
        )
        for community in communities
        if community
    ]

    return tuple(
        sorted(
            normalized,
            key=lambda community: (
                community[0],
                len(community),
                community,
            ),
        )
    )


def detect_primary_communities(
    graph: nx.Graph,
) -> tuple[tuple[str, ...], ...]:
    """Run weighted greedy-modularity detection."""
    communities = (
        nx.algorithms.community
        .greedy_modularity_communities(
            graph,
            weight="weight",
        )
    )

    return canonicalize_communities(
        communities
    )


def detect_secondary_communities(
    graph: nx.Graph,
    *,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Run seeded asynchronous label propagation."""
    communities = (
        nx.algorithms.community
        .asyn_lpa_communities(
            graph,
            weight="weight",
            seed=seed,
        )
    )

    return canonicalize_communities(
        communities
    )


def partition_map(
    communities: tuple[
        tuple[str, ...],
        ...,
    ],
) -> dict[str, int]:
    """Map every feature to one community index."""
    output: dict[str, int] = {}

    for community_index, community in enumerate(
        communities,
        start=1,
    ):
        for node_id in community:
            if node_id in output:
                fail(
                    f"Node {node_id} appears in "
                    "multiple communities."
                )

            output[node_id] = community_index

    return output


def deterministic_bootstrap_seed(
    base_seed: int,
    repetition: int,
) -> int:
    """Derive one stable graph-bootstrap seed."""
    token = (
        f"{base_seed}::family_bootstrap::{repetition}"
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

def bootstrap_graph(
    pair_edges: pd.DataFrame,
    *,
    seed: int,
) -> nx.Graph:
    """
    Generate one weighted graph bootstrap.

    Pair edges are resampled with replacement. Repeated sampled edges
    contribute their mean weight multiplied by their sampling count.
    """
    rng = np.random.default_rng(
        seed
    )

    sampled_indices = rng.integers(
        low=0,
        high=len(
            pair_edges
        ),
        size=len(
            pair_edges
        ),
    )

    sampled = pair_edges.iloc[
        sampled_indices
    ].copy()

    graph = nx.Graph()

    for row in pair_edges.itertuples(
        index=False
    ):
        graph.add_node(
            str(
                row.source_id
            )
        )

        graph.add_node(
            str(
                row.target_id
            )
        )

    for pair_id, frame in sampled.groupby(
        "pair_id",
        sort=True,
    ):
        first = frame.iloc[
            0
        ]

        weight = float(
            frame[
                "graph_edge_weight"
            ].mean()
            * len(
                frame
            )
        )

        graph.add_edge(
            str(
                first[
                    "source_id"
                ]
            ),
            str(
                first[
                    "target_id"
                ]
            ),
            pair_id=
                str(
                    pair_id
                ),
            weight=
                weight,
        )

    return graph


def build_coassignment_table(
    pair_edges: pd.DataFrame,
    *,
    repetitions: int,
    base_seed: int,
) -> pd.DataFrame:
    """Estimate pairwise node coassignment across graph bootstraps."""
    nodes = sorted(
        set(
            pair_edges[
                "source_id"
            ].astype(str)
        )
        | set(
            pair_edges[
                "target_id"
            ].astype(str)
        )
    )

    counters: Counter[
        tuple[str, str]
    ] = Counter()

    for repetition in range(
        1,
        repetitions + 1,
    ):
        graph = bootstrap_graph(
            pair_edges,
            seed=deterministic_bootstrap_seed(
                base_seed,
                repetition,
            ),
        )

        communities = detect_primary_communities(
            graph
        )

        membership = partition_map(
            communities
        )

        for left_index, left_node in enumerate(
            nodes
        ):
            for right_node in nodes[
                left_index + 1:
            ]:
                if (
                    membership.get(
                        left_node
                    )
                    == membership.get(
                        right_node
                    )
                ):
                    counters[
                        (
                            left_node,
                            right_node,
                        )
                    ] += 1

    records: list[dict[str, Any]] = []

    for left_index, left_node in enumerate(
        nodes
    ):
        for right_node in nodes[
            left_index + 1:
        ]:
            coassignment_count = int(
                counters[
                    (
                        left_node,
                        right_node,
                    )
                ]
            )

            records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_03",

                    "source_feature_id":
                        left_node,

                    "target_feature_id":
                        right_node,

                    "bootstrap_repetitions":
                        repetitions,

                    "coassignment_count":
                        coassignment_count,

                    "coassignment_fraction":
                        float(
                            coassignment_count
                            / repetitions
                        ),
                }
            )

    return pd.DataFrame.from_records(
        records
    ).sort_values(
        by=[
            "source_feature_id",
            "target_feature_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def family_stability_score(
    members: tuple[str, ...],
    coassignment: pd.DataFrame,
) -> float:
    """Calculate mean internal bootstrap coassignment."""
    if len(
        members
    ) < 2:
        return 0.0

    member_set = set(
        members
    )

    internal = coassignment.loc[
        coassignment[
            "source_feature_id"
        ].isin(
            member_set
        )
        & coassignment[
            "target_feature_id"
        ].isin(
            member_set
        )
    ]

    if internal.empty:
        return 0.0

    return float(
        internal[
            "coassignment_fraction"
        ].mean()
    )


def suggest_preliminary_label(
    members: tuple[str, ...],
    label_candidates: Mapping[str, Any],
) -> tuple[str | None, float]:
    """
    Suggest an interpretive label only after community detection.

    The label never changes family membership.
    """
    member_set = set(
        members
    )

    best_label: str | None = None
    best_overlap = 0.0

    for label_id, payload in label_candidates.items():
        candidate = require_mapping(
            payload,
            (
                "family_audit.preliminary_labels."
                f"candidates.{label_id}"
            ),
        )

        components_raw = candidate.get(
            "components"
        )

        if not isinstance(
            components_raw,
            (list, tuple),
        ):
            fail(
                f"Preliminary label {label_id} "
                "must declare a component list."
            )

        candidate_set = {
            str(
                component
            ).strip()
            for component in components_raw
        }

        union = member_set | candidate_set

        overlap = (
            float(
                len(
                    member_set
                    & candidate_set
                )
                / len(
                    union
                )
            )
            if union
            else 0.0
        )

        if (
            overlap > best_overlap
            or (
                np.isclose(
                    overlap,
                    best_overlap,
                )
                and best_label is not None
                and str(
                    label_id
                )
                < best_label
            )
        ):
            best_label = str(
                label_id
            )

            best_overlap = overlap

    return (
        best_label,
        float(
            best_overlap
        ),
          )

def build_family_outputs(
    context: ID03ExperimentContext,
    evidence: pd.DataFrame,
    pair_edges: pd.DataFrame,
    primary_communities: tuple[
        tuple[str, ...],
        ...,
    ],
    secondary_communities: tuple[
        tuple[str, ...],
        ...,
    ],
    coassignment: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build family and relation-membership tables."""
    secondary_membership = partition_map(
        secondary_communities
    )

    family_records: list[
        dict[str, Any]
    ] = []

    membership_records: list[
        dict[str, Any]
    ] = []

    for family_index, members in enumerate(
        primary_communities,
        start=1,
    ):
        family_id = contract[
            "family_id_format"
        ].format(
            index=family_index
        )

        member_set = set(
            members
        )

        family_edges = pair_edges.loc[
            pair_edges[
                "source_id"
            ].isin(
                member_set
            )
            & pair_edges[
                "target_id"
            ].isin(
                member_set
            )
        ].copy()

        family_pair_ids = set(
            family_edges[
                "pair_id"
            ].astype(str)
        )

        scale_relations = evidence.loc[
            evidence[
                "pair_id"
            ].astype(str)
            .isin(
                family_pair_ids
            )
        ].copy()

        stability = family_stability_score(
            members,
            coassignment,
        )

        secondary_labels = {
            secondary_membership.get(
                member
            )
            for member in members
        }

        secondary_agreement = bool(
            len(
                secondary_labels
            )
            == 1
        )

        reproducible = bool(
            len(
                members
            )
            >= contract[
                "minimum_family_nodes"
            ]
            and len(
                family_edges
            )
            >= contract[
                "minimum_family_relations"
            ]
            and stability
            >= contract[
                "minimum_family_stability"
            ]
        )

        (
            preliminary_label,
            preliminary_overlap,
        ) = suggest_preliminary_label(
            members,
            contract[
                "preliminary_label_candidates"
            ],
        )

        family_records.append(
            {
                "experiment_id":
                    "CGIE3_ID_03",

                "family_id":
                    family_id,

                "member_node_count":
                    int(
                        len(
                            members
                        )
                    ),

                "member_relation_pair_count":
                    int(
                        len(
                            family_edges
                        )
                    ),

                "member_scale_relation_count":
                    int(
                        len(
                            scale_relations
                        )
                    ),

                "member_components":
                    "|".join(
                        members
                    ),

                "member_pair_ids":
                    "|".join(
                        sorted(
                            family_pair_ids
                        )
                    ),

                "eligible_relation_count":
                    int(
                        (
                            scale_relations[
                                "classification_status"
                            ]
                            == "eligible"
                        ).sum()
                    ),

                "candidate_relation_count":
                    int(
                        (
                            scale_relations[
                                "classification_status"
                            ]
                            == "candidate"
                        ).sum()
                    ),

                "mean_graph_edge_weight":
                    (
                        float(
                            family_edges[
                                "graph_edge_weight"
                            ].mean()
                        )
                        if not family_edges.empty
                        else None
                    ),

                "maximum_supported_scale_count":
                    (
                        int(
                            family_edges[
                                "supported_scale_count"
                            ].max()
                        )
                        if not family_edges.empty
                        else 0
                    ),

                "overlap_robust_relation_count":
                    int(
                        (
                            scale_relations[
                                "overlap_class"
                            ]
                            == "overlap_robust"
                        ).sum()
                    ),

                "null_exceeds_relation_count":
                    int(
                        (
                            scale_relations[
                                "null_outcome"
                            ]
                            == "exceeds_null"
                        ).sum()
                    ),

                "residual_information_relation_count":
                    int(
                        scale_relations[
                            "retains_residual_information"
                        ]
                        .fillna(
                            False
                        )
                        .sum()
                    ),

                "dependency_statuses":
                    "|".join(
                        sorted(
                            set(
                                scale_relations[
                                    "dependency_status"
                                ].astype(str)
                            )
                        )
                    ),

                "family_stability":
                    float(
                        stability
                    ),

                "secondary_partition_agreement":
                    secondary_agreement,

                "reproducible_family":
                    reproducible,

                "preliminary_label_suggestion":
                    preliminary_label,

                "preliminary_label_jaccard_overlap":
                    float(
                        preliminary_overlap
                    ),

                "preliminary_label_used_for_detection":
                    False,
            }
        )

        for row in scale_relations.itertuples(
            index=False
        ):
            membership_records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_03",

                    "family_id":
                        family_id,

                    "window_id":
                        str(
                            row.window_id
                        ),

                    "relation_id":
                        str(
                            row.relation_id
                        ),

                    "pair_id":
                        str(
                            row.pair_id
                        ),

                    "source_id":
                        str(
                            row.source_id
                        ),

                    "target_id":
                        str(
                            row.target_id
                        ),

                    "id02_status":
                        str(
                            row.classification_status
                        ),

                    "edge_weight":
                        float(
                            row.edge_weight
                        ),

                    "dependency_status":
                        str(
                            row.dependency_status
                        ),

                    "overlap_class":
                        str(
                            row.overlap_class
                        ),

                    "null_outcome":
                        str(
                            row.null_outcome
                        ),

                    "redundancy_status":
                        str(
                            row.redundancy_status
                        ),

                    "supported_scale_count":
                        int(
                            row.supported_scale_count
                        ),

                    "family_reproducible":
                        reproducible,

                    "id02_status_modified":
                        False,
                }
            )

        families = pd.DataFrame.from_records(
        family_records
    )

        membership = pd.DataFrame.from_records(
        membership_records
    )

# ------------------------------------------------------------------
# Ensure every primary relation belongs to one family.
# Any unassigned relation becomes its own singleton family.
# ------------------------------------------------------------------

    assigned = set(
    zip(
        membership["window_id"],
        membership["relation_id"],
    )
)

    missing = evidence.loc[
    ~evidence.apply(
        lambda r: (
            str(r["window_id"]),
            str(r["relation_id"]),
        ) in assigned,
        axis=1,
    )
]

    next_index = len(families) + 1

    for row in missing.itertuples(index=False):

        family_id = contract["family_id_format"].format(
        index=next_index
    )
    next_index += 1

    families.loc[len(families)] = {
        "experiment_id": "CGIE3_ID_03",
        "family_id": family_id,
        "member_node_count": 2,
        "member_relation_pair_count": 1,
        "member_scale_relation_count": 1,
        "member_components": f"{row.source_id}|{row.target_id}",
        "member_pair_ids": row.pair_id,
        "eligible_relation_count": int(row.classification_status == "eligible"),
        "candidate_relation_count": int(row.classification_status == "candidate"),
        "mean_graph_edge_weight": float(row.edge_weight),
        "maximum_supported_scale_count": int(row.supported_scale_count),
        "overlap_robust_relation_count": int(row.overlap_class == "overlap_robust"),
        "null_exceeds_relation_count": int(row.null_outcome == "exceeds_null"),
        "residual_information_relation_count": int(bool(row.retains_residual_information)),
        "dependency_statuses": str(row.dependency_status),
        "family_stability": 0.0,
        "secondary_partition_agreement": True,
        "reproducible_family": False,
        "preliminary_label_suggestion": None,
        "preliminary_label_jaccard_overlap": 0.0,
        "preliminary_label_used_for_detection": False,
    }

    membership.loc[len(membership)] = {
        "experiment_id": "CGIE3_ID_03",
        "family_id": family_id,
        "window_id": str(row.window_id),
        "relation_id": str(row.relation_id),
        "pair_id": str(row.pair_id),
        "source_id": str(row.source_id),
        "target_id": str(row.target_id),
        "id02_status": str(row.classification_status),
        "edge_weight": float(row.edge_weight),
        "dependency_status": str(row.dependency_status),
        "overlap_class": str(row.overlap_class),
        "null_outcome": str(row.null_outcome),
        "redundancy_status": str(row.redundancy_status),
        "supported_scale_count": int(row.supported_scale_count),
        "family_reproducible": False,
        "id02_status_modified": False,
    }
# Ensure every primary relation belongs to one family.
    # Any unassigned relation becomes its own singleton family.
    assigned = set(
        zip(
            membership["window_id"].astype(str),
            membership["relation_id"].astype(str),
        )
    )

    missing = evidence.loc[
        ~evidence.apply(
            lambda row: (
                str(row["window_id"]),
                str(row["relation_id"]),
            )
            in assigned,
            axis=1,
        )
    ].copy()

    next_index = len(families) + 1

    for row in missing.itertuples(
        index=False
    ):
        family_id = contract[
            "family_id_format"
        ].format(
            index=next_index
        )

        next_index += 1

        families.loc[
            len(families)
        ] = {
            "experiment_id":
                "CGIE3_ID_03",

            "family_id":
                family_id,

            "member_node_count":
                2,

            "member_relation_pair_count":
                1,

            "member_scale_relation_count":
                1,

            "member_components":
                f"{row.source_id}|{row.target_id}",

            "member_pair_ids":
                str(row.pair_id),

            "eligible_relation_count":
                int(
                    row.classification_status
                    == "eligible"
                ),

            "candidate_relation_count":
                int(
                    row.classification_status
                    == "candidate"
                ),

            "mean_graph_edge_weight":
                float(row.edge_weight),

            "maximum_supported_scale_count":
                int(row.supported_scale_count),

            "overlap_robust_relation_count":
                int(
                    row.overlap_class
                    == "overlap_robust"
                ),

            "null_exceeds_relation_count":
                int(
                    row.null_outcome
                    == "exceeds_null"
                ),

            "residual_information_relation_count":
                int(
                    bool(
                        row.retains_residual_information
                    )
                ),

            "dependency_statuses":
                str(
                    row.dependency_status
                ),

            "family_stability":
                0.0,

            "secondary_partition_agreement":
                True,

            "reproducible_family":
                False,

            "preliminary_label_suggestion":
                None,

            "preliminary_label_jaccard_overlap":
                0.0,

            "preliminary_label_used_for_detection":
                False,
        }

    # Preserve every primary relation exactly once.
    assignment_columns = [
        "window_id",
        "relation_id",
        "pair_id",
        "source_id",
        "target_id",
    ]

    assigned_keys = set(
        membership[
            assignment_columns
        ]
        .astype(str)
        .itertuples(
            index=False,
            name=None,
        )
    )

    missing_mask = evidence.apply(
        lambda row: (
            str(row["window_id"]),
            str(row["relation_id"]),
            str(row["pair_id"]),
            str(row["source_id"]),
            str(row["target_id"]),
        )
        not in assigned_keys,
        axis=1,
    )

    missing = evidence.loc[
        missing_mask
    ].copy()

    next_index = len(families) + 1

    for row in missing.itertuples(
        index=False
    ):
        family_id = contract[
            "family_id_format"
        ].format(
            index=next_index
        )

        next_index += 1

        families.loc[
            len(families)
        ] = {
            "experiment_id":
                "CGIE3_ID_03",

            "family_id":
                family_id,

            "member_node_count":
                2,

            "member_relation_pair_count":
                1,

            "member_scale_relation_count":
                1,

            "member_components":
                f"{row.source_id}|{row.target_id}",

            "member_pair_ids":
                str(row.pair_id),

            "eligible_relation_count":
                int(
                    row.classification_status
                    == "eligible"
                ),

            "candidate_relation_count":
                int(
                    row.classification_status
                    == "candidate"
                ),

            "mean_graph_edge_weight":
                float(row.edge_weight),

            "maximum_supported_scale_count":
                int(row.supported_scale_count),

            "overlap_robust_relation_count":
                int(
                    row.overlap_class
                    == "overlap_robust"
                ),

            "null_exceeds_relation_count":
                int(
                    row.null_outcome
                    == "exceeds_null"
                ),

            "residual_information_relation_count":
                int(
                    bool(
                        row.retains_residual_information
                    )
                ),

            "dependency_statuses":
                str(row.dependency_status),

            "family_stability":
                0.0,

            "secondary_partition_agreement":
                True,

            "reproducible_family":
                False,

            "preliminary_label_suggestion":
                None,

            "preliminary_label_jaccard_overlap":
                0.0,

            "preliminary_label_used_for_detection":
                False,
        }

        membership.loc[
            len(membership)
        ] = {
            "experiment_id":
                "CGIE3_ID_03",

            "family_id":
                family_id,

            "window_id":
                str(row.window_id),

            "relation_id":
                str(row.relation_id),

            "pair_id":
                str(row.pair_id),

            "source_id":
                str(row.source_id),

            "target_id":
                str(row.target_id),

            "id02_status":
                str(row.classification_status),

            "edge_weight":
                float(row.edge_weight),

            "dependency_status":
                str(row.dependency_status),

            "overlap_class":
                str(row.overlap_class),

            "null_outcome":
                str(row.null_outcome),

            "redundancy_status":
                str(row.redundancy_status),

            "supported_scale_count":
                int(row.supported_scale_count),

            "family_reproducible":
                False,

            "id02_status_modified":
                False,
        }

        if families.empty:
            fail(
            "Family audit produced no family records."
        )

        if len(membership) != 74:
            fail(
            "Family membership must preserve all "
            f"74 primary relations; observed "
            f"{len(membership)}."
        )

        if membership.duplicated(
            subset=[
                "window_id",
                "relation_id",
        ],
        keep=False,
        ).any():
            fail(
            "Family membership contains duplicate "
            "window-relation keys."
        )

        return (
        families.sort_values(
            by=[
                "family_id",
            ],
            kind="stable",
        ).reset_index(
def build_family_outputs(
    context: ID03ExperimentContext,
    evidence: pd.DataFrame,
    pair_edges: pd.DataFrame,
    primary_communities: tuple[
        tuple[str, ...],
        ...,
    ],
    secondary_communities: tuple[
        tuple[str, ...],
        ...,
    ],
    coassignment: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build family and relation-membership tables."""
    secondary_membership = partition_map(
        secondary_communities
    )

    family_records: list[
        dict[str, Any]
    ] = []

    membership_records: list[
        dict[str, Any]
    ] = []

    for family_index, members in enumerate(
        primary_communities,
        start=1,
    ):
        family_id = contract[
            "family_id_format"
        ].format(
            index=family_index
        )

        member_set = set(
            members
        )

        family_edges = pair_edges.loc[
            pair_edges[
                "source_id"
            ].isin(
                member_set
            )
            & pair_edges[
                "target_id"
            ].isin(
                member_set
            )
        ].copy()

        family_pair_ids = set(
            family_edges[
                "pair_id"
            ].astype(str)
        )

        scale_relations = evidence.loc[
            evidence[
                "pair_id"
            ]
            .astype(str)
            .isin(
                family_pair_ids
            )
        ].copy()

        stability = family_stability_score(
            members,
            coassignment,
        )

        secondary_labels = {
            secondary_membership.get(
                member
            )
            for member in members
        }

        secondary_agreement = bool(
            len(
                secondary_labels
            )
            == 1
        )

        reproducible = bool(
            len(
                members
            )
            >= contract[
                "minimum_family_nodes"
            ]
            and len(
                family_edges
            )
            >= contract[
                "minimum_family_relations"
            ]
            and stability
            >= contract[
                "minimum_family_stability"
            ]
        )

        (
            preliminary_label,
            preliminary_overlap,
        ) = suggest_preliminary_label(
            members,
            contract[
                "preliminary_label_candidates"
            ],
        )

        family_records.append(
            {
                "experiment_id":
                    "CGIE3_ID_03",

                "family_id":
                    family_id,

                "member_node_count":
                    int(
                        len(
                            members
                        )
                    ),

                "member_relation_pair_count":
                    int(
                        len(
                            family_edges
                        )
                    ),

                "member_scale_relation_count":
                    int(
                        len(
                            scale_relations
                        )
                    ),

                "member_components":
                    "|".join(
                        members
                    ),

                "member_pair_ids":
                    "|".join(
                        sorted(
                            family_pair_ids
                        )
                    ),

                "eligible_relation_count":
                    int(
                        (
                            scale_relations[
                                "classification_status"
                            ]
                            == "eligible"
                        ).sum()
                    ),

                "candidate_relation_count":
                    int(
                        (
                            scale_relations[
                                "classification_status"
                            ]
                            == "candidate"
                        ).sum()
                    ),

                "mean_graph_edge_weight":
                    (
                        float(
                            family_edges[
                                "graph_edge_weight"
                            ].mean()
                        )
                        if not family_edges.empty
                        else None
                    ),

                "maximum_supported_scale_count":
                    (
                        int(
                            family_edges[
                                "supported_scale_count"
                            ].max()
                        )
                        if not family_edges.empty
                        else 0
                    ),

                "overlap_robust_relation_count":
                    int(
                        (
                            scale_relations[
                                "overlap_class"
                            ]
                            == "overlap_robust"
                        ).sum()
                    ),

                "null_exceeds_relation_count":
                    int(
                        (
                            scale_relations[
                                "null_outcome"
                            ]
                            == "exceeds_null"
                        ).sum()
                    ),

                "residual_information_relation_count":
                    int(
                        scale_relations[
                            "retains_residual_information"
                        ]
                        .fillna(
                            False
                        )
                        .sum()
                    ),

                "dependency_statuses":
                    "|".join(
                        sorted(
                            set(
                                scale_relations[
                                    "dependency_status"
                                ].astype(str)
                            )
                        )
                    ),

                "family_stability":
                    float(
                        stability
                    ),

                "secondary_partition_agreement":
                    secondary_agreement,

                "reproducible_family":
                    reproducible,

                "preliminary_label_suggestion":
                    preliminary_label,

                "preliminary_label_jaccard_overlap":
                    float(
                        preliminary_overlap
                    ),

                "preliminary_label_used_for_detection":
                    False,
            }
        )

        for row in scale_relations.itertuples(
            index=False
        ):
            membership_records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_03",

                    "family_id":
                        family_id,

                    "window_id":
                        str(
                            row.window_id
                        ),

                    "relation_id":
                        str(
                            row.relation_id
                        ),

                    "pair_id":
                        str(
                            row.pair_id
                        ),

                    "source_id":
                        str(
                            row.source_id
                        ),

                    "target_id":
                        str(
                            row.target_id
                        ),

                    "id02_status":
                        str(
                            row.classification_status
                        ),

                    "edge_weight":
                        float(
                            row.edge_weight
                        ),

                    "dependency_status":
                        str(
                            row.dependency_status
                        ),

                    "overlap_class":
                        str(
                            row.overlap_class
                        ),

                    "null_outcome":
                        str(
                            row.null_outcome
                        ),

                    "redundancy_status":
                        str(
                            row.redundancy_status
                        ),

                    "supported_scale_count":
                        int(
                            row.supported_scale_count
                        ),

                    "family_reproducible":
                        reproducible,

                    "id02_status_modified":
                        False,
                }
            )

    families = pd.DataFrame.from_records(
        family_records
    )

    membership = pd.DataFrame.from_records(
        membership_records
    )

    assignment_columns = [
        "window_id",
        "relation_id",
        "pair_id",
        "source_id",
        "target_id",
    ]

    if membership.empty:
        assigned_keys: set[
            tuple[str, ...]
        ] = set()
    else:
        assigned_keys = set(
            membership[
                assignment_columns
            ]
            .astype(str)
            .itertuples(
                index=False,
                name=None,
            )
        )

    missing_mask = evidence.apply(
        lambda row: (
            str(
                row["window_id"]
            ),
            str(
                row["relation_id"]
            ),
            str(
                row["pair_id"]
            ),
            str(
                row["source_id"]
            ),
            str(
                row["target_id"]
            ),
        )
        not in assigned_keys,
        axis=1,
    )

    missing = evidence.loc[
        missing_mask
    ].copy()

    next_index = len(
        families
    ) + 1

    for row in missing.itertuples(
        index=False
    ):
        family_id = contract[
            "family_id_format"
        ].format(
            index=next_index
        )

        next_index += 1

        singleton_family = {
            "experiment_id":
                "CGIE3_ID_03",

            "family_id":
                family_id,

            "member_node_count":
                2,

            "member_relation_pair_count":
                1,

            "member_scale_relation_count":
                1,

            "member_components":
                f"{row.source_id}|{row.target_id}",

            "member_pair_ids":
                str(
                    row.pair_id
                ),

            "eligible_relation_count":
                int(
                    row.classification_status
                    == "eligible"
                ),

            "candidate_relation_count":
                int(
                    row.classification_status
                    == "candidate"
                ),

            "mean_graph_edge_weight":
                float(
                    row.edge_weight
                ),

            "maximum_supported_scale_count":
                int(
                    row.supported_scale_count
                ),

            "overlap_robust_relation_count":
                int(
                    row.overlap_class
                    == "overlap_robust"
                ),

            "null_exceeds_relation_count":
                int(
                    row.null_outcome
                    == "exceeds_null"
                ),

            "residual_information_relation_count":
                int(
                    bool(
                        row.retains_residual_information
                    )
                ),

            "dependency_statuses":
                str(
                    row.dependency_status
                ),

            "family_stability":
                0.0,

            "secondary_partition_agreement":
                True,

            "reproducible_family":
                False,

            "preliminary_label_suggestion":
                None,

            "preliminary_label_jaccard_overlap":
                0.0,

            "preliminary_label_used_for_detection":
                False,
        }

        singleton_membership = {
            "experiment_id":
                "CGIE3_ID_03",

            "family_id":
                family_id,

            "window_id":
                str(
                    row.window_id
                ),

            "relation_id":
                str(
                    row.relation_id
                ),

            "pair_id":
                str(
                    row.pair_id
                ),

            "source_id":
                str(
                    row.source_id
                ),

            "target_id":
                str(
                    row.target_id
                ),

            "id02_status":
                str(
                    row.classification_status
                ),

            "edge_weight":
                float(
                    row.edge_weight
                ),

            "dependency_status":
                str(
                    row.dependency_status
                ),

            "overlap_class":
                str(
                    row.overlap_class
                ),

            "null_outcome":
                str(
                    row.null_outcome
                ),

            "redundancy_status":
                str(
                    row.redundancy_status
                ),

            "supported_scale_count":
                int(
                    row.supported_scale_count
                ),

            "family_reproducible":
                False,

            "id02_status_modified":
                False,
        }

        families = pd.concat(
            [
                families,
                pd.DataFrame(
                    [
                        singleton_family
                    ]
                ),
            ],
            ignore_index=True,
        )

        membership = pd.concat(
            [
                membership,
                pd.DataFrame(
                    [
                        singleton_membership
                    ]
                ),
            ],
            ignore_index=True,
        )

    if families.empty:
        fail(
            "Family audit produced no family records."
        )

    if len(
        membership
    ) != len(
        evidence
    ):
        fail(
            "Family membership must preserve all "
            f"{len(evidence)} primary relations; observed "
            f"{len(membership)}."
        )

    if membership.duplicated(
        subset=assignment_columns,
        keep=False,
    ).any():
        fail(
            "Family membership contains duplicate "
            "primary-relation keys."
        )

    return (
        families.sort_values(
            by=[
                "family_id",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
        membership.sort_values(
            by=[
                "family_id",
                "source_id",
                "target_id",
                "window_id",
                "relation_id",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
        )

def build_family_summary(
    graph: nx.Graph,
    families: pd.DataFrame,
    membership: pd.DataFrame,
    primary_communities: tuple[
        tuple[str, ...],
        ...,
    ],
    secondary_communities: tuple[
        tuple[str, ...],
        ...,
    ],
) -> dict[str, Any]:
    """Build descriptive family-audit results."""
    reproducible_count = int(
        families[
            "reproducible_family"
        ].sum()
    )

    return {
        "status":
            "COMPLETED",

        "graph_node_count":
            int(
                graph.number_of_nodes()
            ),

        "graph_edge_count":
            int(
                graph.number_of_edges()
            ),

        "primary_family_count":
            int(
                len(
                    primary_communities
                )
            ),

        "secondary_family_count":
            int(
                len(
                    secondary_communities
                )
            ),

        "reproducible_family_count":
            reproducible_count,

        "primary_relation_membership_count":
            int(
                len(
                    membership
                )
            ),

        "all_primary_relations_assigned":
            bool(
                len(
                    membership
                )
                == 74
            ),

        "preliminary_labels_used_for_detection":
            False,

        "id02_statuses_modified":
            False,
      }

def audit_families(
    context: ID03ExperimentContext,
) -> ID03ExperimentContext:
    """Execute the frozen CGIE3-ID-03 relational-family audit."""
    validate_context(
        context
    )

    contract = get_family_contract(
        context
    )

    evidence = build_edge_evidence(
        context,
        contract,
    )

    pair_edges = aggregate_pair_edges(
        evidence
    )

    graph = build_graph(
        context,
        pair_edges,
    )

    primary_communities = (
        detect_primary_communities(
            graph
        )
    )

    secondary_communities = (
        detect_secondary_communities(
            graph,
            seed=contract[
                "random_seed"
            ],
        )
    )

    coassignment = build_coassignment_table(
        pair_edges,
        repetitions=contract[
            "bootstrap_repetitions"
        ],
        base_seed=contract[
            "random_seed"
        ],
    )

    (
        families,
        membership,
    ) = build_family_outputs(
        context,
        evidence,
        pair_edges,
        primary_communities,
        secondary_communities,
        coassignment,
        contract,
    )

    summary = build_family_summary(
        graph,
        families,
        membership,
        primary_communities,
        secondary_communities,
    )

    context.register_output(
        "family_edge_evidence",
        evidence,
    )

    context.register_output(
        "family_pair_edges",
        pair_edges,
    )

    context.register_output(
        "family_coassignment",
        coassignment,
    )

    context.register_output(
        "relation_families",
        families,
    )

    context.register_output(
        "family_membership",
        membership,
    )

    context.register_output(
        "family_audit_summary",
        summary,
    )

    context.register_output(
        "family_graph",
        graph,
    )

    context.register_runtime(
        "family_audit_status",
        "COMPLETED",
    )

    return context
