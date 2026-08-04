"""Relation discovery tools for Congruity Framework Core."""

from congruity.relations.discovery import (
    RelationDiscoveryError,
    canonical_relation_id,
    discover_spearman_relations,
    estimate_spearman_relation,
    generate_candidate_pairs,
    relation_estimates_to_frame,
)

__all__ = [
    "RelationDiscoveryError",
    "canonical_relation_id",
    "discover_spearman_relations",
    "estimate_spearman_relation",
    "generate_candidate_pairs",
    "relation_estimates_to_frame",
]
