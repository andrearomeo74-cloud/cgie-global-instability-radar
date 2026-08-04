"""
Congruity Framework Core v0.1
Identity-domain data models.

This module is domain independent.

It does not know whether the observed system represents seismicity,
biology, finance, networks or another application domain.

The module defines only validated data structures used to represent:

- a declared system;
- its functional identity declaration;
- candidate and eligible relations;
- frozen reference identity;
- estimability and provenance.

No metric or alert decision is calculated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


class IdentityModelError(ValueError):
    """Raised when an identity model violates its declared contract."""


class RelationStatus(str, Enum):
    """Permitted classification of a relation."""

    CANDIDATE = "candidate"
    ELIGIBLE = "eligible"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    REJECTED = "rejected"
    NON_ESTIMABLE = "non_estimable"


class EstimabilityStatus(str, Enum):
    """Whether a quantity can be estimated from the available evidence."""

    ESTIMABLE = "estimable"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    MISSING_INPUT = "missing_input"
    NUMERICAL_FAILURE = "numerical_failure"
    EXCLUDED_BY_PROTOCOL = "excluded_by_protocol"
    NON_IDENTIFIABLE = "non_identifiable"


class RelationDirection(str, Enum):
    """Directionality of a declared relation."""

    UNDIRECTED = "undirected"
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"
    BIDIRECTIONAL = "bidirectional"


def utc_now_iso() -> str:
    """Return the current UTC time in canonical ISO-8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def require_non_empty(
    value: str,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""
    normalized = str(value).strip()

    if not normalized:
        raise IdentityModelError(
            f"{field_name} must not be empty."
        )

    return normalized


def require_unique_strings(
    values: Sequence[str],
    field_name: str,
) -> tuple[str, ...]:
    """Validate an ordered collection of unique, non-empty strings."""
    normalized = tuple(
        require_non_empty(
            value,
            field_name,
        )
        for value in values
    )

    if not normalized:
        raise IdentityModelError(
            f"{field_name} must contain at least one value."
        )

    if len(normalized) != len(set(normalized)):
        raise IdentityModelError(
            f"{field_name} contains duplicate values."
        )

    return normalized


@dataclass(frozen=True)
class ProvenanceReference:
    """
    Traceability information attached to an analytical object.

    Hash values may be empty only while constructing a draft object.
    Official frozen outputs should populate every applicable hash.
    """

    experiment_id: str
    framework_version: str
    implementation_version: str
    source_commit: str
    configuration_sha256: str
    input_sha256: Mapping[str, str]
    generated_at_utc: str = field(
        default_factory=utc_now_iso
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_id",
            require_non_empty(
                self.experiment_id,
                "experiment_id",
            ),
        )

        object.__setattr__(
            self,
            "framework_version",
            require_non_empty(
                self.framework_version,
                "framework_version",
            ),
        )

        object.__setattr__(
            self,
            "implementation_version",
            require_non_empty(
                self.implementation_version,
                "implementation_version",
            ),
        )

        object.__setattr__(
            self,
            "source_commit",
            require_non_empty(
                self.source_commit,
                "source_commit",
            ),
        )

        if not isinstance(
            self.input_sha256,
            Mapping,
        ):
            raise IdentityModelError(
                "input_sha256 must be a mapping."
            )


@dataclass(frozen=True)
class IdentityDeclaration:
    """
    Operational declaration of the system whose identity is evaluated.

    This object does not claim that the declared identity is correct.
    It records the candidate definition that an experiment will test.
    """

    system_id: str
    system_name: str
    domain: str
    functional_purpose: str
    system_boundary: str
    observation_context: str
    temporal_scales: tuple[str, ...]
    component_ids: tuple[str, ...]
    excluded_interpretations: tuple[str, ...]
    protocol_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "system_id",
            "system_name",
            "domain",
            "functional_purpose",
            "system_boundary",
            "observation_context",
            "protocol_version",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "temporal_scales",
            require_unique_strings(
                self.temporal_scales,
                "temporal_scales",
            ),
        )

        object.__setattr__(
            self,
            "component_ids",
            require_unique_strings(
                self.component_ids,
                "component_ids",
            ),
        )

        object.__setattr__(
            self,
            "excluded_interpretations",
            require_unique_strings(
                self.excluded_interpretations,
                "excluded_interpretations",
            ),
        )


@dataclass(frozen=True)
class RelationEstimate:
    """
    One estimated relation between declared system components.

    A RelationEstimate records evidence. It does not determine whether
    the relation is functionally indispensable.
    """

    relation_id: str
    source_id: str
    target_id: str
    estimator_id: str
    direction: RelationDirection
    status: RelationStatus
    estimability: EstimabilityStatus
    strength: float | None
    sign: int | None
    uncertainty: float | None
    sample_count: int
    persistence: float | None
    window_id: str
    timestamp_utc: str
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field_name in (
            "relation_id",
            "source_id",
            "target_id",
            "estimator_id",
            "window_id",
            "timestamp_utc",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        if self.source_id == self.target_id:
            raise IdentityModelError(
                "A relation cannot connect a component to itself."
            )

        if self.sample_count < 0:
            raise IdentityModelError(
                "sample_count must be non-negative."
            )

        if self.sign not in (-1, 0, 1, None):
            raise IdentityModelError(
                "sign must be -1, 0, 1 or None."
            )

        if self.persistence is not None:
            if not 0.0 <= self.persistence <= 1.0:
                raise IdentityModelError(
                    "persistence must lie inside [0, 1]."
                )

        if self.uncertainty is not None:
            if self.uncertainty < 0.0:
                raise IdentityModelError(
                    "uncertainty must be non-negative."
                )

        if (
            self.estimability
            != EstimabilityStatus.ESTIMABLE
            and self.strength is not None
        ):
            raise IdentityModelError(
                "A non-estimable relation must not carry "
                "a numerical strength."
            )

        if (
            self.estimability
            == EstimabilityStatus.ESTIMABLE
            and self.strength is None
        ):
            raise IdentityModelError(
                "An estimable relation requires a strength."
            )


@dataclass(frozen=True)
class ReferenceIdentity:
    """
    Frozen candidate identity estimated from a baseline interval.

    The reference identity contains evidence-supported relations.
    It does not assert causal or indispensable relationships.
    """

    declaration: IdentityDeclaration
    baseline_start_utc: str
    baseline_end_utc: str
    relation_estimator_id: str
    eligible_relations: tuple[RelationEstimate, ...]
    primary_relation_ids: tuple[str, ...]
    secondary_relation_ids: tuple[str, ...]
    provenance: ProvenanceReference
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_start_utc",
            require_non_empty(
                self.baseline_start_utc,
                "baseline_start_utc",
            ),
        )

        object.__setattr__(
            self,
            "baseline_end_utc",
            require_non_empty(
                self.baseline_end_utc,
                "baseline_end_utc",
            ),
        )

        object.__setattr__(
            self,
            "relation_estimator_id",
            require_non_empty(
                self.relation_estimator_id,
                "relation_estimator_id",
            ),
        )

        relation_ids = tuple(
            relation.relation_id
            for relation in self.eligible_relations
        )

        if len(relation_ids) != len(set(relation_ids)):
            raise IdentityModelError(
                "eligible_relations contains duplicate relation IDs."
            )

        known_relation_ids = set(relation_ids)

        primary_ids = require_unique_strings(
            self.primary_relation_ids,
            "primary_relation_ids",
        )

        secondary_ids = require_unique_strings(
            self.secondary_relation_ids,
            "secondary_relation_ids",
        )

        unknown_primary = (
            set(primary_ids)
            - known_relation_ids
        )

        if unknown_primary:
            raise IdentityModelError(
                "Unknown primary relation IDs: "
                + ", ".join(
                    sorted(unknown_primary)
                )
            )

        unknown_secondary = (
            set(secondary_ids)
            - known_relation_ids
        )

        if unknown_secondary:
            raise IdentityModelError(
                "Unknown secondary relation IDs: "
                + ", ".join(
                    sorted(unknown_secondary)
                )
            )

        overlap = (
            set(primary_ids)
            & set(secondary_ids)
        )

        if overlap:
            raise IdentityModelError(
                "Relations cannot be both primary and secondary: "
                + ", ".join(
                    sorted(overlap)
                )
            )

        object.__setattr__(
            self,
            "primary_relation_ids",
            primary_ids,
        )

        object.__setattr__(
            self,
            "secondary_relation_ids",
            secondary_ids,
        )


@dataclass(frozen=True)
class IdentitySnapshot:
    """
    Candidate relational identity estimated at one time and scale.

    A snapshot may contain non-estimable relations. Their status must
    remain explicit and must never be silently converted into normality.
    """

    system_id: str
    timestamp_utc: str
    window_id: str
    relation_estimates: tuple[RelationEstimate, ...]
    provenance: ProvenanceReference
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field_name in (
            "system_id",
            "timestamp_utc",
            "window_id",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_empty(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        relation_ids = [
            relation.relation_id
            for relation in self.relation_estimates
        ]

        if len(relation_ids) != len(set(relation_ids)):
            raise IdentityModelError(
                "relation_estimates contains duplicate relation IDs."
            )

        for relation in self.relation_estimates:
            if relation.window_id != self.window_id:
                raise IdentityModelError(
                    "Every relation in a snapshot must use "
                    "the snapshot window_id."
                )

            if relation.timestamp_utc != self.timestamp_utc:
                raise IdentityModelError(
                    "Every relation in a snapshot must use "
                    "the snapshot timestamp_utc."
                )


def serialize_model(
    value: Any,
) -> Any:
    """
    Convert framework models into JSON-compatible structures.

    This serializer is intentionally explicit to keep official outputs
    deterministic and independent from third-party serialization tools.
    """
    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: serialize_model(
                getattr(value, field_name)
            )
            for field_name in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): serialize_model(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [
            serialize_model(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            serialize_model(item)
            for item in value
        ]

    return value
