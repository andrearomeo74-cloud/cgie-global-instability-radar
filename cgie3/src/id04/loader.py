"""
CGIE3-ID-04 frozen experiment loader.

This module loads and validates:

- the frozen ID-04 configuration;
- the operational identity declaration;
- the frozen feature table;
- frozen ID-02 relation classifications;
- frozen ID-03 relation states;
- frozen ID-03 family membership;
- frozen ID-03 relation families;
- frozen ID-03 official summary.

The loader enforces the upstream scientific boundary before any
ID-04 continuity calculation is permitted.

It does not:

- recompute ID-02;
- recompute ID-03;
- modify upstream classifications;
- select new relations;
- inspect earthquake targets;
- optimize thresholds or scales.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from congruity.identity.declaration import (
    load_identity_declaration,
)
from congruity.identity.models import (
    IdentityDeclaration,
)


class ID04LoaderError(ValueError):
    """Raised when frozen CGIE3-ID-04 inputs violate the contract."""


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]

CONFIGURATION_PATH = (
    REPOSITORY_ROOT
    / "engines"
    / "cgie3"
    / "config"
    / "cgie3_id_04_relational_continuity.yaml"
)


def fail(message: str) -> None:
    """Raise a normalized ID-04 loader error."""
    raise ID04LoaderError(
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


def require_string(
    value: Any,
    field_name: str,
) -> str:
    """Require a non-empty string."""
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
        raise ID04LoaderError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
        )

    return normalized


def sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 for one file."""
    if not path.exists():
        fail(
            f"Required file does not exist: {path}"
        )

    if not path.is_file():
        fail(
            f"Required path is not a file: {path}"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def resolve_repository_path(
    relative_path: str,
) -> Path:
    """Resolve and constrain one repository-relative path."""
    candidate = (
        REPOSITORY_ROOT
        / relative_path
    ).resolve()

    try:
        candidate.relative_to(
            REPOSITORY_ROOT
        )
    except ValueError as exc:
        raise ID04LoaderError(
            "Configured path escapes repository root: "
            f"{relative_path}"
        ) from exc

    return candidate


def load_yaml(
    path: Path,
    field_name: str,
) -> Mapping[str, Any]:
    """Load one non-empty YAML mapping."""
    if not path.exists():
        fail(
            f"{field_name} file not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        fail(
            f"{field_name} file is empty: {path}"
        )

    try:
        payload = yaml.safe_load(
            text
        )
    except yaml.YAMLError as exc:
        raise ID04LoaderError(
            f"Invalid YAML in {field_name}: {exc}"
        ) from exc

    return require_mapping(
        payload,
        field_name,
    )


def load_json(
    path: Path,
    field_name: str,
) -> Mapping[str, Any]:
    """Load one non-empty JSON mapping."""
    if not path.exists():
        fail(
            f"{field_name} file not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        fail(
            f"{field_name} file is empty: {path}"
        )

    try:
        payload = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise ID04LoaderError(
            f"Invalid JSON in {field_name}: {exc}"
        ) from exc

    return require_mapping(
        payload,
        field_name,
    )


def load_csv(
    path: Path,
    field_name: str,
) -> pd.DataFrame:
    """Load one required non-empty CSV."""
    if not path.exists():
        fail(
            f"{field_name} file not found: {path}"
        )

    if path.stat().st_size == 0:
        fail(
            f"{field_name} file is empty: {path}"
        )

    try:
        frame = pd.read_csv(
            path
        )
    except Exception as exc:
        raise ID04LoaderError(
            f"Unable to read {field_name}: {exc}"
        ) from exc

    if frame.empty:
        fail(
            f"{field_name} contains no rows."
        )

    return frame


def load_configuration() -> Mapping[str, Any]:
    """Load and validate the frozen ID-04 configuration."""
    configuration = load_yaml(
        CONFIGURATION_PATH,
        "ID-04 configuration",
    )

    experiment = require_mapping(
        configuration.get(
            "experiment"
        ),
        "experiment",
    )

    if experiment.get(
        "id"
    ) != "CGIE3_ID_04":
        fail(
            "Unexpected experiment ID in ID-04 configuration."
        )

    if experiment.get(
        "protocol_version"
    ) != "CGIE3_ID_04_v1.0":
        fail(
            "Unexpected ID-04 protocol version."
        )

    if experiment.get(
        "status"
    ) != "FROZEN":
        fail(
            "ID-04 configuration must remain FROZEN."
        )

    return configuration


def get_input_path(
    configuration: Mapping[str, Any],
    input_id: str,
) -> Path:
    """Resolve one frozen input path from configuration."""
    inputs = require_mapping(
        configuration.get(
            "inputs"
        ),
        "inputs",
    )

    entry = require_mapping(
        inputs.get(
            input_id
        ),
        f"inputs.{input_id}",
    )

    relative_path = require_string(
        entry.get(
            "file"
        ),
        f"inputs.{input_id}.file",
    )

    return resolve_repository_path(
        relative_path
    )


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    field_name: str,
) -> None:
    """Require a declared set of columns."""
    missing = sorted(
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        fail(
            f"{field_name} is missing columns: "
            + ", ".join(
                missing
            )
        )


def normalize_timestamp_column(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Find and normalize the frozen feature timestamp column."""
    timestamp_candidates = (
        "timestamp_utc",
        "timestamp",
        "date",
        "datetime",
    )

    timestamp_column: str | None = None

    for candidate in timestamp_candidates:
        if candidate in frame.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        fail(
            "Frozen feature table contains no recognized "
            "timestamp column."
        )

    output = frame.copy()

    output[
        timestamp_column
    ] = pd.to_datetime(
        output[
            timestamp_column
        ],
        utc=True,
        errors="coerce",
    )

    if output[
        timestamp_column
    ].isna().any():
        fail(
            "Frozen feature table contains invalid timestamps."
        )

    output = output.sort_values(
        by=timestamp_column,
        kind="stable",
    ).reset_index(
        drop=True
    )

    output.attrs[
        "timestamp_column"
    ] = timestamp_column

    return output


def validate_id02_population(
    relation_classification: pd.DataFrame,
    configuration: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate and extract the frozen 74-relation population."""
    requirements = require_mapping(
        configuration.get(
            "frozen_input_requirements"
        ),
        "frozen_input_requirements",
    )

    expected_total = require_positive_integer(
        requirements.get(
            "id02_total_relation_count"
        ),
        (
            "frozen_input_requirements."
            "id02_total_relation_count"
        ),
    )

    if len(
        relation_classification
    ) != expected_total:
        fail(
            "ID-02 relation classification must contain "
            f"{expected_total} rows; observed "
            f"{len(relation_classification)}."
        )

    validate_required_columns(
        relation_classification,
        {
            "window_id",
            "relation_id",
            "source_id",
            "target_id",
            "classification_status",
        },
        "ID-02 relation classification",
    )

    analysis_population = require_mapping(
        configuration.get(
            "analysis_population"
        ),
        "analysis_population",
    )

    accepted = analysis_population.get(
        "accepted_id02_statuses"
    )

    if not isinstance(
        accepted,
        list,
    ):
        fail(
            "analysis_population.accepted_id02_statuses "
            "must be a list."
        )

    accepted_statuses = {
        str(
            value
        ).strip()
        for value in accepted
    }

    if accepted_statuses != {
        "eligible",
        "candidate",
    }:
        fail(
            "Primary ID-04 population must remain "
            "eligible + candidate."
        )

    primary = relation_classification.loc[
        relation_classification[
            "classification_status"
        ]
        .astype(str)
        .isin(
            accepted_statuses
        )
    ].copy()

    expected_primary = require_positive_integer(
        requirements.get(
            "id03_primary_relation_count"
        ),
        (
            "frozen_input_requirements."
            "id03_primary_relation_count"
        ),
    )

    if len(
        primary
    ) != expected_primary:
        fail(
            "Frozen primary relation population must contain "
            f"{expected_primary} rows; observed "
            f"{len(primary)}."
        )

    if primary.duplicated(
        subset=[
            "window_id",
            "relation_id",
        ],
        keep=False,
    ).any():
        fail(
            "Frozen ID-04 relation population contains "
            "duplicate window-relation keys."
        )

    return primary.sort_values(
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


def validate_id03_state(
    relation_states: pd.DataFrame,
    family_membership: pd.DataFrame,
    relation_families: pd.DataFrame,
    summary: Mapping[str, Any],
    primary_population: pd.DataFrame,
    configuration: Mapping[str, Any],
) -> None:
    """Enforce the frozen upstream ID-03 scientific state."""
    expected_primary = len(
        primary_population
    )

    if len(
        relation_states
    ) != expected_primary:
        fail(
            "ID-03 relation-state table must contain "
            f"{expected_primary} rows; observed "
            f"{len(relation_states)}."
        )

    if len(
        family_membership
    ) != expected_primary:
        fail(
            "ID-03 family membership must contain "
            f"{expected_primary} rows; observed "
            f"{len(family_membership)}."
        )

    validate_required_columns(
        relation_states,
        {
            "window_id",
            "relation_id",
            "id03_state",
        },
        "ID-03 relation states",
    )

    validate_required_columns(
        family_membership,
        {
            "window_id",
            "relation_id",
            "family_id",
        },
        "ID-03 family membership",
    )

    validate_required_columns(
        relation_families,
        {
            "family_id",
            "reproducible_family",
        },
        "ID-03 relation families",
    )

    required_outcome = require_mapping(
        require_mapping(
            configuration.get(
                "frozen_input_requirements"
            ),
            "frozen_input_requirements",
        ).get(
            "id03_required_outcome"
        ),
        (
            "frozen_input_requirements."
            "id03_required_outcome"
        ),
    )

    expected_outcome = require_string(
        required_outcome.get(
            "scientific_outcome"
        ),
        (
            "frozen_input_requirements."
            "id03_required_outcome."
            "scientific_outcome"
        ),
    )

    observed_outcome = str(
        summary.get(
            "scientific_outcome"
        )
    )

    if observed_outcome != expected_outcome:
        fail(
            "ID-04 was frozen against ID-03 outcome "
            f"{expected_outcome}; observed "
            f"{observed_outcome}."
        )

    reproducible_family_count = int(
        relation_families[
            "reproducible_family"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )

    if reproducible_family_count != 0:
        fail(
            "ID-04 frozen upstream condition requires "
            "zero reproducible ID-03 families."
        )

    primary_keys = set(
        primary_population[
            [
                "window_id",
                "relation_id",
            ]
        ]
        .astype(str)
        .itertuples(
            index=False,
            name=None,
        )
    )

    state_keys = set(
        relation_states[
            [
                "window_id",
                "relation_id",
            ]
        ]
        .astype(str)
        .itertuples(
            index=False,
            name=None,
        )
    )

    membership_keys = set(
        family_membership[
            [
                "window_id",
                "relation_id",
            ]
        ]
        .astype(str)
        .itertuples(
            index=False,
            name=None,
        )
    )

    if state_keys != primary_keys:
        fail(
            "ID-03 relation states do not match the frozen "
            "74-relation population."
        )

    if membership_keys != primary_keys:
        fail(
            "ID-03 family membership does not match the frozen "
            "74-relation population."
        )


@dataclass
class ID04ExperimentContext:
    """Shared mutable runtime context for frozen ID-04 execution."""

    experiment_id: str

    configuration: Mapping[str, Any]

    identity: IdentityDeclaration

    frozen_features: pd.DataFrame

    id02_relation_classification: pd.DataFrame

    primary_population: pd.DataFrame

    id03_relation_states: pd.DataFrame

    id03_family_membership: pd.DataFrame

    id03_relation_families: pd.DataFrame

    id03_summary: Mapping[str, Any]

    provenance: dict[str, Any] = field(
        default_factory=dict
    )

    outputs: dict[str, Any] = field(
        default_factory=dict
    )

    runtime: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def register_output(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Register one stage output."""
        normalized = require_string(
            key,
            "output key",
        )

        if normalized in self.outputs:
            fail(
                f"Output already registered: {normalized}"
            )

        self.outputs[
            normalized
        ] = value

    def register_runtime(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Register one runtime value."""
        normalized = require_string(
            key,
            "runtime key",
        )

        self.runtime[
            normalized
        ] = value


def load_experiment() -> ID04ExperimentContext:
    """Load and validate every frozen CGIE3-ID-04 input."""
    configuration = load_configuration()

    identity_path = get_input_path(
        configuration,
        "identity_declaration",
    )

    feature_path = get_input_path(
        configuration,
        "frozen_features",
    )

    id02_path = get_input_path(
        configuration,
        "id02_relation_classification",
    )

    id03_states_path = get_input_path(
        configuration,
        "id03_relation_states",
    )

    id03_membership_path = get_input_path(
        configuration,
        "id03_family_membership",
    )

    id03_families_path = get_input_path(
        configuration,
        "id03_relation_families",
    )

    id03_summary_path = get_input_path(
        configuration,
        "id03_summary",
    )

    identity = load_identity_declaration(
        identity_path,
        reject_unknown_fields=True,
    )

    frozen_features = normalize_timestamp_column(
        load_csv(
            feature_path,
            "frozen feature table",
        )
    )

    relation_classification = load_csv(
        id02_path,
        "ID-02 relation classification",
    )

    primary_population = validate_id02_population(
        relation_classification,
        configuration,
    )

    relation_states = load_csv(
        id03_states_path,
        "ID-03 relation states",
    )

    family_membership = load_csv(
        id03_membership_path,
        "ID-03 family membership",
    )

    relation_families = load_csv(
        id03_families_path,
        "ID-03 relation families",
    )

    id03_summary = load_json(
        id03_summary_path,
        "ID-03 summary",
    )

    validate_id03_state(
        relation_states,
        family_membership,
        relation_families,
        id03_summary,
        primary_population,
        configuration,
    )

    provenance = {
        "loader": {
            "configuration": {
                "path":
                    str(
                        CONFIGURATION_PATH.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        CONFIGURATION_PATH
                    ),
            },

            "identity_declaration": {
                "path":
                    str(
                        identity_path.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        identity_path
                    ),
            },

            "frozen_features": {
                "path":
                    str(
                        feature_path.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        feature_path
                    ),
            },

            "id02_relation_classification": {
                "path":
                    str(
                        id02_path.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        id02_path
                    ),
            },

            "id03_relation_states": {
                "path":
                    str(
                        id03_states_path.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        id03_states_path
                    ),
            },

            "id03_family_membership": {
                "path":
                    str(
                        id03_membership_path.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        id03_membership_path
                    ),
            },

            "id03_relation_families": {
                "path":
                    str(
                        id03_families_path.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        id03_families_path
                    ),
            },

            "id03_summary": {
                "path":
                    str(
                        id03_summary_path.relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        id03_summary_path
                    ),
            },
        }
    }

    context = ID04ExperimentContext(
        experiment_id="CGIE3_ID_04",
        configuration=configuration,
        identity=identity,
        frozen_features=frozen_features,
        id02_relation_classification=(
            relation_classification
        ),
        primary_population=primary_population,
        id03_relation_states=relation_states,
        id03_family_membership=family_membership,
        id03_relation_families=relation_families,
        id03_summary=id03_summary,
        provenance=provenance,
    )

    context.register_output(
        "primary_population",
        primary_population.copy(),
    )

    context.register_runtime(
        "loader_status",
        "COMPLETED",
    )

    context.metadata[
        "timestamp_column"
    ] = frozen_features.attrs[
        "timestamp_column"
    ]

    context.metadata[
        "primary_relation_count"
    ] = int(
        len(
            primary_population
        )
    )

    context.metadata[
        "id03_scientific_outcome"
    ] = str(
        id03_summary[
            "scientific_outcome"
        ]
    )

    context.metadata[
        "id03_reproducible_family_count"
    ] = 0

    return context
