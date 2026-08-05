"""
CGIE3-ID-03 frozen input loader.

This module is the only ID-03 component permitted to read:

- the frozen ID-03 configuration;
- the identity declaration;
- the frozen ID-02 configuration;
- the frozen ID-02 outputs;
- the frozen CF_RETRO_01 feature table.

It performs structural and lineage validation only.

It does not:

- modify ID-02 results;
- recompute ID-02 classifications;
- use evaluation-period data;
- use target-event labels;
- identify relational families;
- select representative candidates.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from congruity.identity import (
    IdentityDeclaration,
    IdentityDeclarationError,
    load_identity_declaration,
)


class ID03LoaderError(ValueError):
    """Raised when an ID-03 frozen input violates its contract."""


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]

CONFIGURATION_PATH = (
    REPOSITORY_ROOT
    / "engines"
    / "cgie3"
    / "config"
    / "cgie3_id_03_family_audit.yaml"
)


@dataclass
class ID03ExperimentContext:
    """
    Shared mutable execution context for CGIE3-ID-03.

    Scientific outputs are added by later stages without rewriting
    the frozen input objects.
    """

    experiment_id: str
    configuration: dict[str, Any]
    identity: IdentityDeclaration
    id02_configuration: dict[str, Any]
    frozen_features: pd.DataFrame
    candidate_relations: pd.DataFrame
    block_relations: pd.DataFrame
    bootstrap_relations: pd.DataFrame
    relation_classification: pd.DataFrame
    equivalence_flags: pd.DataFrame
    id02_summary: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    runtime: dict[str, Any]
    outputs: dict[str, Any]

    def register_output(
        self,
        name: str,
        value: Any,
    ) -> None:
        """Register one generated ID-03 output."""
        self.outputs[name] = value

    def register_runtime(
        self,
        name: str,
        value: Any,
    ) -> None:
        """Register one runtime-stage value."""
        self.runtime[name] = value

    def register_metadata(
        self,
        name: str,
        value: Any,
    ) -> None:
        """Register descriptive experiment metadata."""
        self.metadata[name] = value

    def register_provenance(
        self,
        name: str,
        value: Any,
    ) -> None:
        """Register input or execution provenance."""
        self.provenance[name] = value


def fail(message: str) -> None:
    """Raise a loader error with a normalized message."""
    raise ID03LoaderError(
        str(message).strip()
    )


def utc_now_iso() -> str:
    """Return current UTC time in canonical ISO-8601 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like value."""
    if not isinstance(value, Mapping):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Require and normalize one non-empty string."""
    if not isinstance(value, str):
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
) -> tuple[str, ...]:
    """Require an ordered list of unique non-empty strings."""
    if not isinstance(
        value,
        (list, tuple),
    ):
        fail(
            f"{field_name} must be a list."
        )

    normalized = tuple(
        require_non_empty_string(
            item,
            f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )

    if not normalized:
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


def resolve_repository_path(
    value: Any,
    field_name: str,
) -> Path:
    """
    Resolve a repository-relative path and prevent directory escape.
    """
    relative_path = Path(
        require_non_empty_string(
            value,
            field_name,
        )
    )

    if relative_path.is_absolute():
        fail(
            f"{field_name} must be repository-relative."
        )

    resolved = (
        REPOSITORY_ROOT
        / relative_path
    ).resolve()

    try:
        resolved.relative_to(
            REPOSITORY_ROOT
        )
    except ValueError as exc:
        raise ID03LoaderError(
            f"{field_name} escapes the repository root."
        ) from exc

    return resolved


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one file."""
    if not path.exists():
        fail(
            f"File not found for hashing: {path}"
        )

    if not path.is_file():
        fail(
            f"Path is not a file: {path}"
        )

    digest = hashlib.sha256()

    try:
        with path.open("rb") as handle:
            for block in iter(
                lambda: handle.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(block)
    except OSError as exc:
        raise ID03LoaderError(
            f"Unable to hash {path}: {exc}"
        ) from exc

    return digest.hexdigest()


def get_git_commit() -> str:
    """Return the current Git commit when available."""
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return "UNAVAILABLE"

    commit = result.stdout.strip()

    return (
        commit
        if commit
        else "UNAVAILABLE"
    )


def read_yaml(
    path: Path,
    field_name: str,
) -> dict[str, Any]:
    """Read one YAML file as a mapping."""
    if not path.exists():
        fail(
            f"{field_name} not found: {path}"
        )

    if not path.is_file():
        fail(
            f"{field_name} is not a file: {path}"
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ID03LoaderError(
            f"Unable to read {field_name}: {exc}"
        ) from exc

    if not text.strip():
        fail(
            f"{field_name} is empty."
        )

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ID03LoaderError(
            f"Invalid YAML in {field_name}: {exc}"
        ) from exc

    return dict(
        require_mapping(
            payload,
            field_name,
        )
    )


def read_json(
    path: Path,
    field_name: str,
) -> dict[str, Any]:
    """Read one strict JSON file as a mapping."""
    if not path.exists():
        fail(
            f"{field_name} not found: {path}"
        )

    if not path.is_file():
        fail(
            f"{field_name} is not a file: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ID03LoaderError(
            f"Unable to read {field_name}: {exc}"
        ) from exc

    return dict(
        require_mapping(
            payload,
            field_name,
        )
    )


def read_csv(
    path: Path,
    field_name: str,
) -> pd.DataFrame:
    """Read one non-empty frozen CSV table."""
    if not path.exists():
        fail(
            f"{field_name} not found: {path}"
        )

    if not path.is_file():
        fail(
            f"{field_name} is not a file: {path}"
        )

    try:
        frame = pd.read_csv(
            path,
            encoding="utf-8",
        )
    except Exception as exc:
        raise ID03LoaderError(
            f"Unable to read {field_name}: {exc}"
        ) from exc

    if frame.empty:
        fail(
            f"{field_name} is empty."
        )

    return frame

def load_configuration(
    path: Path = CONFIGURATION_PATH,
) -> dict[str, Any]:
    """
    Load and validate the frozen CGIE3-ID-03 configuration.
    """
    configuration_path = path.resolve()

    configuration = read_yaml(
        configuration_path,
        "CGIE3-ID-03 configuration",
    )

    required_sections = {
        "experiment",
        "interpretation_boundary",
        "inputs",
        "analysis_population",
        "analysis_period",
        "feature_table",
        "feature_dependency_registry",
        "multiscale_audit",
        "overlap_audit",
        "effective_temporal_support",
        "null_controls",
        "conditional_redundancy",
        "family_audit",
        "representative_selection",
        "id03_states",
        "scientific_success",
        "required_outputs",
        "reproducibility",
        "safety_rules",
        "prohibited_practices",
        "decision_boundary",
        "advancement_to_id04",
    }

    missing_sections = sorted(
        required_sections
        - set(configuration)
    )

    if missing_sections:
        fail(
            "ID-03 configuration is missing sections: "
            + ", ".join(missing_sections)
        )

    experiment = require_mapping(
        configuration["experiment"],
        "experiment",
    )

    experiment_id = require_non_empty_string(
        experiment.get("id"),
        "experiment.id",
    )

    if experiment_id != "CGIE3_ID_03":
        fail(
            "Unexpected ID-03 experiment ID. "
            f"Observed: {experiment_id}"
        )

    protocol_version = require_non_empty_string(
        experiment.get("protocol_version"),
        "experiment.protocol_version",
    )

    if protocol_version != "CGIE3_ID_03_v1.0":
        fail(
            "Unexpected ID-03 protocol version. "
            f"Observed: {protocol_version}"
        )

    status = require_non_empty_string(
        experiment.get("status"),
        "experiment.status",
    )

    if status != "FROZEN":
        fail(
            "CGIE3-ID-03 configuration must have "
            "status FROZEN."
        )

    interpretation_boundary = require_mapping(
        configuration[
            "interpretation_boundary"
        ],
        "interpretation_boundary",
    )

    prohibited_true_claims = {
        "modifies_id02_statuses",
        "establishes_primary_relations",
        "establishes_indispensability",
        "establishes_minimum_identity_core",
        "establishes_causality",
        "establishes_predictive_capability",
        "establishes_earthquake_prediction",
        "establishes_universal_transferability",
    }

    invalid_claims = sorted(
        claim
        for claim in prohibited_true_claims
        if interpretation_boundary.get(
            claim
        )
        is True
    )

    if invalid_claims:
        fail(
            "ID-03 interpretation boundary violated: "
            + ", ".join(invalid_claims)
        )

    analysis_population = require_mapping(
        configuration[
            "analysis_population"
        ],
        "analysis_population",
    )

    expected_counts = require_mapping(
        analysis_population[
            "expected_counts"
        ],
        "analysis_population.expected_counts",
    )

    frozen_counts = {
        "eligible": 45,
        "candidate": 29,
        "rejected": 62,
        "non_estimable": 0,
        "primary_audit_population": 74,
        "total_relations": 136,
    }

    observed_counts = {
        key: int(
            expected_counts.get(key)
        )
        for key in frozen_counts
    }

    if observed_counts != frozen_counts:
        fail(
            "ID-03 frozen analysis-population "
            "counts do not match ID-02."
        )

    configuration["_loader"] = {
        "configuration_file":
            str(
                configuration_path.relative_to(
                    REPOSITORY_ROOT
                )
            ),
        "configuration_sha256":
            sha256_file(
                configuration_path
            ),
        "loaded_at_utc":
            utc_now_iso(),
    }

    return configuration


def load_identity(
    configuration: dict[str, Any],
) -> IdentityDeclaration:
    """
    Load the frozen identity declaration referenced by ID-03.
    """
    inputs = require_mapping(
        configuration["inputs"],
        "inputs",
    )

    identity_section = require_mapping(
        inputs["identity_declaration"],
        "inputs.identity_declaration",
    )

    identity_path = resolve_repository_path(
        identity_section["file"],
        "inputs.identity_declaration.file",
    )

    expected_system_id = (
        require_non_empty_string(
            identity_section[
                "expected_system_id"
            ],
            (
                "inputs.identity_declaration."
                "expected_system_id"
            ),
        )
    )

    expected_protocol_version = (
        require_non_empty_string(
            identity_section[
                "expected_protocol_version"
            ],
            (
                "inputs.identity_declaration."
                "expected_protocol_version"
            ),
        )
    )

    try:
        identity = load_identity_declaration(
            identity_path,
            reject_unknown_fields=True,
        )
    except IdentityDeclarationError as exc:
        raise ID03LoaderError(
            "ID-03 identity declaration "
            f"validation failed: {exc}"
        ) from exc

    if identity.system_id != expected_system_id:
        fail(
            "ID-03 identity system ID mismatch. "
            f"Expected {expected_system_id}, "
            f"observed {identity.system_id}."
        )

    if (
        identity.protocol_version
        != expected_protocol_version
    ):
        fail(
            "ID-03 identity protocol mismatch. "
            f"Expected {expected_protocol_version}, "
            f"observed {identity.protocol_version}."
        )

    loader_metadata = dict(
        require_mapping(
            configuration["_loader"],
            "_loader",
        )
    )

    loader_metadata.update(
        {
            "identity_declaration_file":
                str(
                    identity_path.relative_to(
                        REPOSITORY_ROOT
                    )
                ),
            "identity_declaration_sha256":
                sha256_file(
                    identity_path
                ),
        }
    )

    configuration["_loader"] = (
        loader_metadata
    )

    return identity


def load_id02_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    Load and validate the frozen CGIE3-ID-02 configuration.
    """
    inputs = require_mapping(
        configuration["inputs"],
        "inputs",
    )

    id02_section = require_mapping(
        inputs["id02_configuration"],
        "inputs.id02_configuration",
    )

    id02_path = resolve_repository_path(
        id02_section["file"],
        "inputs.id02_configuration.file",
    )

    id02_configuration = read_yaml(
        id02_path,
        "CGIE3-ID-02 configuration",
    )

    experiment = require_mapping(
        id02_configuration.get(
            "experiment"
        ),
        "ID-02 experiment",
    )

    expected_experiment_id = (
        require_non_empty_string(
            id02_section[
                "expected_experiment_id"
            ],
            (
                "inputs.id02_configuration."
                "expected_experiment_id"
            ),
        )
    )

    expected_protocol_version = (
        require_non_empty_string(
            id02_section[
                "expected_protocol_version"
            ],
            (
                "inputs.id02_configuration."
                "expected_protocol_version"
            ),
        )
    )

    expected_status = (
        require_non_empty_string(
            id02_section[
                "expected_status"
            ],
            (
                "inputs.id02_configuration."
                "expected_status"
            ),
        )
    )

    if (
        experiment.get("id")
        != expected_experiment_id
    ):
        fail(
            "Loaded ID-02 configuration has "
            "an unexpected experiment ID."
        )

    if (
        experiment.get("protocol_version")
        != expected_protocol_version
    ):
        fail(
            "Loaded ID-02 configuration has "
            "an unexpected protocol version."
        )

    if (
        experiment.get("status")
        != expected_status
    ):
        fail(
            "Loaded ID-02 configuration does not "
            "have the expected frozen status."
        )

    loader_metadata = dict(
        require_mapping(
            configuration["_loader"],
            "_loader",
        )
    )

    loader_metadata.update(
        {
            "id02_configuration_file":
                str(
                    id02_path.relative_to(
                        REPOSITORY_ROOT
                    )
                ),
            "id02_configuration_sha256":
                sha256_file(
                    id02_path
                ),
        }
    )

    configuration["_loader"] = (
        loader_metadata
    )

    return id02_configuration

def load_id02_outputs(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    Load and validate the frozen CGIE3-ID-02 output package.

    The original ID-02 classifications are preserved exactly.
    """
    inputs = require_mapping(
        configuration["inputs"],
        "inputs",
    )

    csv_input_keys = (
        "id02_candidate_relations",
        "id02_block_relations",
        "id02_bootstrap_relations",
        "id02_relation_classification",
        "id02_equivalence_flags",
    )

    loaded_frames: dict[str, pd.DataFrame] = {}
    input_provenance: dict[str, Any] = {}

    for input_key in csv_input_keys:
        input_section = require_mapping(
            inputs[input_key],
            f"inputs.{input_key}",
        )

        input_path = resolve_repository_path(
            input_section["file"],
            f"inputs.{input_key}.file",
        )

        frame = read_csv(
            input_path,
            input_key,
        )

        loaded_frames[
            input_key
        ] = frame

        input_provenance[
            input_key
        ] = {
            "file":
                str(
                    input_path.relative_to(
                        REPOSITORY_ROOT
                    )
                ),
            "sha256":
                sha256_file(
                    input_path
                ),
            "row_count":
                int(len(frame)),
            "column_count":
                int(len(frame.columns)),
        }

    summary_section = require_mapping(
        inputs["id02_summary"],
        "inputs.id02_summary",
    )

    summary_path = resolve_repository_path(
        summary_section["file"],
        "inputs.id02_summary.file",
    )

    id02_summary = read_json(
        summary_path,
        "CGIE3-ID-02 summary",
    )

    input_provenance[
        "id02_summary"
    ] = {
        "file":
            str(
                summary_path.relative_to(
                    REPOSITORY_ROOT
                )
            ),
        "sha256":
            sha256_file(
                summary_path
            ),
    }

    candidate_relations = loaded_frames[
        "id02_candidate_relations"
    ]

    relation_classification = loaded_frames[
        "id02_relation_classification"
    ]

    equivalence_flags = loaded_frames[
        "id02_equivalence_flags"
    ]

    block_relations = loaded_frames[
        "id02_block_relations"
    ]

    bootstrap_relations = loaded_frames[
        "id02_bootstrap_relations"
    ]

    if len(candidate_relations) != 136:
        fail(
            "ID-02 candidate relations must contain "
            f"136 rows; observed {len(candidate_relations)}."
        )

    if len(relation_classification) != 136:
        fail(
            "ID-02 relation classification must contain "
            f"136 rows; observed "
            f"{len(relation_classification)}."
        )

    if len(equivalence_flags) != 136:
        fail(
            "ID-02 equivalence flags must contain "
            f"136 rows; observed {len(equivalence_flags)}."
        )

    if len(bootstrap_relations) != 68000:
        fail(
            "ID-02 bootstrap relations must contain "
            f"68000 rows; observed "
            f"{len(bootstrap_relations)}."
        )

    required_relation_key_columns = {
        "window_id",
        "relation_id",
        "source_id",
        "target_id",
    }

    for frame_name, frame in (
        (
            "candidate_relations",
            candidate_relations,
        ),
        (
            "relation_classification",
            relation_classification,
        ),
        (
            "equivalence_flags",
            equivalence_flags,
        ),
    ):
        missing_columns = sorted(
            required_relation_key_columns
            - set(frame.columns)
        )

        if missing_columns:
            fail(
                f"{frame_name} is missing relation-key "
                "columns: "
                + ", ".join(
                    missing_columns
                )
            )

        duplicate_mask = frame.duplicated(
            subset=[
                "window_id",
                "relation_id",
            ],
            keep=False,
        )

        if duplicate_mask.any():
            fail(
                f"{frame_name} contains duplicate "
                "window-relation keys."
            )

    classification_status_column = (
        "classification_status"
    )

    if (
        classification_status_column
        not in relation_classification.columns
    ):
        fail(
            "ID-02 relation classification is missing "
            "classification_status."
        )

    observed_status_counts = (
        relation_classification[
            classification_status_column
        ]
        .astype(str)
        .value_counts()
        .reindex(
            [
                "eligible",
                "candidate",
                "rejected",
                "non_estimable",
            ],
            fill_value=0,
        )
        .to_dict()
    )

    observed_status_counts = {
        str(key): int(value)
        for key, value
        in observed_status_counts.items()
    }

    expected_counts = require_mapping(
        require_mapping(
            configuration[
                "analysis_population"
            ],
            "analysis_population",
        )[
            "expected_counts"
        ],
        "analysis_population.expected_counts",
    )

    expected_status_counts = {
        "eligible":
            int(
                expected_counts[
                    "eligible"
                ]
            ),
        "candidate":
            int(
                expected_counts[
                    "candidate"
                ]
            ),
        "rejected":
            int(
                expected_counts[
                    "rejected"
                ]
            ),
        "non_estimable":
            int(
                expected_counts[
                    "non_estimable"
                ]
            ),
    }

    if (
        observed_status_counts
        != expected_status_counts
    ):
        fail(
            "Loaded ID-02 status counts do not match "
            "the frozen ID-03 population contract. "
            f"Observed: {observed_status_counts}. "
            f"Expected: {expected_status_counts}."
        )

    primary_statuses = set(
        require_string_list(
            require_mapping(
                configuration[
                    "analysis_population"
                ],
                "analysis_population",
            )[
                "primary_statuses"
            ],
            "analysis_population.primary_statuses",
        )
    )

    primary_population = (
        relation_classification.loc[
            relation_classification[
                classification_status_column
            ].isin(
                primary_statuses
            )
        ]
        .copy()
    )

    expected_primary_count = int(
        expected_counts[
            "primary_audit_population"
        ]
    )

    if len(
        primary_population
    ) != expected_primary_count:
        fail(
            "ID-03 primary audit population must contain "
            f"{expected_primary_count} relations; observed "
            f"{len(primary_population)}."
        )

    candidate_keys = set(
        zip(
            candidate_relations[
                "window_id"
            ].astype(str),
            candidate_relations[
                "relation_id"
            ].astype(str),
        )
    )

    classification_keys = set(
        zip(
            relation_classification[
                "window_id"
            ].astype(str),
            relation_classification[
                "relation_id"
            ].astype(str),
        )
    )

    equivalence_keys = set(
        zip(
            equivalence_flags[
                "window_id"
            ].astype(str),
            equivalence_flags[
                "relation_id"
            ].astype(str),
        )
    )

    if (
        candidate_keys
        != classification_keys
        or candidate_keys
        != equivalence_keys
    ):
        fail(
            "ID-02 candidate, classification and "
            "equivalence relation keys are not identical."
        )

    if (
        id02_summary.get(
            "experiment_id"
        )
        != "CGIE3_ID_02"
    ):
        fail(
            "ID-02 summary contains an unexpected "
            "experiment identifier."
        )

    if (
        id02_summary.get(
            "technical_status"
        )
        != "COMPLETED"
    ):
        fail(
            "ID-02 summary is not technically completed."
        )

    summary_status_counts = (
        id02_summary.get(
            "status_counts"
        )
    )

    if not isinstance(
        summary_status_counts,
        Mapping,
    ):
        fail(
            "ID-02 summary is missing status_counts."
        )

    normalized_summary_counts = {
        status: int(
            summary_status_counts.get(
                status,
                0,
            )
        )
        for status in (
            "eligible",
            "candidate",
            "rejected",
            "non_estimable",
        )
    }

    if (
        normalized_summary_counts
        != observed_status_counts
    ):
        fail(
            "ID-02 summary status counts differ from "
            "the loaded classification table."
        )

    loader_metadata = dict(
        require_mapping(
            configuration["_loader"],
            "_loader",
        )
    )

    loader_metadata[
        "id02_outputs"
    ] = input_provenance

    loader_metadata[
        "id02_status_counts"
    ] = observed_status_counts

    loader_metadata[
        "primary_audit_population_count"
    ] = int(
        len(primary_population)
    )

    loader_metadata[
        "block_relation_row_count"
    ] = int(
        len(block_relations)
    )

    configuration["_loader"] = (
        loader_metadata
    )

    return {
        "candidate_relations":
            candidate_relations,

        "block_relations":
            block_relations,

        "bootstrap_relations":
            bootstrap_relations,

        "relation_classification":
            relation_classification,

        "equivalence_flags":
            equivalence_flags,

        "id02_summary":
            id02_summary,

        "primary_population":
            primary_population,
    }

def load_frozen_features(
    configuration: dict[str, Any],
) -> pd.DataFrame:
    """
    Load and structurally validate the frozen CF_RETRO_01 feature table.

    No ID-03 scientific transformation is performed here.
    """
    inputs = require_mapping(
        configuration["inputs"],
        "inputs",
    )

    feature_section = require_mapping(
        inputs["frozen_features"],
        "inputs.frozen_features",
    )

    feature_path = resolve_repository_path(
        feature_section["file"],
        "inputs.frozen_features.file",
    )

    frozen_features = read_csv(
        feature_path,
        "CF_RETRO_01 frozen features",
    )

    feature_contract = require_mapping(
        configuration["feature_table"],
        "feature_table",
    )

    timestamp_column = (
        require_non_empty_string(
            feature_contract[
                "timestamp_column"
            ],
            "feature_table.timestamp_column",
        )
    )

    window_column = (
        require_non_empty_string(
            feature_contract[
                "window_column"
            ],
            "feature_table.window_column",
        )
    )

    required_windows = require_string_list(
        feature_contract[
            "required_windows"
        ],
        "feature_table.required_windows",
    )

    required_components = require_string_list(
        feature_contract[
            "required_components"
        ],
        "feature_table.required_components",
    )

    required_columns = {
        timestamp_column,
        window_column,
        *required_components,
    }

    missing_columns = sorted(
        required_columns
        - set(
            frozen_features.columns
        )
    )

    if missing_columns:
        fail(
            "Frozen feature table is missing columns: "
            + ", ".join(missing_columns)
        )

    frozen_features[
        timestamp_column
    ] = pd.to_datetime(
        frozen_features[
            timestamp_column
        ],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    invalid_timestamp_count = int(
        frozen_features[
            timestamp_column
        ]
        .isna()
        .sum()
    )

    if invalid_timestamp_count > 0:
        fail(
            "Frozen feature table contains "
            f"{invalid_timestamp_count} invalid timestamps "
            f"in {timestamp_column}."
        )

    frozen_features[
        window_column
    ] = (
        frozen_features[
            window_column
        ]
        .astype(str)
        .str.strip()
    )

    if (
        frozen_features[
            window_column
        ]
        == ""
    ).any():
        fail(
            "Frozen feature table contains "
            "an empty window identifier."
        )

    observed_windows = set(
        frozen_features[
            window_column
        ].unique()
    )

    expected_windows = set(
        required_windows
    )

    missing_windows = sorted(
        expected_windows
        - observed_windows
    )

    if missing_windows:
        fail(
            "Frozen feature table is missing windows: "
            + ", ".join(missing_windows)
        )

    unexpected_windows = sorted(
        observed_windows
        - expected_windows
    )

    if unexpected_windows:
        fail(
            "Frozen feature table contains "
            "unexpected windows: "
            + ", ".join(unexpected_windows)
        )

    duplicate_mask = (
        frozen_features.duplicated(
            subset=[
                timestamp_column,
                window_column,
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():
        fail(
            "Frozen feature table contains duplicate "
            "timestamp-window keys."
        )

    analysis_period = require_mapping(
        configuration[
            "analysis_period"
        ],
        "analysis_period",
    )

    baseline = require_mapping(
        analysis_period[
            "baseline"
        ],
        "analysis_period.baseline",
    )

    baseline_start = pd.Timestamp(
        baseline[
            "start_utc"
        ]
    )

    baseline_end = pd.Timestamp(
        baseline[
            "end_utc"
        ]
    )

    if baseline_start.tzinfo is None:
        baseline_start = (
            baseline_start.tz_localize(
                "UTC"
            )
        )
    else:
        baseline_start = (
            baseline_start.tz_convert(
                "UTC"
            )
        )

    if baseline_end.tzinfo is None:
        baseline_end = (
            baseline_end.tz_localize(
                "UTC"
            )
        )
    else:
        baseline_end = (
            baseline_end.tz_convert(
                "UTC"
            )
        )

    if baseline_start >= baseline_end:
        fail(
            "ID-03 baseline start must precede "
            "baseline end."
        )

    if (
        analysis_period.get(
            "use_evaluation_interval"
        )
        is not False
    ):
        fail(
            "ID-03 must not use the evaluation interval."
        )

    if (
        analysis_period.get(
            "use_target_event_labels"
        )
        is not False
    ):
        fail(
            "ID-03 must not use target-event labels."
        )

    if (
        analysis_period.get(
            "future_observations_allowed"
        )
        is not False
    ):
        fail(
            "ID-03 must not use future observations."
        )

    baseline_mask = (
        (
            frozen_features[
                timestamp_column
            ]
            >= baseline_start
        )
        & (
            frozen_features[
                timestamp_column
            ]
            <= baseline_end
        )
    )

    baseline_row_count = int(
        baseline_mask.sum()
    )

    if baseline_row_count == 0:
        fail(
            "Frozen feature table contains no rows "
            "inside the ID-03 baseline interval."
        )

    frozen_features = (
        frozen_features.sort_values(
            by=[
                window_column,
                timestamp_column,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    loader_metadata = dict(
        require_mapping(
            configuration[
                "_loader"
            ],
            "_loader",
        )
    )

    loader_metadata.update(
        {
            "frozen_features_file":
                str(
                    feature_path.relative_to(
                        REPOSITORY_ROOT
                    )
                ),

            "frozen_features_sha256":
                sha256_file(
                    feature_path
                ),

            "frozen_features_row_count":
                int(
                    len(
                        frozen_features
                    )
                ),

            "frozen_features_column_count":
                int(
                    len(
                        frozen_features.columns
                    )
                ),

            "baseline_feature_row_count":
                baseline_row_count,

            "timestamp_column":
                timestamp_column,

            "window_column":
                window_column,

            "observed_windows":
                sorted(
                    observed_windows
                ),
        }
    )

    configuration[
        "_loader"
    ] = loader_metadata

    return frozen_features


def collect_runtime_information() -> dict[str, Any]:
    """
    Collect descriptive runtime information.

    These values never influence scientific calculations.
    """
    return {
        "generated_at_utc":
            utc_now_iso(),

        "python_version":
            platform.python_version(),

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "git_commit":
            get_git_commit(),
    }


def build_context(
    configuration: dict[str, Any],
    identity: IdentityDeclaration,
    id02_configuration: dict[str, Any],
    frozen_features: pd.DataFrame,
    id02_outputs: Mapping[str, Any],
) -> ID03ExperimentContext:
    """
    Build the shared CGIE3-ID-03 experiment context.
    """
    experiment = require_mapping(
        configuration[
            "experiment"
        ],
        "experiment",
    )

    primary_population = (
        id02_outputs[
            "primary_population"
        ]
    )

    if not isinstance(
        primary_population,
        pd.DataFrame,
    ):
        fail(
            "ID-03 primary population must be "
            "a pandas DataFrame."
        )

    context = ID03ExperimentContext(
        experiment_id=
            require_non_empty_string(
                experiment[
                    "id"
                ],
                "experiment.id",
            ),

        configuration=
            configuration,

        identity=
            identity,

        id02_configuration=
            id02_configuration,

        frozen_features=
            frozen_features.copy(),

        candidate_relations=
            id02_outputs[
                "candidate_relations"
            ].copy(),

        block_relations=
            id02_outputs[
                "block_relations"
            ].copy(),

        bootstrap_relations=
            id02_outputs[
                "bootstrap_relations"
            ].copy(),

        relation_classification=
            id02_outputs[
                "relation_classification"
            ].copy(),

        equivalence_flags=
            id02_outputs[
                "equivalence_flags"
            ].copy(),

        id02_summary=
            dict(
                id02_outputs[
                    "id02_summary"
                ]
            ),

        metadata={},

        provenance={},

        runtime={},

        outputs={},
    )

    context.register_output(
        "primary_population",
        primary_population.copy(),
    )

    context.register_metadata(
        "total_id02_relation_count",
        int(
            len(
                context.relation_classification
            )
        ),
    )

    context.register_metadata(
        "primary_audit_population_count",
        int(
            len(
                primary_population
            )
        ),
    )

    context.register_metadata(
        "id02_status_counts",
        dict(
            configuration[
                "_loader"
            ][
                "id02_status_counts"
            ]
        ),
    )

    context.register_metadata(
        "frozen_feature_row_count",
        int(
            len(
                frozen_features
            )
        ),
    )

    feature_contract = require_mapping(
        configuration[
            "feature_table"
        ],
        "feature_table",
    )

    window_column = (
        require_non_empty_string(
            feature_contract[
                "window_column"
            ],
            "feature_table.window_column",
        )
    )

    context.register_metadata(
        "windows",
        sorted(
            frozen_features[
                window_column
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    context.register_provenance(
        "loader",
        dict(
            configuration[
                "_loader"
            ]
        ),
    )

    context.register_runtime(
        "environment",
        collect_runtime_information(),
    )

    context.register_runtime(
        "loader_status",
        "COMPLETED",
    )

    return context


def load_experiment() -> ID03ExperimentContext:
    """
    Load the complete frozen CGIE3-ID-03 experiment context.
    """
    configuration = load_configuration()

    identity = load_identity(
        configuration
    )

    id02_configuration = (
        load_id02_configuration(
            configuration
        )
    )

    id02_outputs = load_id02_outputs(
        configuration
    )

    frozen_features = (
        load_frozen_features(
            configuration
        )
    )

    context = build_context(
        configuration=
            configuration,

        identity=
            identity,

        id02_configuration=
            id02_configuration,

        frozen_features=
            frozen_features,

        id02_outputs=
            id02_outputs,
    )

    return context
 
    
