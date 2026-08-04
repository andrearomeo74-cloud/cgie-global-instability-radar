"""
CGIE3-ID-02 experiment loader.

This module is the only CGIE3-ID-02 component permitted to read
configuration files, identity declarations and frozen feature tables
directly from the filesystem.

It does not calculate relations, metrics, alerts or scientific results.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from congruity.core import ExperimentContext
from congruity.identity import (
    IdentityDeclaration,
    IdentityDeclarationError,
    load_identity_declaration,
)


class ExperimentLoaderError(ValueError):
    """Raised when CGIE3-ID-02 inputs violate the frozen contract."""


SCRIPT_PATH = Path(__file__).resolve()

REPOSITORY_ROOT = SCRIPT_PATH.parents[4]

CONFIGURATION_PATH = (
    REPOSITORY_ROOT
    / "engines"
    / "cgie3"
    / "config"
    / "cgie3_id_02_relation_discovery.yaml"
)


def utc_now_iso() -> str:
    """Return the current UTC timestamp in canonical ISO-8601 form."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def fail(message: str) -> None:
    """Raise a loader error with a clear message."""
    raise ExperimentLoaderError(
        str(message).strip()
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like configuration value."""
    if not isinstance(value, Mapping):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Require and normalize a non-empty string."""
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
        raise ExperimentLoaderError(
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
        raise ExperimentLoaderError(
            f"Unable to hash file {path}: {exc}"
        ) from exc

    return digest.hexdigest()


def get_git_commit() -> str:
    """Return the current repository commit, when available."""
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


def load_configuration(
    path: Path = CONFIGURATION_PATH,
) -> dict[str, Any]:
    """
    Load and validate the frozen CGIE3-ID-02 YAML configuration.

    This function validates only the configuration envelope.
    Detailed scientific rules are checked by their respective stages.
    """
    configuration_path = path.resolve()

    if not configuration_path.exists():
        fail(
            "Relation-discovery configuration not found: "
            f"{configuration_path}"
        )

    if not configuration_path.is_file():
        fail(
            "Relation-discovery configuration path "
            f"is not a file: {configuration_path}"
        )

    try:
        text = configuration_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ExperimentLoaderError(
            "Unable to read relation-discovery "
            f"configuration: {exc}"
        ) from exc

    if not text.strip():
        fail(
            "Relation-discovery configuration is empty."
        )

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ExperimentLoaderError(
            "Invalid YAML in relation-discovery "
            f"configuration: {exc}"
        ) from exc

    configuration = dict(
        require_mapping(
            payload,
            "configuration root",
        )
    )

    required_sections = {
        "experiment",
        "interpretation_boundary",
        "identity_declaration",
        "inputs",
        "analysis_period",
        "feature_table",
        "candidate_generation",
        "estimability",
        "classification",
        "window_rules",
        "reproducibility",
        "safety_rules",
        "decision_boundary",
    }

    missing_sections = sorted(
        required_sections
        - set(configuration)
    )

    if missing_sections:
        fail(
            "Configuration is missing required sections: "
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

    if experiment_id != "CGIE3_ID_02":
        fail(
            "Unexpected experiment ID. "
            f"Expected CGIE3_ID_02, observed {experiment_id}."
        )

    status = require_non_empty_string(
        experiment.get("status"),
        "experiment.status",
    )

    if status != "FROZEN":
        fail(
            "CGIE3-ID-02 configuration must have "
            "status FROZEN."
        )

    protocol_version = require_non_empty_string(
        experiment.get("protocol_version"),
        "experiment.protocol_version",
    )

    if protocol_version != "CGIE3_ID_02_v1.0":
        fail(
            "Unexpected protocol version. "
            "Expected CGIE3_ID_02_v1.0, "
            f"observed {protocol_version}."
        )

    interpretation_boundary = require_mapping(
        configuration["interpretation_boundary"],
        "interpretation_boundary",
    )

    prohibited_true_claims = {
        "establishes_primary_relations",
        "establishes_indispensability",
        "establishes_causality",
        "establishes_predictive_capability",
        "establishes_earthquake_prediction",
        "allows_target_informed_selection",
        "allows_post_result_threshold_tuning",
    }

    active_prohibited_claims = sorted(
        claim
        for claim in prohibited_true_claims
        if interpretation_boundary.get(claim) is True
    )

    if active_prohibited_claims:
        fail(
            "Configuration violates the interpretation "
            "boundary: "
            + ", ".join(active_prohibited_claims)
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
    configuration: Mapping[str, Any],
) -> IdentityDeclaration:
    """
    Load and validate the frozen identity declaration.

    The declaration must match the system ID and protocol version
    frozen inside the CGIE3-ID-02 configuration.
    """
    identity_section = require_mapping(
        configuration.get("identity_declaration"),
        "identity_declaration",
    )

    declaration_path = resolve_repository_path(
        identity_section.get("file"),
        "identity_declaration.file",
    )

    expected_system_id = require_non_empty_string(
        identity_section.get("expected_system_id"),
        "identity_declaration.expected_system_id",
    )

    expected_protocol_version = require_non_empty_string(
        identity_section.get("expected_protocol_version"),
        "identity_declaration.expected_protocol_version",
    )

    if not declaration_path.exists():
        fail(
            "Identity declaration not found: "
            f"{declaration_path}"
        )

    if not declaration_path.is_file():
        fail(
            "Identity declaration path is not a file: "
            f"{declaration_path}"
        )

    try:
        identity = load_identity_declaration(
            declaration_path,
            reject_unknown_fields=True,
        )
    except IdentityDeclarationError as exc:
        raise ExperimentLoaderError(
            "Identity declaration validation failed: "
            f"{exc}"
        ) from exc

    if identity.system_id != expected_system_id:
        fail(
            "Identity declaration system ID mismatch. "
            f"Expected {expected_system_id}, "
            f"observed {identity.system_id}."
        )

    if (
        identity.protocol_version
        != expected_protocol_version
    ):
        fail(
            "Identity declaration protocol mismatch. "
            f"Expected {expected_protocol_version}, "
            f"observed {identity.protocol_version}."
        )

    declared_components = tuple(
        identity.component_ids
    )

    if len(declared_components) < 2:
        fail(
            "Identity declaration must contain "
            "at least two components."
        )

    configuration_loader = require_mapping(
        configuration.get("_loader"),
        "_loader",
    )

    mutable_loader = dict(
        configuration_loader
    )

    mutable_loader["identity_declaration_file"] = str(
        declaration_path.relative_to(
            REPOSITORY_ROOT
        )
    )

    mutable_loader["identity_declaration_sha256"] = (
        sha256_file(
            declaration_path
        )
    )

    if isinstance(configuration, dict):
        configuration["_loader"] = mutable_loader

    return identity
