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
