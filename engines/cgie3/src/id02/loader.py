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
