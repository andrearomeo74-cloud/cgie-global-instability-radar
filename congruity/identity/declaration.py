"""
Congruity Framework Core v0.1
Identity declaration loader and validator.

This module converts a YAML identity declaration into the
domain-independent IdentityDeclaration model.

It performs structural validation only.

It does not determine whether the declared functional identity is
scientifically correct. That question remains experimental.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from congruity.identity.models import (
    IdentityDeclaration,
    IdentityModelError,
)


class IdentityDeclarationError(ValueError):
    """Raised when an identity declaration cannot be loaded."""


REQUIRED_ROOT_FIELDS = {
    "system_id",
    "system_name",
    "domain",
    "functional_purpose",
    "system_boundary",
    "observation_context",
    "temporal_scales",
    "component_ids",
    "excluded_interpretations",
    "protocol_version",
}

ALLOWED_ROOT_FIELDS = REQUIRED_ROOT_FIELDS | {
    "metadata",
}


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like value."""
    if not isinstance(value, Mapping):
        raise IdentityDeclarationError(
            f"{field_name} must be a mapping."
        )

    return value


def require_string(
    value: Any,
    field_name: str,
) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str):
        raise IdentityDeclarationError(
            f"{field_name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise IdentityDeclarationError(
            f"{field_name} must not be empty."
        )

    return normalized


def require_string_sequence(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    """
    Require a non-empty ordered sequence of unique strings.

    Plain strings are rejected because otherwise each character would
    incorrectly be interpreted as one sequence element.
    """
    if isinstance(value, str):
        raise IdentityDeclarationError(
            f"{field_name} must be a YAML list, "
            "not a single string."
        )

    if not isinstance(value, (list, tuple)):
        raise IdentityDeclarationError(
            f"{field_name} must be a list."
        )

    normalized = tuple(
        require_string(
            item,
            f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )

    if not normalized:
        raise IdentityDeclarationError(
            f"{field_name} must contain at least one item."
        )

    duplicates = sorted(
        {
            item
            for item in normalized
            if normalized.count(item) > 1
        }
    )

    if duplicates:
        raise IdentityDeclarationError(
            f"{field_name} contains duplicates: "
            + ", ".join(duplicates)
        )

    return normalized


def validate_root_fields(
    payload: Mapping[str, Any],
    *,
    reject_unknown_fields: bool,
) -> None:
    """Validate required and optionally unknown root fields."""
    observed_fields = set(payload)

    missing_fields = sorted(
        REQUIRED_ROOT_FIELDS - observed_fields
    )

    if missing_fields:
        raise IdentityDeclarationError(
            "Identity declaration is missing required fields: "
            + ", ".join(missing_fields)
        )

    if reject_unknown_fields:
        unknown_fields = sorted(
            observed_fields - ALLOWED_ROOT_FIELDS
        )

        if unknown_fields:
            raise IdentityDeclarationError(
                "Identity declaration contains unknown fields: "
                + ", ".join(unknown_fields)
            )


def build_identity_declaration(
    payload: Mapping[str, Any],
    *,
    reject_unknown_fields: bool = True,
) -> IdentityDeclaration:
    """
    Construct an IdentityDeclaration from a validated mapping.

    The function performs no file-system access and may therefore be
    reused by tests, APIs and future domain adapters.
    """
    payload = require_mapping(
        payload,
        "identity declaration",
    )

    validate_root_fields(
        payload,
        reject_unknown_fields=reject_unknown_fields,
    )

    try:
        declaration = IdentityDeclaration(
            system_id=require_string(
                payload["system_id"],
                "system_id",
            ),
            system_name=require_string(
                payload["system_name"],
                "system_name",
            ),
            domain=require_string(
                payload["domain"],
                "domain",
            ),
            functional_purpose=require_string(
                payload["functional_purpose"],
                "functional_purpose",
            ),
            system_boundary=require_string(
                payload["system_boundary"],
                "system_boundary",
            ),
            observation_context=require_string(
                payload["observation_context"],
                "observation_context",
            ),
            temporal_scales=require_string_sequence(
                payload["temporal_scales"],
                "temporal_scales",
            ),
            component_ids=require_string_sequence(
                payload["component_ids"],
                "component_ids",
            ),
            excluded_interpretations=(
                require_string_sequence(
                    payload["excluded_interpretations"],
                    "excluded_interpretations",
                )
            ),
            protocol_version=require_string(
                payload["protocol_version"],
                "protocol_version",
            ),
        )
    except IdentityModelError as exc:
        raise IdentityDeclarationError(
            "Identity declaration violates the "
            f"framework model contract: {exc}"
        ) from exc

    return declaration


def load_identity_declaration(
    path: str | Path,
    *,
    reject_unknown_fields: bool = True,
) -> IdentityDeclaration:
    """
    Load an IdentityDeclaration from a YAML file.

    Parameters
    ----------
    path:
        YAML file containing the declaration.

    reject_unknown_fields:
        When True, undeclared root fields cause validation failure.
        Official frozen experiments should keep this enabled.
    """
    declaration_path = Path(path).expanduser().resolve()

    if not declaration_path.exists():
        raise IdentityDeclarationError(
            "Identity declaration file not found: "
            f"{declaration_path}"
        )

    if not declaration_path.is_file():
        raise IdentityDeclarationError(
            "Identity declaration path is not a file: "
            f"{declaration_path}"
        )

    try:
        text = declaration_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise IdentityDeclarationError(
            "Unable to read identity declaration: "
            f"{declaration_path}: {exc}"
        ) from exc

    if not text.strip():
        raise IdentityDeclarationError(
            "Identity declaration file is empty: "
            f"{declaration_path}"
        )

    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise IdentityDeclarationError(
            "Invalid YAML in identity declaration "
            f"{declaration_path}: {exc}"
        ) from exc

    if payload is None:
        raise IdentityDeclarationError(
            "Identity declaration contains no YAML data: "
            f"{declaration_path}"
        )

    return build_identity_declaration(
        require_mapping(
            payload,
            "identity declaration root",
        ),
        reject_unknown_fields=reject_unknown_fields,
  )
