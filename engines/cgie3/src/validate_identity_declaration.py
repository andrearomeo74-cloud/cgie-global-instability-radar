#!/usr/bin/env python3
"""
CGIE-3 identity declaration validator.

This script loads the Campi Flegrei identity declaration through
the domain-independent Congruity Core and writes a deterministic
audit output.

Run from the repository root:

    python engines/cgie3/src/validate_identity_declaration.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]

IDENTITY_DECLARATION_PATH = (
    REPOSITORY_ROOT
    / "engines"
    / "cgie3"
    / "config"
    / "campi_flegrei_identity.yaml"
)

OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "outputs"
)

AUDIT_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "CGIE3_ID_01_identity_declaration_audit.json"
)

AUDIT_MD_PATH = (
    OUTPUT_DIRECTORY
    / "CGIE3_ID_01_identity_declaration_audit.md"
)


def fail(message: str) -> None:
    """Terminate execution with a clear error."""
    print(
        f"\nERROR: {message}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    if not path.exists():
        fail(
            f"File not found for hashing: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def get_git_commit() -> str:
    """Return the current Git commit."""
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


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_markdown(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a human-readable audit report."""
    declaration = payload["declaration"]

    temporal_scales = "\n".join(
        f"- `{value}`"
        for value in declaration[
            "temporal_scales"
        ]
    )

    components = "\n".join(
        f"- `{value}`"
        for value in declaration[
            "component_ids"
        ]
    )

    excluded = "\n".join(
        f"- `{value}`"
        for value in declaration[
            "excluded_interpretations"
        ]
    )

    text = f"""# CGIE3-ID-01 Identity Declaration Audit

## Status

VALID

## System

- System ID: `{declaration["system_id"]}`
- Name: {declaration["system_name"]}
- Domain: `{declaration["domain"]}`
- Protocol version: `{declaration["protocol_version"]}`

## Functional purpose

{declaration["functional_purpose"]}

## System boundary

{declaration["system_boundary"]}

## Observation context

{declaration["observation_context"]}

## Temporal scales

{temporal_scales}

## Declared components

{components}

## Excluded interpretations

{excluded}

## Provenance

- Generated at UTC: `{payload["generated_at_utc"]}`
- Source commit: `{payload["source_commit"]}`
- Declaration SHA-256: `{payload["identity_declaration_sha256"]}`
- Validator SHA-256: `{payload["validator_script_sha256"]}`

## Interpretation boundary

This audit confirms only that the YAML declaration satisfies the
Congruity Core structural contract.

It does not establish that the declared candidate identity is
physically correct, indispensable, predictive or scientifically
validated.
"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    """
    Load, validate and serialize the frozen identity declaration.
    """
    try:
        from congruity.identity import (
            IdentityDeclarationError,
            load_identity_declaration,
            serialize_model,
        )
    except ImportError as exc:
        fail(
            "Unable to import Congruity Core. "
            "Run the script from the repository root "
            "and verify that congruity/__init__.py and "
            "congruity/identity/__init__.py exist. "
            f"Original error: {exc}"
        )

    if not IDENTITY_DECLARATION_PATH.exists():
        fail(
            "Identity declaration not found: "
            f"{IDENTITY_DECLARATION_PATH}"
        )

    try:
        declaration = (
            load_identity_declaration(
                IDENTITY_DECLARATION_PATH,
                reject_unknown_fields=True,
            )
        )
    except IdentityDeclarationError as exc:
        fail(
            "Identity declaration validation failed: "
            f"{exc}"
        )

    serialized_declaration = (
        serialize_model(declaration)
    )

    if not isinstance(
        serialized_declaration,
        dict,
    ):
        fail(
            "Serialized declaration is not a mapping."
        )

    expected_scales = {
        "1d",
        "3d",
        "7d",
        "30d",
    }

    observed_scales = set(
        serialized_declaration[
            "temporal_scales"
        ]
    )

    if observed_scales != expected_scales:
        fail(
            "Unexpected temporal-scale set. "
            f"Expected {sorted(expected_scales)}, "
            f"observed {sorted(observed_scales)}."
        )

    required_components = {
        "event_count",
        "maximum_magnitude",
        "log10_cumulative_energy_joule",
        "median_depth_km",
        "depth_mad_km",
        "spatial_dispersion_km",
        "median_interevent_time_hours",
        "interevent_time_mad_hours",
        "temporal_burstiness",
    }

    observed_components = set(
        serialized_declaration[
            "component_ids"
        ]
    )

    if observed_components != required_components:
        missing = sorted(
            required_components
            - observed_components
        )

        unexpected = sorted(
            observed_components
            - required_components
        )

        fail(
            "Identity component set differs from "
            "the frozen CGIE3-ID-01 declaration. "
            f"Missing: {missing}. "
            f"Unexpected: {unexpected}."
        )

    required_exclusions = {
        "deterministic_earthquake_prediction",
        "operational_public_warning",
        "non_estimable_state_interpreted_as_normal",
    }

    observed_exclusions = set(
        serialized_declaration[
            "excluded_interpretations"
        ]
    )

    missing_exclusions = sorted(
        required_exclusions
        - observed_exclusions
    )

    if missing_exclusions:
        fail(
            "Required interpretation exclusions "
            "are absent: "
            + ", ".join(
                missing_exclusions
            )
        )

    payload = {
        "audit_id":
            "CGIE3_ID_01_IDENTITY_DECLARATION_AUDIT",
        "status":
            "VALID",
        "generated_at_utc":
            utc_now_iso(),
        "source_commit":
            get_git_commit(),
        "identity_declaration_file":
            str(
                IDENTITY_DECLARATION_PATH.relative_to(
                    REPOSITORY_ROOT
                )
            ),
        "identity_declaration_sha256":
            sha256_file(
                IDENTITY_DECLARATION_PATH
            ),
        "validator_script_file":
            str(
                SCRIPT_PATH.relative_to(
                    REPOSITORY_ROOT
                )
            ),
        "validator_script_sha256":
            sha256_file(
                SCRIPT_PATH
            ),
        "declaration":
            serialized_declaration,
        "scientific_claims": {
            "structural_contract_validated":
                True,
            "physical_identity_validated":
                False,
            "indispensable_relations_identified":
                False,
            "predictive_capability_established":
                False,
        },
    }

    write_json(
        AUDIT_JSON_PATH,
        payload,
    )

    write_markdown(
        AUDIT_MD_PATH,
        payload,
    )

    print("=" * 78)
    print(
        "CGIE3-ID-01 — IDENTITY DECLARATION AUDIT"
    )
    print("=" * 78)
    print("Status: VALID")
    print(
        "System ID: "
        f"{declaration.system_id}"
    )
    print(
        "Temporal scales: "
        + ", ".join(
            declaration.temporal_scales
        )
    )
    print(
        "Declared components: "
        f"{len(declaration.component_ids)}"
    )
    print(
        f"Generated: {AUDIT_JSON_PATH}"
    )
    print(
        f"Generated: {AUDIT_MD_PATH}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
