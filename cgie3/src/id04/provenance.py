"""
CGIE3-ID-04 workflow provenance.

This stage records:

- Git revision;
- frozen protocol/configuration/input hashes;
- ID-04 source hashes;
- official output hashes;
- runtime environment;
- scientific outcome and claim boundary.

It does not recompute scientific analyses.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy

from engines.cgie3.src.id04.loader import (
    ID04ExperimentContext,
)


class ID04ProvenanceError(ValueError):
    """Raised when ID-04 provenance violates the frozen contract."""


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]

OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CGIE3_ID_04_workflow_provenance.json"
)


SOURCE_FILES = {
    "id04_package":
        "engines/cgie3/src/id04/__init__.py",

    "id04_loader":
        "engines/cgie3/src/id04/loader.py",

    "id04_snapshots":
        "engines/cgie3/src/id04/snapshots.py",

    "id04_continuity":
        "engines/cgie3/src/id04/continuity.py",

    "id04_null_controls":
        "engines/cgie3/src/id04/null_controls.py",

    "id04_robustness":
        "engines/cgie3/src/id04/robustness.py",

    "id04_reporting":
        "engines/cgie3/src/id04/reporting.py",

    "id04_provenance":
        "engines/cgie3/src/id04/provenance.py",

    "identity_models":
        "congruity/identity/models.py",

    "identity_declaration":
        "congruity/identity/declaration.py",

    "id04_main":
    "engines/cgie3/src/id04/main.py",
}


OFFICIAL_OUTPUT_FILES = {
    "snapshot_relations":
        "outputs/CGIE3_ID_04_snapshot_relations.csv",

    "transition_continuity":
        "outputs/CGIE3_ID_04_transition_continuity.csv",

    "multiscale_continuity":
        "outputs/CGIE3_ID_04_multiscale_continuity.csv",

    "null_controls":
        "outputs/CGIE3_ID_04_null_controls.csv",

    "null_summaries":
        "outputs/CGIE3_ID_04_null_summaries.csv",

    "leave_one_transition_out":
        "outputs/CGIE3_ID_04_leave_one_transition_out.csv",

    "estimability":
        "outputs/CGIE3_ID_04_estimability.csv",

    "summary":
        "outputs/CGIE3_ID_04_summary.json",

    "report":
        "outputs/CGIE3_ID_04_report.md",

    "manifest":
        "outputs/CGIE3_ID_04_manifest.json",
}


def fail(message: str) -> None:
    """Raise normalized provenance error."""
    raise ID04ProvenanceError(
        str(message).strip()
    )


def utc_now_iso() -> str:
    """Return canonical UTC timestamp."""
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
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


def validate_inputs(
    context: ID04ExperimentContext,
    result: Mapping[str, Any],
) -> None:
    """Validate provenance prerequisites."""
    if not isinstance(
        context,
        ID04ExperimentContext,
    ):
        fail(
            "context must be an ID04ExperimentContext."
        )

    if not isinstance(
        result,
        Mapping,
    ):
        fail(
            "result must be a mapping."
        )

    if context.experiment_id != "CGIE3_ID_04":
        fail(
            "Unexpected experiment ID."
        )

    if result.get(
        "experiment_id"
    ) != "CGIE3_ID_04":
        fail(
            "Unexpected result experiment ID."
        )

    if (
        context.runtime.get(
            "reporting_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Reporting must complete before provenance."
        )

    required_outputs = {
        "official_summary",
        "official_manifest",
        "official_output_hashes",
        "official_result",
    }

    missing = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing:
        fail(
            "Official reporting outputs are missing: "
            + ", ".join(
                missing
            )
        )


def repository_file(
    relative_path: str,
) -> Path:
    """Resolve one safe repository-relative file."""
    path = (
        REPOSITORY_ROOT
        / relative_path
    ).resolve()

    try:
        path.relative_to(
            REPOSITORY_ROOT
        )
    except ValueError as exc:
        raise ID04ProvenanceError(
            "Path escapes repository root: "
            f"{relative_path}"
        ) from exc

    return path


def sha256_file(
    path: Path,
) -> str:
    """Calculate SHA-256 for one file."""
    if not path.exists():
        fail(
            f"File not found for hashing: {path}"
        )

    if not path.is_file():
        fail(
            f"Path is not a file: {path}"
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


def collect_file_hashes(
    files: Mapping[str, str],
    *,
    require_all: bool = True,
) -> dict[str, Any]:
    """Collect path, size and hash for declared files."""
    output: dict[
        str,
        Any,
    ] = {}

    for file_id, relative_path in files.items():
        path = repository_file(
            relative_path
        )

        if not path.exists():
            if require_all:
                fail(
                    "Required provenance file is missing: "
                    f"{relative_path}"
                )

            output[
                file_id
            ] = {
                "path":
                    relative_path,

                "exists":
                    False,

                "size_bytes":
                    None,

                "sha256":
                    None,
            }

            continue

        output[
            file_id
        ] = {
            "path":
                relative_path,

            "exists":
                True,

            "size_bytes":
                int(
                    path.stat().st_size
                ),

            "sha256":
                sha256_file(
                    path
                ),
        }

    return output


def collect_input_files(
    context: ID04ExperimentContext,
) -> dict[str, str]:
    """Recover frozen ID-04 input paths."""
    inputs = require_mapping(
        context.configuration.get(
            "inputs"
        ),
        "inputs",
    )

    files: dict[
        str,
        str,
    ] = {}

    for input_id in (
        "identity_declaration",
        "frozen_features",
        "id02_relation_classification",
        "id03_relation_states",
        "id03_family_membership",
        "id03_relation_families",
        "id03_summary",
    ):
        entry = require_mapping(
            inputs.get(
                input_id
            ),
            f"inputs.{input_id}",
        )

        files[
            input_id
        ] = str(
            entry[
                "file"
            ]
        )

    files[
        "id04_configuration"
    ] = (
        "engines/cgie3/config/"
        "cgie3_id_04_relational_continuity.yaml"
    )

    files[
        "id04_protocol"
    ] = (
        "docs/experiments/"
        "CGIE3_ID_04_PROTOCOL.md"
    )

    return files


def get_git_value(
    arguments: list[str],
) -> str:
    """Run one Git command with stable fallback."""
    try:
        completed = subprocess.run(
            [
                "git",
                *arguments,
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

    value = completed.stdout.strip()

    return (
        value
        if value
        else "UNAVAILABLE"
    )


def get_git_context() -> dict[str, Any]:
    """Collect Git revision and working-tree information."""
    status = get_git_value(
        [
            "status",
            "--porcelain",
        ]
    )

    if status == "UNAVAILABLE":
        clean: bool | None = None
    else:
        clean = (
            status == ""
        )

    return {
        "commit":
            get_git_value(
                [
                    "rev-parse",
                    "HEAD",
                ]
            ),

        "branch":
            get_git_value(
                [
                    "rev-parse",
                    "--abbrev-ref",
                    "HEAD",
                ]
            ),

        "repository_root":
            str(
                REPOSITORY_ROOT
            ),

        "working_tree_clean":
            clean,
    }


def json_compatible(
    value: Any,
) -> Any:
    """Convert scientific values into strict JSON-compatible objects."""
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        normalized = float(
            value
        )

        if not np.isfinite(
            normalized
        ):
            return None

        return normalized

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(
                key
            ):
                json_compatible(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            json_compatible(
                item
            )
            for item in value
        ]

    return str(
        value
    )


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write deterministic strict JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            json_compatible(
                payload
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_payload(
    context: ID04ExperimentContext,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete ID-04 provenance payload."""
    source_hashes = collect_file_hashes(
        SOURCE_FILES,
        require_all=True,
    )

    input_hashes = collect_file_hashes(
        collect_input_files(
            context
        ),
        require_all=True,
    )

    output_hashes = collect_file_hashes(
        OFFICIAL_OUTPUT_FILES,
        require_all=True,
    )

    experiment = require_mapping(
        context.configuration.get(
            "experiment"
        ),
        "experiment",
    )

    return {
        "provenance_id":
            "CGIE3_ID_04_WORKFLOW_PROVENANCE",

        "experiment_id":
            "CGIE3_ID_04",

        "generated_at_utc":
            utc_now_iso(),

        "protocol": {
            "name":
                experiment.get(
                    "name"
                ),

            "framework_version":
                experiment.get(
                    "framework_version"
                ),

            "protocol_version":
                experiment.get(
                    "protocol_version"
                ),

            "configuration_version":
                experiment.get(
                    "configuration_version"
                ),

            "configuration_status":
                experiment.get(
                    "status"
                ),
        },

        "git":
            get_git_context(),

        "execution_environment": {
            "python_version":
                platform.python_version(),

            "python_implementation":
                platform.python_implementation(),

            "python_executable":
                sys.executable,

            "platform":
                platform.platform(),

            "operating_system":
                platform.system(),

            "machine":
                platform.machine(),

            "processor":
                platform.processor(),

            "numpy_version":
                np.__version__,

            "pandas_version":
                pd.__version__,

            "scipy_version":
                scipy.__version__,

            "github_actions": {
                "repository":
                    os.environ.get(
                        "GITHUB_REPOSITORY"
                    ),

                "workflow":
                    os.environ.get(
                        "GITHUB_WORKFLOW"
                    ),

                "run_id":
                    os.environ.get(
                        "GITHUB_RUN_ID"
                    ),

                "run_number":
                    os.environ.get(
                        "GITHUB_RUN_NUMBER"
                    ),

                "run_attempt":
                    os.environ.get(
                        "GITHUB_RUN_ATTEMPT"
                    ),

                "ref_name":
                    os.environ.get(
                        "GITHUB_REF_NAME"
                    ),

                "sha":
                    os.environ.get(
                        "GITHUB_SHA"
                    ),

                "actor":
                    os.environ.get(
                        "GITHUB_ACTOR"
                    ),

                "runner_os":
                    os.environ.get(
                        "RUNNER_OS"
                    ),
            },
        },

        "identity": {
            "system_id":
                context.identity.system_id,

            "system_name":
                context.identity.system_name,

            "domain":
                context.identity.domain,

            "protocol_version":
                context.identity.protocol_version,

            "component_ids":
                list(
                    context.identity.component_ids
                ),

            "temporal_scales":
                list(
                    context.identity.temporal_scales
                ),

            "excluded_interpretations":
                list(
                    context.identity.excluded_interpretations
                ),
        },

        "upstream_frozen_state": {
            "id02_total_relation_count":
                136,

            "id04_primary_relation_count":
                74,

            "id03_scientific_outcome":
                context.metadata.get(
                    "id03_scientific_outcome"
                ),

            "id03_reproducible_family_count":
                context.metadata.get(
                    "id03_reproducible_family_count"
                ),

            "id02_statuses_modified":
                False,

            "id03_states_modified":
                False,
        },

        "configuration_boundary": {
            "snapshot_definition":
                context.configuration.get(
                    "snapshot_definition"
                ),

            "continuity_components":
                context.configuration.get(
                    "continuity_components"
                ),

            "relational_continuity_score":
                context.configuration.get(
                    "relational_continuity_score"
                ),

            "null_controls":
                context.configuration.get(
                    "null_controls"
                ),

            "empirical_significance":
                context.configuration.get(
                    "empirical_significance"
                ),

            "robustness":
                context.configuration.get(
                    "robustness"
                ),

            "estimability":
                context.configuration.get(
                    "estimability"
                ),

            "outcomes":
                context.configuration.get(
                    "outcomes"
                ),

            "forbidden_analyses":
                context.configuration.get(
                    "forbidden_analyses"
                ),
        },

        "runtime_stages":
            context.runtime,

        "experiment_metadata":
            context.metadata,

        "source_files":
            source_hashes,

        "input_files":
            input_hashes,

        "official_output_files":
            output_hashes,

        "official_output_hashes_from_reporting":
            context.outputs.get(
                "official_output_hashes"
            ),

        "scientific_result": {
            "technical_status":
                result.get(
                    "technical_status"
                ),

            "scientific_outcome":
                result.get(
                    "scientific_outcome"
                ),

            "statistics":
                result.get(
                    "statistics"
                ),

            "scientific_claims":
                result.get(
                    "scientific_claims"
                ),

            "warnings":
                result.get(
                    "warnings"
                ),
        },

        "scientific_claim_boundary": {
            "relational_continuity_supported":
                bool(
                    result.get(
                        "scientific_claims",
                        {},
                    ).get(
                        "relational_continuity_supported",
                        False,
                    )
                ),

            "minimum_identity_core_established":
                False,

            "indispensable_relations_established":
                False,

            "causality_established":
                False,

            "predictive_capability_established":
                False,

            "earthquake_prediction_established":
                False,

            "universal_transferability_established":
                False,

            "earthquake_event_information_used":
                False,
        },
    }


def generate_provenance(
    context: ID04ExperimentContext,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate official CGIE3-ID-04 workflow provenance."""
    validate_inputs(
        context,
        result,
    )

    payload = build_payload(
        context,
        result,
    )

    write_json(
        OUTPUT_PATH,
        payload,
    )

    provenance_hash = sha256_file(
        OUTPUT_PATH
    )

    context.register_output(
        "workflow_provenance",
        payload,
    )

    context.register_output(
        "workflow_provenance_sha256",
        provenance_hash,
    )

    context.register_runtime(
        "provenance_status",
        "COMPLETED",
    )

    updated_result = dict(
        result
    )

    updated_outputs = dict(
        require_mapping(
            updated_result.get(
                "outputs"
            ),
            "result.outputs",
        )
    )

    updated_outputs[
        "workflow_provenance"
    ] = {
        "path":
            str(
                OUTPUT_PATH.relative_to(
                    REPOSITORY_ROOT
                )
            ),

        "sha256":
            provenance_hash,
    }

    updated_result[
        "outputs"
    ] = updated_outputs

    updated_result[
        "provenance"
    ] = {
        "path":
            str(
                OUTPUT_PATH.relative_to(
                    REPOSITORY_ROOT
                )
            ),

        "sha256":
            provenance_hash,

        "payload":
            payload,
    }

    context.register_output(
        "final_result",
        updated_result,
    )

    return updated_result 
