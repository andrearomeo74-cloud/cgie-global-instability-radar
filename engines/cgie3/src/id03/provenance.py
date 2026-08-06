"""
CGIE3-ID-03 workflow provenance.

This stage records:

- Git revision;
- frozen configuration and input hashes;
- ID-03 source hashes;
- official output hashes;
- runtime and software environment;
- scientific result and claim boundary.

It does not:

- alter scientific outputs;
- recompute any audit;
- modify ID-02 classifications;
- change ID-03 states;
- infer causality, indispensability or prediction.
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

import networkx as nx
import numpy as np
import pandas as pd

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class ID03ProvenanceError(ValueError):
    """Raised when ID-03 provenance violates the frozen contract."""


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]

OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CGIE3_ID_03_workflow_provenance.json"
)


SOURCE_FILES = {
    "id03_package":
        "engines/cgie3/src/id03/__init__.py",

    "id03_loader":
        "engines/cgie3/src/id03/loader.py",

    "id03_dependencies":
        "engines/cgie3/src/id03/dependencies.py",

    "id03_multiscale":
        "engines/cgie3/src/id03/multiscale.py",

    "id03_overlap":
        "engines/cgie3/src/id03/overlap.py",

    "id03_null_controls":
        "engines/cgie3/src/id03/null_controls.py",

    "id03_redundancy":
        "engines/cgie3/src/id03/redundancy.py",

    "id03_families":
        "engines/cgie3/src/id03/families.py",

    "id03_representatives":
        "engines/cgie3/src/id03/representatives.py",

    "id03_reporting":
        "engines/cgie3/src/id03/reporting.py",

    "id03_provenance":
        "engines/cgie3/src/id03/provenance.py",

    "identity_models":
        "congruity/identity/models.py",

    "identity_loader":
        "congruity/identity/declaration.py",
}


OFFICIAL_OUTPUT_FILES = {
    "feature_dependencies":
        "outputs/CGIE3_ID_03_feature_dependencies.csv",

    "relation_dependencies":
        "outputs/CGIE3_ID_03_relation_dependencies.csv",

    "multiscale_relations":
        "outputs/CGIE3_ID_03_multiscale_relations.csv",

    "overlap_sensitivity":
        "outputs/CGIE3_ID_03_overlap_sensitivity.csv",

    "overlap_estimates_long":
        "outputs/CGIE3_ID_03_overlap_estimates_long.csv",

    "null_controls":
        "outputs/CGIE3_ID_03_null_controls.csv",

    "null_control_summaries":
        "outputs/CGIE3_ID_03_null_control_summaries.csv",

    "conditional_redundancy":
        "outputs/CGIE3_ID_03_conditional_redundancy.csv",

    "relation_families":
        "outputs/CGIE3_ID_03_relation_families.csv",

    "family_membership":
        "outputs/CGIE3_ID_03_family_membership.csv",

    "family_coassignment":
        "outputs/CGIE3_ID_03_family_coassignment.csv",

    "relation_states":
        "outputs/CGIE3_ID_03_relation_states.csv",

    "representative_candidates":
        "outputs/CGIE3_ID_03_representative_candidates.csv",

    "family_graph":
        "outputs/CGIE3_ID_03_family_graph.graphml",

    "summary":
        "outputs/CGIE3_ID_03_summary.json",

    "report":
        "outputs/CGIE3_ID_03_report.md",

    "manifest":
        "outputs/CGIE3_ID_03_manifest.json",
}


def fail(message: str) -> None:
    """Raise a normalized provenance error."""
    raise ID03ProvenanceError(
        str(message).strip()
    )


def utc_now_iso() -> str:
    """Return canonical UTC time."""
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
    if not isinstance(
        value,
        Mapping,
    ):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def validate_inputs(
    context: ID03ExperimentContext,
    result: Mapping[str, Any],
) -> None:
    """Validate provenance-stage prerequisites."""
    if not isinstance(
        context,
        ID03ExperimentContext,
    ):
        fail(
            "context must be an ID03ExperimentContext."
        )

    if not isinstance(
        result,
        Mapping,
    ):
        fail(
            "result must be a mapping."
        )

    if context.experiment_id != "CGIE3_ID_03":
        fail(
            "Unexpected context experiment ID: "
            f"{context.experiment_id}"
        )

    if result.get(
        "experiment_id"
    ) != "CGIE3_ID_03":
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
        - set(context.outputs)
    )

    if missing:
        fail(
            "Official reporting outputs are missing: "
            + ", ".join(missing)
        )


def sha256_file(
    path: Path,
) -> str:
    """Calculate SHA-256 for one existing file."""
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
        raise ID03ProvenanceError(
            f"Unable to hash {path}: {exc}"
        ) from exc

    return digest.hexdigest()


def repository_file(
    relative_path: str,
) -> Path:
    """Resolve one repository-relative path safely."""
    path = (
        REPOSITORY_ROOT
        / relative_path
    ).resolve()

    try:
        path.relative_to(
            REPOSITORY_ROOT
        )
    except ValueError as exc:
        raise ID03ProvenanceError(
            "Path escapes repository root: "
            f"{relative_path}"
        ) from exc

    return path


def get_git_value(
    arguments: list[str],
) -> str:
    """Run one Git command with a stable fallback."""
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
    """Collect Git revision and worktree information."""
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


def collect_file_hashes(
    files: Mapping[str, str],
    *,
    require_all: bool,
) -> dict[str, Any]:
    """Collect path, size and hash for declared files."""
    output: dict[str, Any] = {}

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

            output[file_id] = {
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

        output[file_id] = {
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
    context: ID03ExperimentContext,
) -> dict[str, str]:
    """Recover frozen input paths from the ID-03 configuration."""
    inputs = require_mapping(
        context.configuration.get(
            "inputs"
        ),
        "inputs",
    )

    files: dict[str, str] = {}

    for input_id in (
        "identity_declaration",
        "id02_configuration",
        "frozen_features",
        "id02_candidate_relations",
        "id02_block_relations",
        "id02_bootstrap_relations",
        "id02_relation_classification",
        "id02_equivalence_flags",
        "id02_summary",
    ):
        section = require_mapping(
            inputs.get(
                input_id
            ),
            f"inputs.{input_id}",
        )

        files[input_id] = str(
            section[
                "file"
            ]
        )

    files[
        "id03_configuration"
    ] = (
        "engines/cgie3/config/"
        "cgie3_id_03_family_audit.yaml"
    )

    files[
        "id03_protocol"
    ] = (
        "docs/experiments/"
        "CGIE3_ID_03_PROTOCOL.md"
    )

    return files


def json_compatible(
    value: Any,
) -> Any:
    """Convert values into strict JSON-compatible structures."""
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
        return int(value)

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        normalized = float(value)

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
        return str(value)

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_compatible(item)
            for key, item in value.items()
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
            json_compatible(item)
            for item in value
        ]

    return str(value)


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
            json_compatible(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_provenance_payload(
    context: ID03ExperimentContext,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete official workflow provenance."""
    official_hashes = require_mapping(
        context.outputs[
            "official_output_hashes"
        ],
        "official_output_hashes",
    )

    configuration_experiment = require_mapping(
        context.configuration[
            "experiment"
        ],
        "experiment",
    )

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

    null_replications = context.outputs.get(
        "null_control_replications"
    )

    if not isinstance(
        null_replications,
        pd.DataFrame,
    ):
        fail(
            "null_control_replications must be a DataFrame."
        )

    return {
        "provenance_id":
            "CGIE3_ID_03_WORKFLOW_PROVENANCE",

        "experiment_id":
            context.experiment_id,

        "generated_at_utc":
            utc_now_iso(),

        "protocol": {
            "name":
                configuration_experiment.get(
                    "name"
                ),

            "framework_version":
                configuration_experiment.get(
                    "framework_version"
                ),

            "protocol_version":
                configuration_experiment.get(
                    "protocol_version"
                ),

            "configuration_version":
                configuration_experiment.get(
                    "configuration_version"
                ),

            "configuration_status":
                configuration_experiment.get(
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

            "networkx_version":
                nx.__version__,

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

        "configuration_boundary": {
            "analysis_period":
                context.configuration.get(
                    "analysis_period"
                ),

            "interpretation_boundary":
                context.configuration.get(
                    "interpretation_boundary"
                ),

            "safety_rules":
                context.configuration.get(
                    "safety_rules"
                ),

            "decision_boundary":
                context.configuration.get(
                    "decision_boundary"
                ),

            "advancement_to_id04":
                context.configuration.get(
                    "advancement_to_id04"
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
            dict(
                official_hashes
            ),

        "non_exported_intermediate_outputs": {
            "null_control_replications": {
                "row_count":
                    int(
                        len(
                            null_replications
                        )
                    ),

                "expected_row_count":
                    66600,

                "exported_as_official_csv":
                    False,

                "reason":
                    (
                        "Preserved in execution context but not "
                        "exported in the principal scientific package "
                        "because it contains 66,600 replication rows."
                    ),
            },
        },

        "result": {
            "status":
                result.get(
                    "status"
                ),

            "technical_status":
                result.get(
                    "technical_status"
                ),

            "summary":
                result.get(
                    "summary"
                ),

            "scientific_claims":
                result.get(
                    "scientific_claims"
                ),

            "statistics":
                result.get(
                    "statistics"
                ),

            "warnings":
                result.get(
                    "warnings"
                ),
        },

        "scientific_claim_boundary": {
            "id02_statuses_modified":
                False,

            "primary_relations_established":
                False,

            "indispensable_relations_established":
                False,

            "minimum_identity_core_established":
                False,

            "causality_established":
                False,

            "predictive_capability_established":
                False,

            "earthquake_prediction_established":
                False,

            "universal_transferability_established":
                False,
        },
    }


def generate_provenance(
    context: ID03ExperimentContext,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Generate official ID-03 workflow provenance.

    Returns a copied result mapping containing the provenance output.
    """
    validate_inputs(
        context,
        result,
    )

    payload = build_provenance_payload(
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
