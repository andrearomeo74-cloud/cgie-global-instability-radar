"""
CGIE3-ID-02 provenance stage.

This module records complete execution provenance after the official
scientific outputs have been generated.

It does not:

- alter scientific results;
- recompute relations;
- change classifications;
- modify frozen inputs;
- infer scientific claims.
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

from congruity.core import (
    ExperimentContext,
    ExperimentResult,
)


class ProvenanceStageError(ValueError):
    """Raised when provenance generation violates the contract."""


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]

OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CGIE3_ID_02_workflow_provenance.json"
)


SOURCE_FILES = {
    "experiment_context":
        "congruity/core/context.py",

    "experiment_result":
        "congruity/core/result.py",

    "identity_models":
        "congruity/identity/models.py",

    "identity_loader":
        "congruity/identity/declaration.py",

    "relation_discovery_core":
        "congruity/relations/discovery.py",

    "id02_loader":
        "engines/cgie3/src/id02/loader.py",

    "id02_preprocessing":
        "engines/cgie3/src/id02/preprocessing.py",

    "id02_discovery":
        "engines/cgie3/src/id02/discovery.py",

    "id02_persistence":
        "engines/cgie3/src/id02/persistence.py",

    "id02_bootstrap":
        "engines/cgie3/src/id02/bootstrap.py",

    "id02_missingness":
        "engines/cgie3/src/id02/missingness.py",

    "id02_classification":
        "engines/cgie3/src/id02/classification.py",

    "id02_equivalence":
        "engines/cgie3/src/id02/equivalence.py",

    "id02_reporting":
        "engines/cgie3/src/id02/reporting.py",

    "id02_provenance":
        "engines/cgie3/src/id02/provenance.py",

    "id02_package":
        "engines/cgie3/src/id02/__init__.py",
    
    "id02_main":
    "engines/cgie3/src/id02/main.py",
}


OFFICIAL_OUTPUT_FILES = {
    "candidate_relations":
        "outputs/CGIE3_ID_02_candidate_relations.csv",

    "block_relations":
        "outputs/CGIE3_ID_02_block_relations.csv",

    "bootstrap_relations":
        "outputs/CGIE3_ID_02_bootstrap_relations.csv",

    "relation_classification":
        "outputs/CGIE3_ID_02_relation_classification.csv",

    "equivalence_flags":
        "outputs/CGIE3_ID_02_equivalence_flags.csv",

    "summary":
        "outputs/CGIE3_ID_02_summary.json",

    "report":
        "outputs/CGIE3_ID_02_report.md",

    "manifest":
        "outputs/CGIE3_ID_02_manifest.json",
}


def fail(message: str) -> None:
    """Raise a provenance-stage error."""
    raise ProvenanceStageError(
        str(message).strip()
    )


def utc_now_iso() -> str:
    """Return current UTC time in canonical ISO-8601 form."""
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
    context: ExperimentContext,
    result: ExperimentResult,
) -> None:
    """Validate provenance-stage prerequisites."""
    if not isinstance(
        context,
        ExperimentContext,
    ):
        fail(
            "context must be an ExperimentContext."
        )

    if not isinstance(
        result,
        ExperimentResult,
    ):
        fail(
            "result must be an ExperimentResult."
        )

    if context.experiment_id != "CGIE3_ID_02":
        fail(
            "Unexpected context experiment ID: "
            f"{context.experiment_id}"
        )

    if result.experiment_id != "CGIE3_ID_02":
        fail(
            "Unexpected result experiment ID: "
            f"{result.experiment_id}"
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
    }

    missing_outputs = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing_outputs:
        fail(
            "Official reporting outputs are missing: "
            + ", ".join(missing_outputs)
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
        raise ProvenanceStageError(
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
        raise ProvenanceStageError(
            "Path escapes repository root: "
            f"{relative_path}"
        ) from exc

    return path


def get_git_value(
    arguments: list[str],
) -> str:
    """Execute a Git command and return a stable fallback."""
    try:
        result = subprocess.run(
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

    value = result.stdout.strip()

    return (
        value
        if value
        else "UNAVAILABLE"
    )


def get_git_context() -> dict[str, Any]:
    """Collect repository revision information."""
    status = get_git_value(
        [
            "status",
            "--porcelain",
        ]
    )

    if status == "UNAVAILABLE":
        working_tree_clean: bool | None = None
    else:
        working_tree_clean = (
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
            working_tree_clean,
    }


def collect_file_hashes(
    files: Mapping[str, str],
    *,
    require_all: bool,
) -> dict[str, Any]:
    """Collect hashes and sizes for declared repository files."""
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
    context: ExperimentContext,
) -> dict[str, str]:
    """Recover frozen input paths from configuration."""
    configuration = context.configuration

    identity_section = require_mapping(
        configuration.get(
            "identity_declaration"
        ),
        "identity_declaration",
    )

    inputs = require_mapping(
        configuration.get(
            "inputs"
        ),
        "inputs",
    )

    frozen_features = require_mapping(
        inputs.get(
            "frozen_features"
        ),
        "inputs.frozen_features",
    )

    frozen_catalogue = require_mapping(
        inputs.get(
            "frozen_catalogue"
        ),
        "inputs.frozen_catalogue",
    )

    cgie2_metrics = require_mapping(
        inputs.get(
            "cgie2_metrics"
        ),
        "inputs.cgie2_metrics",
    )

    files = {
        "identity_declaration":
            str(
                identity_section[
                    "file"
                ]
            ),

        "frozen_features":
            str(
                frozen_features[
                    "file"
                ]
            ),

        "frozen_catalogue":
            str(
                frozen_catalogue[
                    "file"
                ]
            ),

        "cgie2_metrics":
            str(
                cgie2_metrics[
                    "file"
                ]
            ),

        "frozen_configuration":
            "engines/cgie3/config/"
            "cgie3_id_02_relation_discovery.yaml",
    }

    catalogue_hash_file = (
        frozen_catalogue.get(
            "hash_file"
        )
    )

    if catalogue_hash_file:
        files[
            "frozen_catalogue_hash_file"
        ] = str(
            catalogue_hash_file
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
            json_compatible(item)
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return {
            field_name: json_compatible(
                getattr(
                    value,
                    field_name,
                )
            )
            for field_name
            in value.__dataclass_fields__
        }

    return str(value)


def write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write strict deterministic JSON."""
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


def build_provenance_payload(
    context: ExperimentContext,
    result: ExperimentResult,
) -> dict[str, Any]:
    """Build complete workflow provenance."""
    official_output_hashes = (
        require_mapping(
            context.outputs[
                "official_output_hashes"
            ],
            "official_output_hashes",
        )
    )

    source_hashes = collect_file_hashes(
        SOURCE_FILES,
        require_all=True,
    )

    input_files = collect_input_files(
        context
    )

    input_hashes = collect_file_hashes(
        input_files,
        require_all=True,
    )

    output_hashes = collect_file_hashes(
        OFFICIAL_OUTPUT_FILES,
        require_all=True,
    )

    return {
        "provenance_id":
            "CGIE3_ID_02_WORKFLOW_PROVENANCE",

        "experiment_id":
            context.experiment_id,

        "generated_at_utc":
            utc_now_iso(),

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

            "pandas_version":
                pd.__version__,

            "numpy_version":
                np.__version__,

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

        "git":
            get_git_context(),

        "configuration": {
            "experiment":
                context.configuration.get(
                    "experiment"
                ),

            "analysis_period":
                context.configuration.get(
                    "analysis_period"
                ),

            "safety_rules":
                context.configuration.get(
                    "safety_rules"
                ),

            "decision_boundary":
                context.configuration.get(
                    "decision_boundary"
                ),
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

            "temporal_scales":
                list(
                    context.identity.temporal_scales
                ),

            "component_ids":
                list(
                    context.identity.component_ids
                ),

            "excluded_interpretations":
                list(
                    context.identity.excluded_interpretations
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
                official_output_hashes
            ),

        "experiment_result": {
            "status":
                result.status,

            "summary":
                result.summary,

            "scientific_claims":
                result.scientific_claims,

            "statistics":
                result.statistics,

            "warnings":
                list(
                    result.warnings
                ),
        },

        "scientific_claim_boundary": {
            "primary_relations_established":
                False,

            "indispensable_relations_established":
                False,

            "causality_established":
                False,

            "predictive_capability_established":
                False,

            "earthquake_prediction_established":
                False,
        },
    }


def generate_provenance(
    context: ExperimentContext,
    result: ExperimentResult,
) -> ExperimentResult:
    """
    Generate official provenance and return an updated immutable result.
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

    updated_outputs = dict(
        result.outputs
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

    updated_provenance = dict(
        result.provenance
    )

    updated_provenance[
        "workflow_provenance"
    ] = payload

    return ExperimentResult(
        experiment_id=
            result.experiment_id,

        status=
            result.status,

        summary=
            result.summary,

        outputs=
            updated_outputs,

        scientific_claims=
            result.scientific_claims,

        statistics=
            result.statistics,

        warnings=
            result.warnings,

        provenance=
            updated_provenance,

        metadata=
            result.metadata,
  )
