#!/usr/bin/env python3
"""
CGIE3-ID-02 pipeline entry point.

This module executes the frozen Eligible Relation Discovery
experiment in the declared order:

1. load frozen inputs;
2. preprocess baseline data;
3. discover candidate relations;
4. evaluate monthly persistence;
5. evaluate bootstrap uncertainty;
6. evaluate missingness robustness;
7. classify relations;
8. audit equivalence and redundancy;
9. generate scientific reports;
10. generate workflow provenance.

Run from the repository root:

    python -m engines.cgie3.src.id02.main

or:

    python engines/cgie3/src/id02/main.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from congruity.core import (
    ExperimentContext,
    ExperimentResult,
)

from engines.cgie3.src.id02.bootstrap import (
    BootstrapStageError,
    evaluate_bootstrap,
)
from engines.cgie3.src.id02.classification import (
    ClassificationStageError,
    classify_relations,
)
from engines.cgie3.src.id02.discovery import (
    DiscoveryStageError,
    discover,
)
from engines.cgie3.src.id02.equivalence import (
    EquivalenceStageError,
    evaluate_equivalence,
)
from engines.cgie3.src.id02.loader import (
    ExperimentLoaderError,
    load_experiment,
)
from engines.cgie3.src.id02.missingness import (
    MissingnessStageError,
    evaluate_missingness,
)
from engines.cgie3.src.id02.persistence import (
    PersistenceStageError,
    evaluate_persistence,
)
from engines.cgie3.src.id02.preprocessing import (
    PreprocessingError,
    preprocess,
)
from engines.cgie3.src.id02.provenance import (
    ProvenanceStageError,
    generate_provenance,
)
from engines.cgie3.src.id02.reporting import (
    ReportingStageError,
    generate_reports,
)


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]

FINAL_STATUS_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CGIE3_ID_02_execution_status.json"
)


EXPECTED_STAGE_ORDER = (
    "load",
    "preprocess",
    "discover",
    "persistence",
    "bootstrap",
    "missingness",
    "classification",
    "equivalence",
    "reporting",
    "provenance",
)


KNOWN_STAGE_ERRORS = (
    ExperimentLoaderError,
    PreprocessingError,
    DiscoveryStageError,
    PersistenceStageError,
    BootstrapStageError,
    MissingnessStageError,
    ClassificationStageError,
    EquivalenceStageError,
    ReportingStageError,
    ProvenanceStageError,
)


class PipelineExecutionError(RuntimeError):
    """Raised when the CGIE3-ID-02 pipeline cannot complete."""


def print_header() -> None:
    """Print the experiment header."""
    print()
    print("=" * 78)
    print("CGIE3-ID-02 — ELIGIBLE RELATION DISCOVERY")
    print("=" * 78)
    print("Framework: Congruity Framework")
    print("Engine: CGIE-3")
    print("Protocol: CGIE3_ID_02_v1.0")
    print("Configuration status: FROZEN")
    print("=" * 78)


def print_stage_start(
    stage_number: int,
    stage_name: str,
) -> None:
    """Print one stage start marker."""
    print()
    print(
        f"[{stage_number:02d}/"
        f"{len(EXPECTED_STAGE_ORDER):02d}] "
        f"START — {stage_name}"
    )


def print_stage_complete(
    stage_number: int,
    stage_name: str,
    duration_seconds: float,
) -> None:
    """Print one stage completion marker."""
    print(
        f"[{stage_number:02d}/"
        f"{len(EXPECTED_STAGE_ORDER):02d}] "
        f"DONE  — {stage_name} "
        f"({duration_seconds:.3f} s)"
    )


def json_safe(
    value: Any,
) -> Any:
    """Convert execution-status values into JSON-compatible objects."""
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
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
            json_safe(item)
            for item in value
        ]

    return str(value)


def write_execution_status(
    payload: dict[str, Any],
) -> None:
    """Write the final technical execution status."""
    FINAL_STATUS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_STATUS_PATH.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def execute_provenance_stage(
    context: ExperimentContext,
    result: ExperimentResult,
    *,
    stage_number: int,
    timing: dict[str, float],
) -> ExperimentResult:
    """Execute the provenance stage."""
    stage_name = "provenance"

    print_stage_start(
        stage_number,
        stage_name,
    )

    started = time.perf_counter()

    final_result = generate_provenance(
        context,
        result,
    )

    duration = (
        time.perf_counter()
        - started
    )

    if not isinstance(
        final_result,
        ExperimentResult,
    ):
        raise PipelineExecutionError(
            "Provenance did not return "
            "an ExperimentResult."
        )

    timing[
        stage_name
    ] = float(duration)

    print_stage_complete(
        stage_number,
        stage_name,
        duration,
    )

    return final_result


def validate_final_result(
    context: ExperimentContext,
    result: ExperimentResult,
) -> None:
    """Validate the final immutable experiment result."""
    if result.experiment_id != "CGIE3_ID_02":
        raise PipelineExecutionError(
            "Final result contains an unexpected "
            "experiment identifier."
        )

    required_outputs = {
        "candidate_relations",
        "block_relations",
        "bootstrap_relations",
        "relation_classification",
        "equivalence_flags",
        "summary",
        "report",
        "manifest",
        "workflow_provenance",
    }

    missing_outputs = sorted(
        required_outputs
        - set(result.outputs)
    )

    if missing_outputs:
        raise PipelineExecutionError(
            "Final ExperimentResult is missing outputs: "
            + ", ".join(missing_outputs)
        )

    required_claims = {
        "candidate_relations_evaluated",
        "eligibility_protocol_applied",
        "primary_relations_established",
        "indispensable_relations_established",
        "causality_established",
        "predictive_capability_established",
        "earthquake_prediction_established",
    }

    missing_claims = sorted(
        required_claims
        - set(result.scientific_claims)
    )

    if missing_claims:
        raise PipelineExecutionError(
            "Final ExperimentResult is missing "
            "scientific-claim fields: "
            + ", ".join(missing_claims)
        )

    prohibited_true_claims = {
        "primary_relations_established",
        "indispensable_relations_established",
        "causality_established",
        "predictive_capability_established",
        "earthquake_prediction_established",
    }

    invalid_claims = sorted(
        claim
        for claim in prohibited_true_claims
        if result.scientific_claims.get(
            claim
        )
        is True
    )

    if invalid_claims:
        raise PipelineExecutionError(
            "Final result violates the scientific "
            "claim boundary: "
            + ", ".join(invalid_claims)
        )

    if (
        context.runtime.get(
            "provenance_status"
        )
        != "COMPLETED"
    ):
        raise PipelineExecutionError(
            "Provenance stage is not marked COMPLETED."
        )


def print_final_summary(
    result: ExperimentResult,
    timing: dict[str, float],
    total_duration: float,
) -> None:
    """Print the final scientific and technical summary."""
    status_counts = result.statistics.get(
        "status_counts",
        {},
    )

    print()
    print("=" * 78)
    print("CGIE3-ID-02 — EXECUTION COMPLETED")
    print("=" * 78)
    print(
        f"Scientific outcome: {result.status}"
    )
    print(
        "Relations evaluated: "
        f"{result.statistics.get('relation_count')}"
    )
    print(
        "Eligible: "
        f"{status_counts.get('eligible', 0)}"
    )
    print(
        "Candidate: "
        f"{status_counts.get('candidate', 0)}"
    )
    print(
        "Rejected: "
        f"{status_counts.get('rejected', 0)}"
    )
    print(
        "Non-estimable: "
        f"{status_counts.get('non_estimable', 0)}"
    )
    print()
    print("Stage durations:")

    for stage_name in EXPECTED_STAGE_ORDER:
        duration = timing.get(
            stage_name
        )

        if duration is not None:
            print(
                f"  {stage_name:<16}"
                f"{duration:>10.3f} s"
            )

    print(
        f"  {'total':<16}"
        f"{total_duration:>10.3f} s"
    )
    print()
    print("Scientific claim boundary:")
    print(
        "  Primary relations established: false"
    )
    print(
        "  Indispensable relations established: false"
    )
    print(
        "  Causality established: false"
    )
    print(
        "  Predictive capability established: false"
    )
    print(
        "  Earthquake prediction established: false"
    )
    print()
    print("Official outputs:")

    for output_name, output_data in (
        result.outputs.items()
    ):
        if isinstance(
            output_data,
            dict,
        ):
            print(
                f"  {output_name}: "
                f"{output_data.get('path')}"
            )

    print("=" * 78)


def run_pipeline() -> ExperimentResult:
    """Run the complete frozen CGIE3-ID-02 pipeline."""
    print_header()

    pipeline_started = time.perf_counter()

    timing: dict[str, float] = {}

    print_stage_start(
        1,
        "load",
    )

    load_started = time.perf_counter()

    context = load_experiment()

    load_duration = (
        time.perf_counter()
        - load_started
    )

    timing[
        "load"
    ] = float(
        load_duration
    )

    print_stage_complete(
        1,
        "load",
        load_duration,
    )

    context = execute_context_stage(
        context,
        stage_number=2,
        stage_name="preprocess",
        function=preprocess,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=3,
        stage_name="discover",
        function=discover,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=4,
        stage_name="persistence",
        function=evaluate_persistence,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=5,
        stage_name="bootstrap",
        function=evaluate_bootstrap,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=6,
        stage_name="missingness",
        function=evaluate_missingness,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=7,
        stage_name="classification",
        function=classify_relations,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=8,
        stage_name="equivalence",
        function=evaluate_equivalence,
        timing=timing,
    )

    result = execute_reporting_stage(
        context,
        stage_number=9,
        timing=timing,
    )

    result = execute_provenance_stage(
        context,
        result,
        stage_number=10,
        timing=timing,
    )

    validate_final_result(
        context,
        result,
    )

    total_duration = (
        time.perf_counter()
        - pipeline_started
    )

    context.register_runtime(
        "total_execution_seconds",
        float(total_duration),
    )

    execution_status = {
        "experiment_id":
            "CGIE3_ID_02",

        "technical_status":
            "COMPLETED",

        "scientific_outcome":
            result.status,

        "stage_order":
            list(
                EXPECTED_STAGE_ORDER
            ),

        "stage_duration_seconds":
            timing,

        "total_execution_seconds":
            float(total_duration),

        "relation_count":
            result.statistics.get(
                "relation_count"
            ),

        "status_counts":
            result.statistics.get(
                "status_counts"
            ),

        "scientific_claims":
            dict(
                result.scientific_claims
            ),

        "output_paths": {
            output_name:
                output_data.get(
                    "path"
                )
            for output_name, output_data
            in result.outputs.items()
            if isinstance(
                output_data,
                dict,
            )
        },
    }

    write_execution_status(
        execution_status
    )

    print_final_summary(
        result,
        timing,
        total_duration,
    )

    return result


def main() -> int:
    """Command-line entry point."""
    try:
        run_pipeline()

    except KNOWN_STAGE_ERRORS as exc:
        error_payload = {
            "experiment_id":
                "CGIE3_ID_02",

            "technical_status":
                "FAILED",

            "error_type":
                type(exc).__name__,

            "error_message":
                str(exc),

            "known_stage_error":
                True,
        }

        write_execution_status(
            error_payload
        )

        print()
        print("=" * 78)
        print("CGIE3-ID-02 — EXECUTION FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 78)

        return 1

    except Exception as exc:
        error_payload = {
            "experiment_id":
                "CGIE3_ID_02",

            "technical_status":
                "FAILED",

            "error_type":
                type(exc).__name__,

            "error_message":
                str(exc),

            "known_stage_error":
                False,

            "traceback":
                traceback.format_exc(),
        }

        write_execution_status(
            error_payload
        )

        print()
        print("=" * 78)
        print("CGIE3-ID-02 — UNEXPECTED FAILURE")
        print("=" * 78)
        traceback.print_exc()
        print("=" * 78)

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
  )
 
