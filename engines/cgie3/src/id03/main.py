#!/usr/bin/env python3
"""
CGIE3-ID-03 pipeline entry point.

Frozen execution order:

1. load frozen inputs;
2. audit feature dependencies;
3. align multiscale relations;
4. audit rolling-window overlap;
5. apply primary null controls;
6. audit conditional redundancy;
7. identify relational families;
8. assign ID-03 states and select representatives;
9. generate official reports;
10. generate workflow provenance.

Run from repository root:

    python -m engines.cgie3.src.id03.main
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from engines.cgie3.src.id03.dependencies import (
    DependencyAuditError,
    audit_dependencies,
)
from engines.cgie3.src.id03.families import (
    FamilyAuditError,
    audit_families,
)
from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
    ID03LoaderError,
    load_experiment,
)
from engines.cgie3.src.id03.multiscale import (
    MultiscaleAuditError,
    audit_multiscale,
)
from engines.cgie3.src.id03.null_controls import (
    NullControlAuditError,
    audit_null_controls,
)
from engines.cgie3.src.id03.overlap import (
    OverlapAuditError,
    audit_overlap,
)
from engines.cgie3.src.id03.provenance import (
    ID03ProvenanceError,
    generate_provenance,
)
from engines.cgie3.src.id03.redundancy import (
    RedundancyAuditError,
    audit_redundancy,
)
from engines.cgie3.src.id03.reporting import (
    ID03ReportingError,
    generate_reports,
)
from engines.cgie3.src.id03.representatives import (
    RepresentativeSelectionError,
    select_representatives,
)


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]

EXECUTION_STATUS_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CGIE3_ID_03_execution_status.json"
)


STAGE_ORDER = (
    "load",
    "dependencies",
    "multiscale",
    "overlap",
    "null_controls",
    "redundancy",
    "families",
    "representatives",
    "reporting",
    "provenance",
)


KNOWN_STAGE_ERRORS = (
    ID03LoaderError,
    DependencyAuditError,
    MultiscaleAuditError,
    OverlapAuditError,
    NullControlAuditError,
    RedundancyAuditError,
    FamilyAuditError,
    RepresentativeSelectionError,
    ID03ReportingError,
    ID03ProvenanceError,
)


class ID03PipelineError(RuntimeError):
    """Raised when the ID-03 pipeline violates its execution contract."""


def print_header() -> None:
    """Print the pipeline header."""
    print()
    print("=" * 78)
    print("CGIE3-ID-03 — RELATIONAL DEPENDENCY AND FAMILY AUDIT")
    print("=" * 78)
    print("Framework: Congruity Framework")
    print("Engine: CGIE-3")
    print("Protocol: CGIE3_ID_03_v1.0")
    print("Configuration status: FROZEN")
    print("=" * 78)


def print_stage_start(
    stage_number: int,
    stage_name: str,
) -> None:
    """Print one stage-start marker."""
    print()
    print(
        f"[{stage_number:02d}/{len(STAGE_ORDER):02d}] "
        f"START — {stage_name}"
    )


def print_stage_complete(
    stage_number: int,
    stage_name: str,
    duration_seconds: float,
) -> None:
    """Print one stage-completion marker."""
    print(
        f"[{stage_number:02d}/{len(STAGE_ORDER):02d}] "
        f"DONE  — {stage_name} "
        f"({duration_seconds:.3f} s)"
    )


def json_safe(value: Any) -> Any:
    """Convert execution-status values to strict JSON-compatible data."""
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
        Mapping,
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
    payload: Mapping[str, Any],
) -> None:
    """Write the official technical execution status."""
    EXECUTION_STATUS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXECUTION_STATUS_PATH.write_text(
        json.dumps(
            json_safe(
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


def execute_context_stage(
    context: ID03ExperimentContext,
    *,
    stage_number: int,
    stage_name: str,
    function: Callable[
        [ID03ExperimentContext],
        ID03ExperimentContext,
    ],
    timing: dict[str, float],
) -> ID03ExperimentContext:
    """
    Execute one stage receiving and returning the shared ID-03 context.
    """
    print_stage_start(
        stage_number,
        stage_name,
    )

    started = time.perf_counter()

    updated_context = function(
        context
    )

    duration = (
        time.perf_counter()
        - started
    )

    if not isinstance(
        updated_context,
        ID03ExperimentContext,
    ):
        raise ID03PipelineError(
            f"Stage {stage_name} did not return "
            "an ID03ExperimentContext."
        )

    if updated_context is not context:
        raise ID03PipelineError(
            f"Stage {stage_name} replaced the shared "
            "ID03ExperimentContext."
        )

    timing[
        stage_name
    ] = float(
        duration
    )

    print_stage_complete(
        stage_number,
        stage_name,
        duration,
    )

    return updated_context


def execute_reporting_stage(
    context: ID03ExperimentContext,
    *,
    stage_number: int,
    timing: dict[str, float],
) -> dict[str, Any]:
    """Execute official scientific reporting."""
    stage_name = "reporting"

    print_stage_start(
        stage_number,
        stage_name,
    )

    started = time.perf_counter()

    result = generate_reports(
        context
    )

    duration = (
        time.perf_counter()
        - started
    )

    if not isinstance(
        result,
        dict,
    ):
        raise ID03PipelineError(
            "Reporting did not return a result mapping."
        )

    timing[
        stage_name
    ] = float(
        duration
    )

    print_stage_complete(
        stage_number,
        stage_name,
        duration,
    )

    return result


def execute_provenance_stage(
    context: ID03ExperimentContext,
    result: Mapping[str, Any],
    *,
    stage_number: int,
    timing: dict[str, float],
) -> dict[str, Any]:
    """Execute official workflow provenance."""
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
        dict,
    ):
        raise ID03PipelineError(
            "Provenance did not return a result mapping."
        )

    timing[
        stage_name
    ] = float(
        duration
    )

    print_stage_complete(
        stage_number,
        stage_name,
        duration,
    )

    return final_result


def validate_final_result(
    context: ID03ExperimentContext,
    result: Mapping[str, Any],
) -> None:
    """Validate the final ID-03 result and scientific boundary."""
    if result.get(
        "experiment_id"
    ) != "CGIE3_ID_03":
        raise ID03PipelineError(
            "Final result contains an unexpected experiment ID."
        )

    if result.get(
        "technical_status"
    ) != "COMPLETED":
        raise ID03PipelineError(
            "Final technical status is not COMPLETED."
        )

    outputs = result.get(
        "outputs"
    )

    if not isinstance(
        outputs,
        Mapping,
    ):
        raise ID03PipelineError(
            "Final result is missing its output mapping."
        )

    required_outputs = {
        "feature_dependencies",
        "relation_dependencies",
        "multiscale_relations",
        "overlap_sensitivity",
        "overlap_estimates_long",
        "null_controls",
        "null_control_summaries",
        "conditional_redundancy",
        "relation_families",
        "family_membership",
        "family_coassignment",
        "id03_relation_states",
        "representative_candidates",
        "family_graph",
        "summary",
        "report",
        "manifest",
        "workflow_provenance",
    }

    missing_outputs = sorted(
        required_outputs
        - set(outputs)
    )

    if missing_outputs:
        raise ID03PipelineError(
            "Final result is missing outputs: "
            + ", ".join(
                missing_outputs
            )
        )

    claims = result.get(
        "scientific_claims"
    )

    if not isinstance(
        claims,
        Mapping,
    ):
        raise ID03PipelineError(
            "Final result is missing scientific claims."
        )

    prohibited_true_claims = {
        "id02_statuses_modified",
        "primary_relations_established",
        "indispensable_relations_established",
        "minimum_identity_core_established",
        "causality_established",
        "predictive_capability_established",
        "earthquake_prediction_established",
        "universal_transferability_established",
    }

    invalid_claims = sorted(
        claim
        for claim in prohibited_true_claims
        if claims.get(
            claim
        )
        is True
    )

    if invalid_claims:
        raise ID03PipelineError(
            "Final result violates the scientific boundary: "
            + ", ".join(
                invalid_claims
            )
        )

    if (
        context.runtime.get(
            "provenance_status"
        )
        != "COMPLETED"
    ):
        raise ID03PipelineError(
            "Provenance is not marked COMPLETED."
        )

    states = context.outputs.get(
        "id03_relation_states"
    )

    if states is None or len(states) != 74:
        raise ID03PipelineError(
            "Final ID-03 state table must contain 74 relations."
        )

    representatives = context.outputs.get(
        "representative_candidates"
    )

    if representatives is None:
        raise ID03PipelineError(
            "Representative-candidate output is missing."
        )


def print_final_summary(
    result: Mapping[str, Any],
    timing: Mapping[str, float],
    total_duration: float,
) -> None:
    """Print the final technical and scientific summary."""
    statistics = result.get(
        "statistics",
        {},
    )

    state_counts = statistics.get(
        "id03_state_counts",
        {},
    )

    print()
    print("=" * 78)
    print("CGIE3-ID-03 — EXECUTION COMPLETED")
    print("=" * 78)
    print(
        "Scientific outcome: "
        f"{result.get('status')}"
    )
    print(
        "Primary relations audited: "
        f"{statistics.get('primary_relation_count')}"
    )
    print(
        "Reproducible families: "
        f"{statistics.get('reproducible_family_count')}"
    )
    print(
        "Representative candidates: "
        f"{statistics.get('representative_candidate_count')}"
    )
    print()
    print("ID-03 states:")

    for state_name in (
        "family_representative_candidate",
        "supporting_relation",
        "definitionally_constrained",
        "redundant_relation",
        "overlap_sensitive",
        "scale_inconsistent",
        "insufficient_evidence",
    ):
        print(
            f"  {state_name:<34}"
            f"{state_counts.get(state_name, 0):>6}"
        )

    print()
    print("Stage durations:")

    for stage_name in STAGE_ORDER:
        duration = timing.get(
            stage_name
        )

        if duration is not None:
            print(
                f"  {stage_name:<18}"
                f"{duration:>10.3f} s"
            )

    print(
        f"  {'total':<18}"
        f"{total_duration:>10.3f} s"
    )

    print()
    print("Scientific claim boundary:")
    print("  ID-02 statuses modified: false")
    print("  Primary relations established: false")
    print("  Indispensable relations established: false")
    print("  Minimum Identity Core established: false")
    print("  Causality established: false")
    print("  Predictive capability established: false")
    print("  Earthquake prediction established: false")
    print("  Universal transferability established: false")
    print()
    print("Official outputs:")

    for output_name, output_data in result.get(
        "outputs",
        {},
    ).items():
        if isinstance(
            output_data,
            Mapping,
        ):
            print(
                f"  {output_name}: "
                f"{output_data.get('path')}"
            )

    print("=" * 78)


def run_pipeline() -> dict[str, Any]:
    """Run the complete frozen CGIE3-ID-03 pipeline."""
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
        stage_name="dependencies",
        function=audit_dependencies,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=3,
        stage_name="multiscale",
        function=audit_multiscale,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=4,
        stage_name="overlap",
        function=audit_overlap,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=5,
        stage_name="null_controls",
        function=audit_null_controls,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=6,
        stage_name="redundancy",
        function=audit_redundancy,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=7,
        stage_name="families",
        function=audit_families,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=8,
        stage_name="representatives",
        function=select_representatives,
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
        float(
            total_duration
        ),
    )

    execution_status = {
        "experiment_id":
            "CGIE3_ID_03",

        "technical_status":
            "COMPLETED",

        "scientific_outcome":
            result.get(
                "status"
            ),

        "stage_order":
            list(
                STAGE_ORDER
            ),

        "stage_duration_seconds":
            timing,

        "total_execution_seconds":
            float(
                total_duration
            ),

        "statistics":
            result.get(
                "statistics"
            ),

        "scientific_claims":
            result.get(
                "scientific_claims"
            ),

        "output_paths": {
            output_name:
                output_data.get(
                    "path"
                )
            for output_name, output_data
            in result.get(
                "outputs",
                {},
            ).items()
            if isinstance(
                output_data,
                Mapping,
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
        write_execution_status(
            {
                "experiment_id":
                    "CGIE3_ID_03",

                "technical_status":
                    "FAILED",

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error_message":
                    str(
                        exc
                    ),

                "known_stage_error":
                    True,
            }
        )

        print()
        print("=" * 78)
        print("CGIE3-ID-03 — EXECUTION FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 78)

        return 1

    except Exception as exc:
        write_execution_status(
            {
                "experiment_id":
                    "CGIE3_ID_03",

                "technical_status":
                    "FAILED",

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error_message":
                    str(
                        exc
                    ),

                "known_stage_error":
                    False,

                "traceback":
                    traceback.format_exc(),
            }
        )

        print()
        print("=" * 78)
        print("CGIE3-ID-03 — UNEXPECTED FAILURE")
        print("=" * 78)
        traceback.print_exc()
        print("=" * 78)

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    ) 
