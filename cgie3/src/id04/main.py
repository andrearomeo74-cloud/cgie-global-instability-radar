#!/usr/bin/env python3
"""
CGIE3-ID-04 pipeline entry point.

Frozen execution order:

1. load frozen inputs;
2. build relational snapshots;
3. compute relational continuity;
4. run frozen null controls;
5. apply robustness and scientific decision;
6. generate official reports;
7. generate workflow provenance.

Run from repository root:

    python -m engines.cgie3.src.id04.main
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from cgie3.src.id04.continuity import (
    ID04ContinuityError,
    compute_continuity,
)
from cgie3.src.id04.loader import (
    ID04ExperimentContext,
    ID04LoaderError,
    load_experiment,
)
from cgie3.src.id04.null_controls import (
    ID04NullControlError,
    run_null_controls,
)
from cgie3.src.id04.provenance import (
    ID04ProvenanceError,
    generate_provenance,
)
from cgie3.src.id04.reporting import (
    ID04ReportingError,
    generate_reports,
)
from cgie3.src.id04.robustness import (
    ID04RobustnessError,
    run_robustness,
)
from cgie3.src.id04.snapshots import (
    ID04SnapshotError,
    build_snapshots,
)


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]

EXECUTION_STATUS_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CGIE3_ID_04_execution_status.json"
)


STAGE_ORDER = (
    "load",
    "snapshots",
    "continuity",
    "null_controls",
    "robustness",
    "reporting",
    "provenance",
)


KNOWN_STAGE_ERRORS = (
    ID04LoaderError,
    ID04SnapshotError,
    ID04ContinuityError,
    ID04NullControlError,
    ID04RobustnessError,
    ID04ReportingError,
    ID04ProvenanceError,
)


class ID04PipelineError(RuntimeError):
    """Raised when the frozen ID-04 pipeline contract is violated."""


def print_header() -> None:
    """Print pipeline header."""
    print()
    print("=" * 78)
    print("CGIE3-ID-04 — RELATIONAL IDENTITY CONTINUITY AUDIT")
    print("=" * 78)
    print("Framework: Congruity Framework")
    print("Engine: CGIE-3")
    print("Protocol: CGIE3_ID_04_v1.0")
    print("Configuration status: FROZEN")
    print("Event alignment: PROHIBITED")
    print("=" * 78)


def print_stage_start(
    stage_number: int,
    stage_name: str,
) -> None:
    """Print stage-start marker."""
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
    """Print stage-completion marker."""
    print(
        f"[{stage_number:02d}/{len(STAGE_ORDER):02d}] "
        f"DONE  — {stage_name} "
        f"({duration_seconds:.3f} s)"
    )


def json_safe(
    value: Any,
) -> Any:
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
                json_safe(
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
            json_safe(
                item
            )
            for item in value
        ]

    return str(
        value
    )


def write_execution_status(
    payload: Mapping[str, Any],
) -> None:
    """Write official technical execution status."""
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
    context: ID04ExperimentContext,
    *,
    stage_number: int,
    stage_name: str,
    function: Callable[
        [ID04ExperimentContext],
        ID04ExperimentContext,
    ],
    timing: dict[str, float],
) -> ID04ExperimentContext:
    """Execute one context-to-context ID-04 stage."""
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
        ID04ExperimentContext,
    ):
        raise ID04PipelineError(
            f"Stage {stage_name} did not return "
            "an ID04ExperimentContext."
        )

    if updated_context is not context:
        raise ID04PipelineError(
            f"Stage {stage_name} replaced the shared "
            "ID04ExperimentContext."
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
    context: ID04ExperimentContext,
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
        raise ID04PipelineError(
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
    context: ID04ExperimentContext,
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
        raise ID04PipelineError(
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
    context: ID04ExperimentContext,
    result: Mapping[str, Any],
) -> None:
    """Validate final technical and scientific boundary."""
    if result.get(
        "experiment_id"
    ) != "CGIE3_ID_04":
        raise ID04PipelineError(
            "Final result contains unexpected experiment ID."
        )

    if result.get(
        "technical_status"
    ) != "COMPLETED":
        raise ID04PipelineError(
            "Final technical status is not COMPLETED."
        )

    allowed_outcomes = {
        "CONTINUITY_SUPPORTED",
        "PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE",
        "CONTINUITY_NOT_SUPPORTED",
        "NON_IDENTIFIABLE",
    }

    outcome = result.get(
        "scientific_outcome"
    )

    if outcome not in allowed_outcomes:
        raise ID04PipelineError(
            "Final scientific outcome is outside "
            "the frozen ID-04 outcome set."
        )

    outputs = result.get(
        "outputs"
    )

    if not isinstance(
        outputs,
        Mapping,
    ):
        raise ID04PipelineError(
            "Final result is missing output mapping."
        )

    required_outputs = {
        "snapshot_relations",
        "transition_continuity",
        "multiscale_continuity",
        "null_controls",
        "null_summaries",
        "leave_one_transition_out",
        "estimability",
        "summary",
        "report",
        "manifest",
        "workflow_provenance",
    }

    missing_outputs = sorted(
        required_outputs
        - set(
            outputs
        )
    )

    if missing_outputs:
        raise ID04PipelineError(
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
        raise ID04PipelineError(
            "Final scientific claim mapping is missing."
        )

    prohibited_true_claims = {
        "id02_statuses_modified",
        "id03_states_modified",
        "minimum_identity_core_established",
        "indispensable_relations_established",
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
        raise ID04PipelineError(
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
        raise ID04PipelineError(
            "Provenance stage is not COMPLETED."
        )

    multiscale = context.outputs.get(
        "multiscale_continuity"
    )

    if multiscale is None:
        raise ID04PipelineError(
            "Multiscale continuity output is missing."
        )

    if len(
        multiscale
    ) == 0:
        raise ID04PipelineError(
            "Multiscale continuity output is empty."
        )

    transitions = context.outputs.get(
        "transition_continuity"
    )

    if transitions is None:
        raise ID04PipelineError(
            "Transition continuity output is missing."
        )

    if len(
        transitions
    ) == 0:
        raise ID04PipelineError(
            "Transition continuity output is empty."
        )

    if (
        "event_information_used"
        in transitions.columns
        and transitions[
            "event_information_used"
        ].astype(
            bool
        ).any()
    ):
        raise ID04PipelineError(
            "Event information entered the frozen ID-04 pipeline."
        )


def print_final_summary(
    result: Mapping[str, Any],
    timing: Mapping[str, float],
    total_duration: float,
) -> None:
    """Print final ID-04 summary."""
    statistics = result.get(
        "statistics",
        {},
    )

    summary = result.get(
        "summary",
        {},
    )

    details = summary.get(
        "scientific_outcome_details",
        {},
    )

    print()
    print("=" * 78)
    print("CGIE3-ID-04 — EXECUTION COMPLETED")
    print("=" * 78)

    print(
        "Scientific outcome: "
        f"{result.get('scientific_outcome')}"
    )

    print(
        "Snapshot relation rows: "
        f"{statistics.get('snapshot_relation_rows')}"
    )

    print(
        "Transitions: "
        f"{statistics.get('transition_count')}"
    )

    print(
        "Temporal scales: "
        f"{statistics.get('scale_count')}"
    )

    print(
        "Null replications: "
        f"{statistics.get('null_replication_count')}"
    )

    print(
        "Estimable scales: "
        f"{details.get('estimable_scale_count')}"
    )

    print(
        "Supporting scales before robustness: "
        f"{details.get('supporting_scale_count_before_robustness')}"
    )

    print(
        "Supporting scales final: "
        f"{details.get('supporting_scale_count_final')}"
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
    print("  ID-03 states modified: false")
    print("  Minimum Identity Core established: false")
    print("  Indispensable relations established: false")
    print("  Causality established: false")
    print("  Predictive capability established: false")
    print("  Earthquake prediction established: false")
    print("  Universal transferability established: false")
    print("  Earthquake-event information used: false")

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
    """Run complete frozen CGIE3-ID-04 pipeline."""
    print_header()

    pipeline_started = time.perf_counter()

    timing: dict[
        str,
        float,
    ] = {}

    print_stage_start(
        1,
        "load",
    )

    started = time.perf_counter()

    context = load_experiment()

    duration = (
        time.perf_counter()
        - started
    )

    timing[
        "load"
    ] = float(
        duration
    )

    print_stage_complete(
        1,
        "load",
        duration,
    )

    context = execute_context_stage(
        context,
        stage_number=2,
        stage_name="snapshots",
        function=build_snapshots,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=3,
        stage_name="continuity",
        function=compute_continuity,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=4,
        stage_name="null_controls",
        function=run_null_controls,
        timing=timing,
    )

    context = execute_context_stage(
        context,
        stage_number=5,
        stage_name="robustness",
        function=run_robustness,
        timing=timing,
    )

    result = execute_reporting_stage(
        context,
        stage_number=6,
        timing=timing,
    )

    result = execute_provenance_stage(
        context,
        result,
        stage_number=7,
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
            "CGIE3_ID_04",

        "technical_status":
            "COMPLETED",

        "scientific_outcome":
            result.get(
                "scientific_outcome"
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
                    "CGIE3_ID_04",

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
        print("CGIE3-ID-04 — EXECUTION FAILED")
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
                    "CGIE3_ID_04",

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
        print("CGIE3-ID-04 — UNEXPECTED FAILURE")
        print("=" * 78)

        traceback.print_exc()

        print("=" * 78)

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
  )
