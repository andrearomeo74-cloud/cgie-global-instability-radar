"""
CGIE3-ID-03 scientific reporting.

This module serializes the frozen outputs produced by the ID-03
dependency, multiscale, overlap, null, redundancy, family and
representative-selection stages.

It does not:

- recompute scientific results;
- alter ID-02 classifications;
- change ID-03 states;
- change frozen thresholds;
- remove negative or inconclusive outcomes;
- infer causality, indispensability or earthquake prediction.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class ID03ReportingError(ValueError):
    """Raised when ID-03 reporting violates the frozen contract."""


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs"


OUTPUT_PATHS = {
    "feature_dependencies":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_feature_dependencies.csv",

    "relation_dependencies":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_relation_dependencies.csv",

    "multiscale_relations":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_multiscale_relations.csv",

    "overlap_sensitivity":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_overlap_sensitivity.csv",

    "overlap_estimates_long":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_overlap_estimates_long.csv",

    "null_controls":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_null_controls.csv",

    "null_control_summaries":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_null_control_summaries.csv",

    "conditional_redundancy":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_conditional_redundancy.csv",

    "relation_families":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_relation_families.csv",

    "family_membership":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_family_membership.csv",

    "family_coassignment":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_family_coassignment.csv",

    "id03_relation_states":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_relation_states.csv",

    "representative_candidates":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_representative_candidates.csv",

    "family_graph":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_family_graph.graphml",

    "summary":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_summary.json",

    "report":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_report.md",

    "manifest":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_03_manifest.json",
}


def fail(message: str) -> None:
    """Raise a normalized reporting error."""
    raise ID03ReportingError(
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
    """Require a mapping-like value."""
    if not isinstance(
        value,
        Mapping,
    ):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def require_dataframe(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Require a DataFrame and optionally an exact row count."""
    if not isinstance(
        value,
        pd.DataFrame,
    ):
        fail(
            f"{field_name} must be a pandas DataFrame."
        )

    if (
        not allow_empty
        and value.empty
    ):
        fail(
            f"{field_name} must not be empty."
        )

    if (
        expected_rows is not None
        and len(value) != expected_rows
    ):
        fail(
            f"{field_name} must contain "
            f"{expected_rows} rows; observed "
            f"{len(value)}."
        )

    return value


def validate_context(
    context: ID03ExperimentContext,
) -> None:
    """Validate all prerequisites for official reporting."""
    if not isinstance(
        context,
        ID03ExperimentContext,
    ):
        fail(
            "context must be an ID03ExperimentContext."
        )

    if context.experiment_id != "CGIE3_ID_03":
        fail(
            "Unexpected experiment ID: "
            f"{context.experiment_id}"
        )

    required_runtime = {
        "loader_status",
        "dependency_audit_status",
        "multiscale_audit_status",
        "overlap_audit_status",
        "null_control_audit_status",
        "conditional_redundancy_status",
        "family_audit_status",
        "representative_selection_status",
    }

    missing_runtime = sorted(
        required_runtime
        - set(context.runtime)
    )

    if missing_runtime:
        fail(
            "Reporting runtime prerequisites are missing: "
            + ", ".join(missing_runtime)
        )

    incomplete_runtime = sorted(
        stage_name
        for stage_name in required_runtime
        if context.runtime.get(stage_name)
        != "COMPLETED"
    )

    if incomplete_runtime:
        fail(
            "Some ID-03 stages are not completed: "
            + ", ".join(incomplete_runtime)
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
        "family_graph",
        "id03_relation_states",
        "representative_candidates",
        "dependency_audit_summary",
        "multiscale_audit_summary",
        "overlap_audit_summary",
        "null_control_audit_summary",
        "conditional_redundancy_summary",
        "family_audit_summary",
        "representative_selection_summary",
    }

    missing_outputs = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing_outputs:
        fail(
            "Reporting scientific outputs are missing: "
            + ", ".join(missing_outputs)
        )


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for one existing file."""
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


def json_compatible(value: Any) -> Any:
    """Convert scientific objects into strict JSON-compatible values."""
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

    if is_dataclass(value):
        return json_compatible(
            asdict(value)
        )

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


def write_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    """Write deterministic UTF-8 CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = frame.copy()

    for column in output.columns:
        contains_structured_value = output[
            column
        ].map(
            lambda value: isinstance(
                value,
                (
                    dict,
                    list,
                    tuple,
                    set,
                ),
            )
        ).any()

        if contains_structured_value:
            output[column] = output[
                column
            ].map(
                lambda value: json.dumps(
                    json_compatible(value),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if isinstance(
                    value,
                    (
                        dict,
                        list,
                        tuple,
                        set,
                    ),
                )
                else value
            )

    output.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.10g",
    )


def graphml_safe_value(value: Any) -> Any:
    """Convert graph attributes to GraphML-compatible scalar values."""
    if value is None:
        return ""

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
            return ""

        return normalized

    return json.dumps(
        json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
    )


def write_graphml(
    path: Path,
    graph: nx.Graph,
) -> None:
    """Write a sanitized deterministic GraphML representation."""
    if not isinstance(
        graph,
        nx.Graph,
    ):
        fail(
            "family_graph must be a networkx Graph."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sanitized = nx.Graph()

    for node_id, attributes in sorted(
        graph.nodes(
            data=True
        ),
        key=lambda item: str(
            item[0]
        ),
    ):
        sanitized.add_node(
            str(node_id),
            **{
                str(key): graphml_safe_value(value)
                for key, value in attributes.items()
            },
        )

    for source_id, target_id, attributes in sorted(
        graph.edges(
            data=True
        ),
        key=lambda item: (
            str(item[0]),
            str(item[1]),
        ),
    ):
        sanitized.add_edge(
            str(source_id),
            str(target_id),
            **{
                str(key): graphml_safe_value(value)
                for key, value in attributes.items()
            },
        )

    nx.write_graphml(
        sanitized,
        path,
        encoding="utf-8",
        prettyprint=True,
    )


def validate_output_shapes(
    context: ID03ExperimentContext,
) -> None:
    """Validate official output row counts before writing."""
    require_dataframe(
        context.outputs[
            "feature_dependencies"
        ],
        "feature_dependencies",
        expected_rows=9,
    )

    require_dataframe(
        context.outputs[
            "relation_dependencies"
        ],
        "relation_dependencies",
        expected_rows=136,
    )

    require_dataframe(
        context.outputs[
            "overlap_sensitivity"
        ],
        "overlap_sensitivity",
        expected_rows=74,
    )

    require_dataframe(
        context.outputs[
            "overlap_estimates_long"
        ],
        "overlap_estimates_long",
        expected_rows=222,
    )

    require_dataframe(
        context.outputs[
            "null_controls"
        ],
        "null_controls",
        expected_rows=74,
    )

    require_dataframe(
        context.outputs[
            "null_control_summaries"
        ],
        "null_control_summaries",
        expected_rows=222,
    )

    require_dataframe(
        context.outputs[
            "conditional_redundancy"
        ],
        "conditional_redundancy",
        expected_rows=74,
    )

    require_dataframe(
        context.outputs[
            "family_membership"
        ],
        "family_membership",
        expected_rows=74,
    )

    require_dataframe(
        context.outputs[
            "id03_relation_states"
        ],
        "id03_relation_states",
        expected_rows=74,
    )

    require_dataframe(
        context.outputs[
            "representative_candidates"
        ],
        "representative_candidates",
        allow_empty=True,
    )

    null_replications = require_dataframe(
        context.outputs[
            "null_control_replications"
        ],
        "null_control_replications",
        expected_rows=66600,
    )

    if len(null_replications) != 66600:
        fail(
            "Unexpected null replication count."
        )


def build_scientific_claims() -> dict[str, bool]:
    """Return the frozen ID-03 scientific claim boundary."""
    return {
        "feature_dependencies_audited":
            True,

        "multiscale_relations_audited":
            True,

        "overlap_sensitivity_audited":
            True,

        "null_controls_applied":
            True,

        "conditional_redundancy_audited":
            True,

        "relational_families_evaluated":
            True,

        "family_representative_candidates_evaluated":
            True,

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
    }


def build_summary(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Build the official CGIE3-ID-03 summary."""
    selection_summary = require_mapping(
        context.outputs[
            "representative_selection_summary"
        ],
        "representative_selection_summary",
    )

    return {
        "experiment_id":
            "CGIE3_ID_03",

        "experiment_name":
            "Relational Dependency and Family Audit",

        "engine":
            "CGIE-3",

        "framework":
            "Congruity Framework",

        "generated_at_utc":
            utc_now_iso(),

        "technical_status":
            "COMPLETED",

        "technical_success":
            bool(
                selection_summary[
                    "technical_success"
                ]
            ),

        "scientific_outcome":
            str(
                selection_summary[
                    "scientific_outcome"
                ]
            ),

        "scientific_positive_result":
            bool(
                selection_summary[
                    "scientific_positive_result"
                ]
            ),

        "id02_population": {
            "total_relations":
                136,

            "eligible":
                45,

            "candidate":
                29,

            "rejected":
                62,

            "non_estimable":
                0,

            "primary_audit_population":
                74,
        },

        "id03_state_counts":
            selection_summary[
                "id03_state_counts"
            ],

        "reproducible_family_count":
            int(
                selection_summary[
                    "reproducible_family_count"
                ]
            ),

        "representative_candidate_count":
            int(
                selection_summary[
                    "representative_candidate_count"
                ]
            ),

        "dependency_audit":
            context.outputs[
                "dependency_audit_summary"
            ],

        "multiscale_audit":
            context.outputs[
                "multiscale_audit_summary"
            ],

        "overlap_audit":
            context.outputs[
                "overlap_audit_summary"
            ],

        "null_control_audit":
            context.outputs[
                "null_control_audit_summary"
            ],

        "conditional_redundancy":
            context.outputs[
                "conditional_redundancy_summary"
            ],

        "family_audit":
            context.outputs[
                "family_audit_summary"
            ],

        "representative_selection":
            selection_summary,

        "scientific_claims":
            build_scientific_claims(),

        "valid_negative_result":
            True,

        "claim_boundary": (
            "CGIE3-ID-03 may identify reproducible relational "
            "families and family-representative candidates. "
            "These outcomes do not establish primary identity "
            "relations, indispensability, causality, predictive "
            "capability, earthquake prediction or universal "
            "transferability."
        ),
    }


def markdown_count_rows(
    values: Mapping[str, Any],
) -> str:
    """Convert one count mapping to Markdown table rows."""
    return "\n".join(
        f"| `{key}` | {value} |"
        for key, value in values.items()
    )


def build_family_markdown(
    families: pd.DataFrame,
) -> str:
    """Build the relational-family report section."""
    if families.empty:
        return (
            "No relational family was produced by the "
            "data-driven family audit."
        )

    sections: list[str] = []

    for row in families.itertuples(
        index=False
    ):
        sections.append(
            f"""### `{row.family_id}`

- Components: `{row.member_components}`
- Component count: `{int(row.member_node_count)}`
- Pair count: `{int(row.member_relation_pair_count)}`
- Scale-specific relation count: `{int(row.member_scale_relation_count)}`
- Eligible relations: `{int(row.eligible_relation_count)}`
- Candidate relations: `{int(row.candidate_relation_count)}`
- Family stability: `{float(row.family_stability):.4f}`
- Reproducible family: `{str(bool(row.reproducible_family)).lower()}`
- Secondary-partition agreement: `{str(bool(row.secondary_partition_agreement)).lower()}`
- Preliminary label suggestion: `{row.preliminary_label_suggestion}`
- Label used for detection: `false`
"""
        )

    return "\n".join(
        sections
    )


def build_representative_markdown(
    representatives: pd.DataFrame,
) -> str:
    """Build the representative-candidate report section."""
    if representatives.empty:
        return (
            "No relation satisfied every frozen condition for "
            "family-representative candidacy."
        )

    lines: list[str] = []

    for row in representatives.itertuples(
        index=False
    ):
        lines.append(
            "- "
            f"`{row.family_id}` — "
            f"`{row.window_id}` — "
            f"`{row.source_id}` ↔ `{row.target_id}`; "
            f"ID-02 status `{row.classification_status}`; "
            f"ρ = `{float(row.strength):.4f}`; "
            f"supported scales = `{int(row.supported_scale_count)}`; "
            f"overlap = `{row.overlap_class}`; "
            f"null outcome = `{row.null_outcome}`; "
            f"redundancy = `{row.redundancy_status}`."
        )

    return "\n".join(
        lines
    )


def build_report(
    summary: Mapping[str, Any],
    context: ID03ExperimentContext,
) -> str:
    """Build the official human-readable scientific report."""
    state_counts = require_mapping(
        summary[
            "id03_state_counts"
        ],
        "summary.id03_state_counts",
    )

    families = require_dataframe(
        context.outputs[
            "relation_families"
        ],
        "relation_families",
    )

    representatives = require_dataframe(
        context.outputs[
            "representative_candidates"
        ],
        "representative_candidates",
        allow_empty=True,
    )

    state_rows = markdown_count_rows(
        state_counts
    )

    family_text = build_family_markdown(
        families
    )

    representative_text = (
        build_representative_markdown(
            representatives
        )
    )

    return f"""# CGIE3-ID-03 — Relational Dependency and Family Audit

## Technical status

`{summary["technical_status"]}`

## Scientific outcome

`{summary["scientific_outcome"]}`

## Purpose

CGIE3-ID-03 evaluates whether the relations retained by
CGIE3-ID-02 represent distinct structural information or multiple
projections of shared feature definitions, overlapping rolling
windows and redundant feature-generating processes.

ID-03 does not search for earthquake prediction.

## Frozen population

- Total ID-02 relations preserved: `136`
- Eligible: `45`
- Candidate: `29`
- Rejected reference relations: `62`
- Primary ID-03 audit population: `74`

ID-02 statuses were not modified.

## Final ID-03 states

| State | Count |
|---|---:|
{state_rows}

## Relational families

Reproducible family count:
`{summary["reproducible_family_count"]}`

{family_text}

## Family-representative candidates

Representative-candidate count:
`{summary["representative_candidate_count"]}`

{representative_text}

## Mandatory audits

The experiment completed the following audits:

1. frozen feature dependency and lineage;
2. multiscale relation alignment;
3. reduced-overlap and non-overlapping sampling;
4. randomized null controls;
5. conditional redundancy;
6. weighted relational-family detection;
7. representative-candidate selection.

## Interpretation

A `family_representative_candidate` is the highest-ranked relation
inside a reproducible family that satisfies every frozen selection
condition.

It is not automatically:

- a primary identity relation;
- an indispensable relation;
- a causal mechanism;
- an earthquake precursor;
- predictive of a specific event;
- universally transferable.

## Claim boundary

{summary["claim_boundary"]}

## Explicit negative claims

- ID-02 statuses modified: `false`
- Primary relations established: `false`
- Indispensable relations established: `false`
- Minimum Identity Core established: `false`
- Causality established: `false`
- Predictive capability established: `false`
- Earthquake prediction established: `false`
- Universal transferability established: `false`

## Valid negative result

The absence of reproducible families or representative candidates
would remain a valid scientific result and would not authorize
retrospective threshold changes.

## Generated

`{summary["generated_at_utc"]}`
"""


def build_manifest(
    context: ID03ExperimentContext,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the official manifest before final file hashes."""
    experiment = require_mapping(
        context.configuration[
            "experiment"
        ],
        "experiment",
    )

    return {
        "experiment_id":
            context.experiment_id,

        "experiment_name":
            experiment.get(
                "name"
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

        "generated_at_utc":
            summary[
                "generated_at_utc"
            ],

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
        },

        "runtime":
            context.runtime,

        "metadata":
            context.metadata,

        "input_provenance":
            context.provenance.get(
                "loader",
                {},
            ),

        "scientific_outcome":
            summary[
                "scientific_outcome"
            ],

        "scientific_claims":
            summary[
                "scientific_claims"
            ],

        "software": {
            "python_version":
                platform.python_version(),

            "python_implementation":
                platform.python_implementation(),

            "python_executable":
                sys.executable,

            "platform":
                platform.platform(),

            "numpy_version":
                np.__version__,

            "pandas_version":
                pd.__version__,

            "networkx_version":
                nx.__version__,
        },

        "output_files": {
            output_id: str(
                output_path.relative_to(
                    REPOSITORY_ROOT
                )
            )
            for output_id, output_path
            in OUTPUT_PATHS.items()
        },
    }


def write_official_outputs(
    context: ID03ExperimentContext,
    summary: Mapping[str, Any],
    report_text: str,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Write every official ID-03 output and return its SHA-256."""
    dataframe_outputs = {
        "feature_dependencies":
            context.outputs[
                "feature_dependencies"
            ],

        "relation_dependencies":
            context.outputs[
                "relation_dependencies"
            ],

        "multiscale_relations":
            context.outputs[
                "multiscale_relations"
            ],

        "overlap_sensitivity":
            context.outputs[
                "overlap_sensitivity"
            ],

        "overlap_estimates_long":
            context.outputs[
                "overlap_estimates_long"
            ],

        "null_controls":
            context.outputs[
                "null_controls"
            ],

        "null_control_summaries":
            context.outputs[
                "null_control_summaries"
            ],

        "conditional_redundancy":
            context.outputs[
                "conditional_redundancy"
            ],

        "relation_families":
            context.outputs[
                "relation_families"
            ],

        "family_membership":
            context.outputs[
                "family_membership"
            ],

        "family_coassignment":
            context.outputs[
                "family_coassignment"
            ],

        "id03_relation_states":
            context.outputs[
                "id03_relation_states"
            ],

        "representative_candidates":
            context.outputs[
                "representative_candidates"
            ],
    }

    for output_id, frame in dataframe_outputs.items():
        write_csv(
            OUTPUT_PATHS[
                output_id
            ],
            require_dataframe(
                frame,
                output_id,
                allow_empty=(
                    output_id
                    == "representative_candidates"
                ),
            ),
        )

    write_graphml(
        OUTPUT_PATHS[
            "family_graph"
        ],
        context.outputs[
            "family_graph"
        ],
    )

    write_json(
        OUTPUT_PATHS[
            "summary"
        ],
        summary,
    )

    OUTPUT_PATHS[
        "report"
    ].write_text(
        report_text,
        encoding="utf-8",
    )

    write_json(
        OUTPUT_PATHS[
            "manifest"
        ],
        manifest,
    )

    return {
        output_id:
            sha256_file(
                output_path
            )
        for output_id, output_path
        in OUTPUT_PATHS.items()
    }


def generate_reports(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """
    Generate official CGIE3-ID-03 outputs.

    Returns a JSON-compatible result mapping. Workflow provenance is
    added by the following dedicated stage.
    """
    validate_context(
        context
    )

    validate_output_shapes(
        context
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = build_summary(
        context
    )

    report_text = build_report(
        summary,
        context,
    )

    manifest = build_manifest(
        context,
        summary,
    )

    initial_hashes = write_official_outputs(
        context,
        summary,
        report_text,
        manifest,
    )

    manifest_with_hashes = dict(
        manifest
    )

    manifest_with_hashes[
        "output_sha256"
    ] = initial_hashes

    write_json(
        OUTPUT_PATHS[
            "manifest"
        ],
        manifest_with_hashes,
    )

    final_hashes = dict(
        initial_hashes
    )

    final_hashes[
        "manifest"
    ] = sha256_file(
        OUTPUT_PATHS[
            "manifest"
        ]
    )

    result = {
        "experiment_id":
            "CGIE3_ID_03",

        "status":
            summary[
                "scientific_outcome"
            ],

        "technical_status":
            "COMPLETED",

        "summary":
            summary,

        "scientific_claims":
            summary[
                "scientific_claims"
            ],

        "outputs": {
            output_id: {
                "path":
                    str(
                        OUTPUT_PATHS[
                            output_id
                        ].relative_to(
                            REPOSITORY_ROOT
                        )
                    ),

                "sha256":
                    digest,
            }
            for output_id, digest
            in final_hashes.items()
        },

        "statistics": {
            "primary_relation_count":
                74,

            "id03_state_counts":
                summary[
                    "id03_state_counts"
                ],

            "reproducible_family_count":
                summary[
                    "reproducible_family_count"
                ],

            "representative_candidate_count":
                summary[
                    "representative_candidate_count"
                ],
        },

        "warnings": [
            (
                "Family-representative candidacy does not establish "
                "primary identity status or indispensability."
            ),

            (
                "CGIE3-ID-03 does not establish causality, "
                "predictive capability or earthquake prediction."
            ),
        ],
    }

    context.register_output(
        "official_summary",
        summary,
    )

    context.register_output(
        "official_manifest",
        manifest_with_hashes,
    )

    context.register_output(
        "official_output_hashes",
        final_hashes,
    )

    context.register_output(
        "official_result",
        result,
    )

    context.register_runtime(
        "reporting_status",
        "COMPLETED",
    )

    return result
