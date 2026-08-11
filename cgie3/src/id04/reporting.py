"""
CGIE3-ID-04 scientific reporting.

This module serializes frozen outputs produced by the ID-04
snapshot, continuity, null-control and robustness stages.

It does not:

- recompute scientific metrics;
- change thresholds;
- change temporal scales;
- change ID-02 classifications;
- change ID-03 states;
- use earthquake-event information;
- infer causality or prediction.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from engines.cgie3.src.id04.loader import (
    ID04ExperimentContext,
)


class ID04ReportingError(ValueError):
    """Raised when ID-04 reporting violates the frozen contract."""


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs"


OUTPUT_PATHS = {
    "snapshot_relations":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_snapshot_relations.csv",

    "transition_continuity":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_transition_continuity.csv",

    "multiscale_continuity":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_multiscale_continuity.csv",

    "null_controls":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_null_controls.csv",

    "null_summaries":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_null_summaries.csv",

    "leave_one_transition_out":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_leave_one_transition_out.csv",

    "estimability":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_estimability.csv",

    "summary":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_summary.json",

    "report":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_report.md",

    "manifest":
        OUTPUT_DIRECTORY
        / "CGIE3_ID_04_manifest.json",
}


ALLOWED_OUTCOMES = {
    "CONTINUITY_SUPPORTED",
    "PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE",
    "CONTINUITY_NOT_SUPPORTED",
    "NON_IDENTIFIABLE",
}


def fail(message: str) -> None:
    """Raise normalized reporting error."""
    raise ID04ReportingError(
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


def require_dataframe(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Require a DataFrame."""
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

    return value


def validate_context(
    context: ID04ExperimentContext,
) -> None:
    """Validate prerequisites for official reporting."""
    if not isinstance(
        context,
        ID04ExperimentContext,
    ):
        fail(
            "context must be an ID04ExperimentContext."
        )

    if context.experiment_id != "CGIE3_ID_04":
        fail(
            "Unexpected experiment ID."
        )

    required_runtime = {
        "loader_status",
        "snapshot_status",
        "continuity_status",
        "null_control_status",
        "robustness_status",
        "scientific_outcome",
    }

    missing_runtime = sorted(
        required_runtime
        - set(
            context.runtime
        )
    )

    if missing_runtime:
        fail(
            "Reporting runtime prerequisites missing: "
            + ", ".join(
                missing_runtime
            )
        )

    completed_stages = (
        "loader_status",
        "snapshot_status",
        "continuity_status",
        "null_control_status",
        "robustness_status",
    )

    incomplete = [
        stage
        for stage in completed_stages
        if context.runtime.get(
            stage
        )
        != "COMPLETED"
    ]

    if incomplete:
        fail(
            "Incomplete ID-04 stages: "
            + ", ".join(
                incomplete
            )
        )

    outcome = str(
        context.runtime[
            "scientific_outcome"
        ]
    )

    if outcome not in ALLOWED_OUTCOMES:
        fail(
            "Unexpected scientific outcome: "
            f"{outcome}"
        )

    required_outputs = {
        "snapshot_relations",
        "snapshots",
        "snapshot_summary",
        "transition_continuity",
        "continuity_summary",
        "null_controls",
        "null_summaries",
        "null_control_summary",
        "multiscale_continuity",
        "leave_one_transition_out",
        "scientific_outcome_details",
        "robustness_summary",
    }

    missing_outputs = sorted(
        required_outputs
        - set(
            context.outputs
        )
    )

    if missing_outputs:
        fail(
            "Reporting outputs missing: "
            + ", ".join(
                missing_outputs
            )
        )


def json_compatible(
    value: Any,
) -> Any:
    """Convert scientific objects to strict JSON-compatible values."""
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


def write_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    """Write deterministic UTF-8 CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.10g",
    )


def sha256_file(
    path: Path,
) -> str:
    """Calculate SHA-256 for one official file."""
    if not path.exists():
        fail(
            f"Cannot hash missing file: {path}"
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


def build_estimability_table(
    context: ID04ExperimentContext,
) -> pd.DataFrame:
    """Build official ID-04 estimability audit."""
    snapshots = require_dataframe(
        context.outputs[
            "snapshots"
        ],
        "snapshots",
    )

    transitions = require_dataframe(
        context.outputs[
            "transition_continuity"
        ],
        "transition_continuity",
    )

    records: list[
        dict[str, Any]
    ] = []

    for scale_id, frame in snapshots.groupby(
        "scale_id",
        sort=True,
    ):
        transition_frame = transitions.loc[
            transitions[
                "scale_id"
            ]
            == scale_id
        ]

        estimable_transitions = int(
            transition_frame[
                "RCS_estimable"
            ]
            .astype(
                bool
            )
            .sum()
        )

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_04",

                "scale_id":
                    str(
                        scale_id
                    ),

                "snapshot_count":
                    int(
                        len(
                            frame
                        )
                    ),

                "transition_count":
                    int(
                        len(
                            transition_frame
                        )
                    ),

                "estimable_transition_count":
                    estimable_transitions,

                "non_estimable_transition_count":
                    int(
                        len(
                            transition_frame
                        )
                        - estimable_transitions
                    ),

                "mean_snapshot_estimable_relation_fraction":
                    float(
                        frame[
                            "estimable_relation_fraction"
                        ].mean()
                    ),

                "scale_estimable":
                    bool(
                        estimable_transitions
                        >= int(
                            context.configuration[
                                "estimability"
                            ][
                                "minimum_estimable_transition_count_per_scale"
                            ]
                        )
                    ),
            }
        )

    return pd.DataFrame.from_records(
        records
    ).sort_values(
        by="scale_id",
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_scientific_claims(
    outcome: str,
) -> dict[str, bool]:
    """Return frozen scientific claim boundary."""
    return {
        "relational_continuity_supported":
            bool(
                outcome
                == "CONTINUITY_SUPPORTED"
            ),

        "partial_or_scale_specific_evidence":
            bool(
                outcome
                == "PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE"
            ),

        "continuity_not_supported":
            bool(
                outcome
                == "CONTINUITY_NOT_SUPPORTED"
            ),

        "non_identifiable":
            bool(
                outcome
                == "NON_IDENTIFIABLE"
            ),

        "id02_statuses_modified":
            False,

        "id03_states_modified":
            False,

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
    }


def build_summary(
    context: ID04ExperimentContext,
    estimability: pd.DataFrame,
) -> dict[str, Any]:
    """Build official ID-04 summary."""
    outcome = str(
        context.runtime[
            "scientific_outcome"
        ]
    )

    outcome_details = require_mapping(
        context.outputs[
            "scientific_outcome_details"
        ],
        "scientific_outcome_details",
    )

    multiscale = require_dataframe(
        context.outputs[
            "multiscale_continuity"
        ],
        "multiscale_continuity",
    )

    return {
        "experiment_id":
            "CGIE3_ID_04",

        "experiment_name":
            "Relational Identity Continuity Audit",

        "protocol_version":
            "CGIE3_ID_04_v1.0",

        "configuration_status":
            "FROZEN",

        "generated_at_utc":
            utc_now_iso(),

        "technical_status":
            "COMPLETED",

        "scientific_outcome":
            outcome,

        "outcome_reason":
            outcome_details.get(
                "outcome_reason"
            ),

        "upstream_state": {
            "id02_total_relation_count":
                136,

            "id04_primary_relation_count":
                74,

            "id03_scientific_outcome":
                context.metadata[
                    "id03_scientific_outcome"
                ],

            "id03_reproducible_family_count":
                int(
                    context.metadata[
                        "id03_reproducible_family_count"
                    ]
                ),
        },

        "snapshot_stage":
            context.outputs[
                "snapshot_summary"
            ],

        "continuity_stage":
            context.outputs[
                "continuity_summary"
            ],

        "null_control_stage":
            context.outputs[
                "null_control_summary"
            ],

        "robustness_stage":
            context.outputs[
                "robustness_summary"
            ],

        "scientific_outcome_details":
            outcome_details,

        "estimability": {
            "scale_count":
                int(
                    len(
                        estimability
                    )
                ),

            "estimable_scale_count":
                int(
                    estimability[
                        "scale_estimable"
                    ]
                    .astype(
                        bool
                    )
                    .sum()
                ),
        },

        "scale_results":
            multiscale.to_dict(
                orient="records"
            ),

        "scientific_claims":
            build_scientific_claims(
                outcome
            ),

        "event_information_used":
            False,

        "claim_boundary": (
            "CGIE3-ID-04 tests whether relational organization "
            "shows temporal continuity beyond frozen null controls. "
            "A positive result does not establish indispensable "
            "relations, a Minimum Identity Core, causality, "
            "predictive capability, earthquake prediction or "
            "universal transferability."
        ),
    }


def build_scale_report(
    multiscale: pd.DataFrame,
) -> str:
    """Build human-readable scale-result section."""
    lines: list[
        str
    ] = []

    for row in multiscale.itertuples(
        index=False
    ):
        q_value = getattr(
            row,
            "fdr_q_value",
            None,
        )

        robustness = getattr(
            row,
            "robustness_fraction",
            None,
        )

        lines.append(
            f"""### `{row.scale_id}`

- All primary nulls estimable: `{bool(row.all_nulls_estimable)}`
- Conservative empirical p-value: `{row.conservative_empirical_p}`
- FDR q-value: `{q_value}`
- FDR significant: `{bool(row.fdr_significant)}`
- Positive effect against all nulls: `{bool(row.positive_effect_against_all_nulls)}`
- Robustness fraction: `{robustness}`
- Final scale support: `{bool(row.scale_support_final)}`
"""
        )

    return "\n".join(
        lines
    )


def build_report(
    summary: Mapping[str, Any],
    context: ID04ExperimentContext,
) -> str:
    """Build official Markdown scientific report."""
    multiscale = require_dataframe(
        context.outputs[
            "multiscale_continuity"
        ],
        "multiscale_continuity",
    )

    scale_text = build_scale_report(
        multiscale
    )

    return f"""# CGIE3-ID-04 — Relational Identity Continuity Audit

## Technical status

`{summary["technical_status"]}`

## Scientific outcome

`{summary["scientific_outcome"]}`

## Outcome reason

`{summary["outcome_reason"]}`

## Scientific question

Does relational organization preserve temporal continuity beyond
frozen null expectations even though CGIE3-ID-03 did not identify
reproducible static relational families?

## Frozen upstream boundary

- ID-02 relations preserved: `136`
- Primary ID-04 population: `74`
- ID-03 outcome: `{summary["upstream_state"]["id03_scientific_outcome"]}`
- Reproducible ID-03 families: `{summary["upstream_state"]["id03_reproducible_family_count"]}`

Neither ID-02 classifications nor ID-03 states were modified.

## Continuity definition

ID-04 evaluates four preregistered components:

- EC — Edge Continuity;
- WC — Weight Continuity;
- SC — Sign Continuity;
- TC — Topological Continuity.

WC and TC retain their raw Spearman values in [-1, 1].

For RCS composition only, WC and TC are mapped to [0, 1] by:

`(rho + 1) / 2`

RCS combines only estimable components using frozen weights.

Missing or non-estimable components are not converted to zero.

## Null controls

Three frozen null procedures were applied:

1. temporal snapshot permutation;
2. relation-label permutation;
3. constrained weight/sign surrogate.

Each uses the preregistered repetition count and deterministic seed.

## Multiple testing

Scale-level evidence is evaluated conservatively against all primary
nulls.

Benjamini-Hochberg FDR correction is applied across temporal scales
using the frozen threshold of `0.05`.

## Scale results

{scale_text}

## Robustness

The experiment applies the frozen leave-one-transition-out audit.

Full continuity support requires the preregistered robustness
threshold of at least `0.80`.

## Interpretation

`CONTINUITY_SUPPORTED` would mean that relational organization shows
cross-scale temporal continuity beyond the frozen null expectations.

`PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE` means some evidence exists but
does not satisfy the complete cross-scale preregistered rule.

`CONTINUITY_NOT_SUPPORTED` is a valid negative result when sufficient
estimability exists but no temporal scale satisfies the frozen support
rule.

`NON_IDENTIFIABLE` means the available evidence is insufficient to
identify the tested property.

## Scientific claim boundary

{summary["claim_boundary"]}

## Explicit non-claims

- Minimum Identity Core established: `false`
- Indispensable relations established: `false`
- Causality established: `false`
- Predictive capability established: `false`
- Earthquake prediction established: `false`
- Universal transferability established: `false`

## Event blindness

Earthquake-event information was not used for snapshot selection,
temporal-scale selection, threshold optimization, weight optimization,
null construction or scientific classification.

## Generated

`{summary["generated_at_utc"]}`
"""


def build_manifest(
    context: ID04ExperimentContext,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build official ID-04 manifest."""
    return {
        "experiment_id":
            "CGIE3_ID_04",

        "protocol_version":
            "CGIE3_ID_04_v1.0",

        "generated_at_utc":
            summary[
                "generated_at_utc"
            ],

        "configuration_status":
            "FROZEN",

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

        "output_files": {
            output_id:
                str(
                    path.relative_to(
                        REPOSITORY_ROOT
                    )
                )
            for output_id, path
            in OUTPUT_PATHS.items()
        },
    }


def generate_reports(
    context: ID04ExperimentContext,
) -> dict[str, Any]:
    """Generate official CGIE3-ID-04 scientific outputs."""
    validate_context(
        context
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_relations = require_dataframe(
        context.outputs[
            "snapshot_relations"
        ],
        "snapshot_relations",
    )

    transition_continuity = require_dataframe(
        context.outputs[
            "transition_continuity"
        ],
        "transition_continuity",
    )

    multiscale = require_dataframe(
        context.outputs[
            "multiscale_continuity"
        ],
        "multiscale_continuity",
    )

    null_controls = require_dataframe(
        context.outputs[
            "null_controls"
        ],
        "null_controls",
    )

    null_summaries = require_dataframe(
        context.outputs[
            "null_summaries"
        ],
        "null_summaries",
    )

    leave_one = require_dataframe(
        context.outputs[
            "leave_one_transition_out"
        ],
        "leave_one_transition_out",
        allow_empty=True,
    )

    estimability = build_estimability_table(
        context
    )

    summary = build_summary(
        context,
        estimability,
    )

    report_text = build_report(
        summary,
        context,
    )

    manifest = build_manifest(
        context,
        summary,
    )

    dataframe_outputs = {
        "snapshot_relations":
            snapshot_relations,

        "transition_continuity":
            transition_continuity,

        "multiscale_continuity":
            multiscale,

        "null_controls":
            null_controls,

        "null_summaries":
            null_summaries,

        "leave_one_transition_out":
            leave_one,

        "estimability":
            estimability,
    }

    for output_id, frame in dataframe_outputs.items():
        write_csv(
            OUTPUT_PATHS[
                output_id
            ],
            frame,
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

    hashes = {
        output_id:
            sha256_file(
                path
            )
        for output_id, path
        in OUTPUT_PATHS.items()
    }

    manifest_with_hashes = dict(
        manifest
    )

    manifest_with_hashes[
        "output_sha256"
    ] = hashes

    write_json(
        OUTPUT_PATHS[
            "manifest"
        ],
        manifest_with_hashes,
    )

    hashes[
        "manifest"
    ] = sha256_file(
        OUTPUT_PATHS[
            "manifest"
        ]
    )

    result = {
        "experiment_id":
            "CGIE3_ID_04",

        "technical_status":
            "COMPLETED",

        "scientific_outcome":
            summary[
                "scientific_outcome"
            ],

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
            in hashes.items()
        },

        "statistics": {
            "snapshot_relation_rows":
                int(
                    len(
                        snapshot_relations
                    )
                ),

            "transition_count":
                int(
                    len(
                        transition_continuity
                    )
                ),

            "scale_count":
                int(
                    len(
                        multiscale
                    )
                ),

            "null_replication_count":
                int(
                    len(
                        null_controls
                    )
                ),

            "scientific_outcome":
                summary[
                    "scientific_outcome"
                ],
        },

        "warnings": [
            (
                "Relational continuity does not establish "
                "a Minimum Identity Core or indispensability."
            ),

            (
                "CGIE3-ID-04 does not establish causality, "
                "predictive capability or earthquake prediction."
            ),
        ],
    }

    context.register_output(
        "estimability",
        estimability,
    )

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
        hashes,
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
