"""
CGIE3-ID-03 multiscale relation audit.

This stage aligns the same unordered component pair across the frozen
temporal scales:

- 1d;
- 3d;
- 7d;
- 30d.

It evaluates:

- ID-02 support across scales;
- sign consistency;
- strength variation;
- persistence variation;
- bootstrap interval information;
- multiscale support class.

It does not:

- modify ID-02 classifications;
- use evaluation-period data;
- infer causality;
- identify primary or indispensable relations;
- select family representatives.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from engines.cgie3.src.id03.loader import (
    ID03ExperimentContext,
)


class MultiscaleAuditError(ValueError):
    """Raised when the multiscale audit violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a normalized multiscale-audit error."""
    raise MultiscaleAuditError(
        str(message).strip()
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


def require_string_list(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    """Require a non-empty ordered list of unique strings."""
    if not isinstance(
        value,
        (list, tuple),
    ):
        fail(
            f"{field_name} must be a list."
        )

    normalized = tuple(
        str(item).strip()
        for item in value
    )

    if not normalized:
        fail(
            f"{field_name} must not be empty."
        )

    if any(
        not item
        for item in normalized
    ):
        fail(
            f"{field_name} contains empty values."
        )

    if len(normalized) != len(
        set(normalized)
    ):
        fail(
            f"{field_name} contains duplicates."
        )

    return normalized


def require_non_negative_float(
    value: Any,
    field_name: str,
) -> float:
    """Require a finite non-negative number."""
    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise MultiscaleAuditError(
            f"{field_name} must be numeric."
        ) from exc

    if (
        not np.isfinite(normalized)
        or normalized < 0.0
    ):
        fail(
            f"{field_name} must be finite and non-negative."
        )

    return normalized


def validate_context(
    context: ID03ExperimentContext,
) -> None:
    """Validate multiscale-audit prerequisites."""
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

    if (
        context.runtime.get(
            "dependency_audit_status"
        )
        != "COMPLETED"
    ):
        fail(
            "Dependency audit must complete before "
            "the multiscale audit."
        )

    required_outputs = {
        "relation_dependencies",
        "primary_population",
    }

    missing_outputs = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing_outputs:
        fail(
            "Multiscale prerequisites are missing: "
            + ", ".join(missing_outputs)
        )

    if len(
        context.relation_classification
    ) != 136:
        fail(
            "Multiscale audit expects 136 ID-02 relations."
        )


def get_multiscale_contract(
    context: ID03ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate the frozen multiscale contract."""
    audit = require_mapping(
        context.configuration.get(
            "multiscale_audit"
        ),
        "multiscale_audit",
    )

    if audit.get("enabled") is not True:
        fail(
            "Multiscale audit must remain enabled."
        )

    pair_alignment = require_mapping(
        audit.get(
            "pair_alignment"
        ),
        "multiscale_audit.pair_alignment",
    )

    if (
        pair_alignment.get(
            "relation_type"
        )
        != "undirected"
    ):
        fail(
            "Multiscale pair alignment must be undirected."
        )

    if (
        pair_alignment.get(
            "canonical_order"
        )
        != "lexicographic"
    ):
        fail(
            "Multiscale pair alignment must use "
            "lexicographic canonical order."
        )

    if (
        pair_alignment.get(
            "align_by_component_pair"
        )
        is not True
    ):
        fail(
            "Relations must be aligned by component pair."
        )

    supported_statuses = require_string_list(
        audit.get(
            "supported_id02_statuses"
        ),
        "multiscale_audit.supported_id02_statuses",
    )

    if set(
        supported_statuses
    ) != {
        "eligible",
        "candidate",
    }:
        fail(
            "Supported ID-02 statuses must be "
            "eligible and candidate."
        )

    minimum_supported_scales = require_mapping(
        audit.get(
            "minimum_supported_scales"
        ),
        "multiscale_audit.minimum_supported_scales",
    )

    representative_minimum = int(
        minimum_supported_scales[
            "family_representative_candidate"
        ]
    )

    if representative_minimum != 2:
        fail(
            "ID-03 requires at least two supported "
            "scales for representative candidacy."
        )

    sign_consistency = require_mapping(
        audit.get(
            "sign_consistency"
        ),
        "multiscale_audit.sign_consistency",
    )

    if (
        sign_consistency.get(
            "require_same_sign_across_supported_scales"
        )
        is not True
    ):
        fail(
            "Supported scales must preserve the same sign."
        )

    if (
        sign_consistency.get(
            "zero_sign_is_neutral"
        )
        is not False
    ):
        fail(
            "Zero sign must not be treated as neutral."
        )

    strong_scale = require_mapping(
        audit.get(
            "strong_scale_definition"
        ),
        "multiscale_audit.strong_scale_definition",
    )

    required_windows = require_string_list(
        require_mapping(
            context.configuration[
                "feature_table"
            ],
            "feature_table",
        )[
            "required_windows"
        ],
        "feature_table.required_windows",
    )

    expected_windows = (
        "1d",
        "3d",
        "7d",
        "30d",
    )

    if required_windows != expected_windows:
        fail(
            "ID-03 requires the frozen window order: "
            "1d, 3d, 7d, 30d."
        )

    return {
        "required_windows":
            required_windows,

        "supported_statuses":
            set(
                supported_statuses
            ),

        "representative_minimum_scales":
            representative_minimum,

        "strong_minimum_strength":
            require_non_negative_float(
                strong_scale[
                    "minimum_absolute_strength"
                ],
                (
                    "multiscale_audit."
                    "strong_scale_definition."
                    "minimum_absolute_strength"
                ),
            ),

        "strong_minimum_persistence":
            require_non_negative_float(
                strong_scale[
                    "minimum_persistence"
                ],
                (
                    "multiscale_audit."
                    "strong_scale_definition."
                    "minimum_persistence"
                ),
            ),
    }


def canonical_pair(
    source_id: str,
    target_id: str,
) -> tuple[str, str]:
    """Return one unordered component pair in lexicographic order."""
    source = str(
        source_id
    ).strip()

    target = str(
        target_id
    ).strip()

    if not source or not target:
        fail(
            "Relation endpoints must not be empty."
        )

    if source == target:
        fail(
            "Self-relations are not permitted."
        )

    return tuple(
        sorted(
            (
                source,
                target,
            )
        )
    )


def canonical_pair_id(
    source_id: str,
    target_id: str,
) -> str:
    """Build a deterministic multiscale component-pair identifier."""
    left, right = canonical_pair(
        source_id,
        target_id,
    )

    return (
        f"{left}--{right}"
    )


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    field_name: str,
) -> None:
    """Require declared columns in one DataFrame."""
    missing_columns = sorted(
        columns
        - set(
            frame.columns
        )
    )

    if missing_columns:
        fail(
            f"{field_name} is missing columns: "
            + ", ".join(
                missing_columns
            )
        )


def prepare_relation_evidence(
    context: ID03ExperimentContext,
) -> pd.DataFrame:
    """
    Create one validated relation-evidence table.

    The ID-02 classification table already contains persistence,
    robustness and bootstrap-summary columns produced by the frozen
    classification merge.
    """
    classification = (
        context.relation_classification.copy()
    )

    required_columns = {
        "window_id",
        "relation_id",
        "source_id",
        "target_id",
        "classification_status",
        "estimability",
        "strength",
        "absolute_strength",
        "sign",
        "persistence",
        "estimable_block_fraction",
        "sign_preservation_fraction",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_ci_width",
        "leave_one_block_out_robust",
        "missingness_stress_robust",
    }

    require_columns(
        classification,
        required_columns,
        "relation_classification",
    )

    dependencies = context.outputs[
        "relation_dependencies"
    ]

    if not isinstance(
        dependencies,
        pd.DataFrame,
    ):
        fail(
            "relation_dependencies must be a DataFrame."
        )

    require_columns(
        dependencies,
        {
            "window_id",
            "relation_id",
            "dependency_status",
            "definitionally_constrained_flag",
        },
        "relation_dependencies",
    )

    dependency_subset = dependencies[
        [
            "window_id",
            "relation_id",
            "dependency_status",
            "definitionally_constrained_flag",
        ]
    ].copy()

    evidence = classification.merge(
        dependency_subset,
        on=[
            "window_id",
            "relation_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(evidence) != 136:
        fail(
            "Merged multiscale evidence must contain "
            f"136 rows; observed {len(evidence)}."
        )

    evidence[
        "pair_id"
    ] = [
        canonical_pair_id(
            source_id,
            target_id,
        )
        for source_id, target_id
        in zip(
            evidence[
                "source_id"
            ],
            evidence[
                "target_id"
            ],
        )
    ]

    evidence[
        "canonical_source_id"
    ] = [
        canonical_pair(
            source_id,
            target_id,
        )[0]
        for source_id, target_id
        in zip(
            evidence[
                "source_id"
            ],
            evidence[
                "target_id"
            ],
        )
    ]

    evidence[
        "canonical_target_id"
    ] = [
        canonical_pair(
            source_id,
            target_id,
        )[1]
        for source_id, target_id
        in zip(
            evidence[
                "source_id"
            ],
            evidence[
                "target_id"
            ],
        )
    ]

    duplicate_mask = evidence.duplicated(
        subset=[
            "window_id",
            "pair_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        fail(
            "Multiple relations exist for the same "
            "window-component pair."
        )

    return evidence


def safe_float(
    value: Any,
) -> float | None:
    """Convert one finite value to float or return None."""
    if value is None:
        return None

    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(
        normalized
    ):
        return None

    return normalized


def safe_int_sign(
    value: Any,
) -> int | None:
    """Normalize one sign to -1, 0 or 1."""
    normalized = safe_float(
        value
    )

    if normalized is None:
        return None

    sign = int(
        normalized
    )

    if sign not in {
        -1,
        0,
        1,
    }:
        return None

    return sign


def status_is_supported(
    status: Any,
    supported_statuses: set[str],
) -> bool:
    """Return whether an ID-02 status supports multiscale evidence."""
    return str(
        status
    ).strip() in supported_statuses


def classify_multiscale_support(
    supported_scale_count: int,
    *,
    sign_consistent: bool,
    scale_inconsistent: bool,
) -> str:
    """Assign one frozen multiscale-support class."""
    if scale_inconsistent:
        return "scale_inconsistent"

    if not sign_consistent:
        return "scale_inconsistent"

    if supported_scale_count >= 4:
        return "multi_scale_4"

    if supported_scale_count == 3:
        return "multi_scale_3"

    if supported_scale_count == 2:
        return "multi_scale_2"

    if supported_scale_count == 1:
        return "single_scale"

    return "insufficient_scale_support"


def build_pair_record(
    pair_id: str,
    pair_frame: pd.DataFrame,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one aligned multiscale relation record."""
    required_windows = tuple(
        contract[
            "required_windows"
        ]
    )

    supported_statuses = set(
        contract[
            "supported_statuses"
        ]
    )

    pair_frame = pair_frame.copy()

    source_values = set(
        pair_frame[
            "canonical_source_id"
        ].astype(str)
    )

    target_values = set(
        pair_frame[
            "canonical_target_id"
        ].astype(str)
    )

    if (
        len(source_values) != 1
        or len(target_values) != 1
    ):
        fail(
            f"Pair {pair_id} contains inconsistent endpoints."
        )

    source_id = next(
        iter(
            source_values
        )
    )

    target_id = next(
        iter(
            target_values
        )
    )

    row_by_window = {
        str(row.window_id): row
        for row in pair_frame.itertuples(
            index=False
        )
    }

    supported_windows: list[str] = []
    estimable_windows: list[str] = []
    supported_signs: list[int] = []
    strong_supported_signs: list[int] = []
    supported_strengths: list[float] = []
    supported_persistence: list[float] = []
    supported_ci_widths: list[float] = []

    record: dict[str, Any] = {
        "experiment_id":
            "CGIE3_ID_03",

        "pair_id":
            pair_id,

        "source_id":
            source_id,

        "target_id":
            target_id,
    }

    dependency_statuses: set[str] = set()
    definitionally_constrained = False

    for window_id in required_windows:
        row = row_by_window.get(
            window_id
        )

        prefix = (
            f"scale_{window_id}_"
        )

        if row is None:
            record[
                f"{prefix}present"
            ] = False

            record[
                f"{prefix}id02_status"
            ] = None

            record[
                f"{prefix}estimability"
            ] = None

            record[
                f"{prefix}strength"
            ] = None

            record[
                f"{prefix}absolute_strength"
            ] = None

            record[
                f"{prefix}sign"
            ] = None

            record[
                f"{prefix}persistence"
            ] = None

            record[
                f"{prefix}bootstrap_ci_lower"
            ] = None

            record[
                f"{prefix}bootstrap_ci_upper"
            ] = None

            record[
                f"{prefix}bootstrap_ci_width"
            ] = None

            record[
                f"{prefix}supported"
            ] = False

            continue

        status = str(
            row.classification_status
        )

        estimability = str(
            row.estimability
        )

        strength = safe_float(
            row.strength
        )

        absolute_strength = safe_float(
            row.absolute_strength
        )

        sign = safe_int_sign(
            row.sign
        )

        persistence = safe_float(
            row.persistence
        )

        ci_lower = safe_float(
            row.bootstrap_ci_lower
        )

        ci_upper = safe_float(
            row.bootstrap_ci_upper
        )

        ci_width = safe_float(
            row.bootstrap_ci_width
        )

        supported = bool(
            status_is_supported(
                status,
                supported_statuses,
            )
            and estimability
            == "estimable"
            and strength
            is not None
            and sign
            is not None
        )

        record[
            f"{prefix}present"
        ] = True

        record[
            f"{prefix}relation_id"
        ] = str(
            row.relation_id
        )

        record[
            f"{prefix}id02_status"
        ] = status

        record[
            f"{prefix}estimability"
        ] = estimability

        record[
            f"{prefix}strength"
        ] = strength

        record[
            f"{prefix}absolute_strength"
        ] = absolute_strength

        record[
            f"{prefix}sign"
        ] = sign

        record[
            f"{prefix}persistence"
        ] = persistence

        record[
            f"{prefix}estimable_block_fraction"
        ] = safe_float(
            row.estimable_block_fraction
        )

        record[
            f"{prefix}sign_preservation_fraction"
        ] = safe_float(
            row.sign_preservation_fraction
        )

        record[
            f"{prefix}bootstrap_ci_lower"
        ] = ci_lower

        record[
            f"{prefix}bootstrap_ci_upper"
        ] = ci_upper

        record[
            f"{prefix}bootstrap_ci_width"
        ] = ci_width

        record[
            f"{prefix}leave_one_block_out_robust"
        ] = bool(
            row.leave_one_block_out_robust
        )

        record[
            f"{prefix}missingness_stress_robust"
        ] = bool(
            row.missingness_stress_robust
        )

        record[
            f"{prefix}supported"
        ] = supported

        dependency_status = str(
            row.dependency_status
        )

        dependency_statuses.add(
            dependency_status
        )

        definitionally_constrained = bool(
            definitionally_constrained
            or bool(
                row.definitionally_constrained_flag
            )
        )

        if estimability == "estimable":
            estimable_windows.append(
                window_id
            )

        if supported:
            supported_windows.append(
                window_id
            )

            supported_signs.append(
                int(sign)
            )

            if strength is not None:
                supported_strengths.append(
                    float(
                        strength
                    )
                )

            if persistence is not None:
                supported_persistence.append(
                    float(
                        persistence
                    )
                )

            if ci_width is not None:
                supported_ci_widths.append(
                    float(
                        ci_width
                    )
                )

            is_strong = bool(
                absolute_strength
                is not None
                and persistence
                is not None
                and absolute_strength
                >= contract[
                    "strong_minimum_strength"
                ]
                and persistence
                >= contract[
                    "strong_minimum_persistence"
                ]
            )

            if is_strong:
                strong_supported_signs.append(
                    int(sign)
                )

    supported_scale_count = int(
        len(
            supported_windows
        )
    )

    estimable_scale_count = int(
        len(
            estimable_windows
        )
    )

    unique_supported_signs = set(
        supported_signs
    )

    sign_consistent = bool(
        supported_scale_count > 0
        and len(
            unique_supported_signs
        )
        == 1
        and 0 not in unique_supported_signs
    )

    opposite_sign_detected = bool(
        1 in unique_supported_signs
        and -1 in unique_supported_signs
    )

    dominant_sign: int | None

    if not supported_signs:
        dominant_sign = None
    else:
        positive_count = supported_signs.count(
            1
        )

        negative_count = supported_signs.count(
            -1
        )

        zero_count = supported_signs.count(
            0
        )

        counts = {
            1: positive_count,
            -1: negative_count,
            0: zero_count,
        }

        maximum_count = max(
            counts.values()
        )

        dominant_candidates = [
            sign_value
            for sign_value, count
            in counts.items()
            if count == maximum_count
        ]

        if len(
            dominant_candidates
        ) == 1:
            dominant_sign = (
                dominant_candidates[0]
            )
        else:
            dominant_sign = None

    strong_scale_contradiction = bool(
        dominant_sign is not None
        and any(
            sign_value
            != dominant_sign
            for sign_value
            in strong_supported_signs
        )
    )

    scale_inconsistent = bool(
        opposite_sign_detected
        or strong_scale_contradiction
        or (
            supported_scale_count > 0
            and not sign_consistent
        )
    )

    multiscale_class = (
        classify_multiscale_support(
            supported_scale_count,
            sign_consistent=(
                sign_consistent
            ),
            scale_inconsistent=(
                scale_inconsistent
            ),
        )
    )

    if supported_strengths:
        minimum_strength = float(
            min(
                supported_strengths
            )
        )

        maximum_strength = float(
            max(
                supported_strengths
            )
        )

        strength_range = float(
            maximum_strength
            - minimum_strength
        )

        mean_strength = float(
            np.mean(
                supported_strengths
            )
        )

        standard_deviation_strength = (
            float(
                np.std(
                    supported_strengths,
                    ddof=1,
                )
            )
            if len(
                supported_strengths
            )
            > 1
            else 0.0
        )
    else:
        minimum_strength = None
        maximum_strength = None
        strength_range = None
        mean_strength = None
        standard_deviation_strength = None

    if supported_persistence:
        minimum_persistence = float(
            min(
                supported_persistence
            )
        )

        maximum_persistence = float(
            max(
                supported_persistence
            )
        )

        mean_persistence = float(
            np.mean(
                supported_persistence
            )
        )
    else:
        minimum_persistence = None
        maximum_persistence = None
        mean_persistence = None

    if supported_ci_widths:
        median_ci_width = float(
            np.median(
                supported_ci_widths
            )
        )
    else:
        median_ci_width = None

    record.update(
        {
            "present_scale_count":
                int(
                    len(
                        pair_frame
                    )
                ),

            "estimable_scale_count":
                estimable_scale_count,

            "supported_scale_count":
                supported_scale_count,

            "supported_windows":
                "|".join(
                    supported_windows
                ),

            "estimable_windows":
                "|".join(
                    estimable_windows
                ),

            "dominant_sign":
                dominant_sign,

            "sign_consistent":
                sign_consistent,

            "opposite_sign_detected":
                opposite_sign_detected,

            "strong_scale_contradiction":
                strong_scale_contradiction,

            "scale_inconsistent":
                scale_inconsistent,

            "multiscale_class":
                multiscale_class,

            "minimum_supported_strength":
                minimum_strength,

            "maximum_supported_strength":
                maximum_strength,

            "supported_strength_range":
                strength_range,

            "mean_supported_strength":
                mean_strength,

            "supported_strength_standard_deviation":
                standard_deviation_strength,

            "minimum_supported_persistence":
                minimum_persistence,

            "maximum_supported_persistence":
                maximum_persistence,

            "mean_supported_persistence":
                mean_persistence,

            "median_supported_bootstrap_ci_width":
                median_ci_width,

            "representative_minimum_scale_count_met":
                bool(
                    supported_scale_count
                    >= contract[
                        "representative_minimum_scales"
                    ]
                ),

            "dependency_statuses":
                "|".join(
                    sorted(
                        dependency_statuses
                    )
                ),

            "definitionally_constrained_flag":
                definitionally_constrained,

            "id02_status_modified":
                False,
        }
    )

    return record


def build_multiscale_table(
    context: ID03ExperimentContext,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Build one record for every unique unordered component pair."""
    evidence = prepare_relation_evidence(
        context
    )

    records: list[dict[str, Any]] = []

    for pair_id, pair_frame in evidence.groupby(
        "pair_id",
        sort=True,
    ):
        records.append(
            build_pair_record(
                str(
                    pair_id
                ),
                pair_frame,
                contract,
            )
        )

    output = pd.DataFrame.from_records(
        records
    )

    if output.empty:
        fail(
            "Multiscale audit produced no component pairs."
        )

    duplicate_mask = output.duplicated(
        subset=[
            "pair_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        fail(
            "Multiscale output contains duplicate pair IDs."
        )

    maximum_possible_pairs = (
        len(
            context.identity.component_ids
        )
        * (
            len(
                context.identity.component_ids
            )
            - 1
        )
        // 2
    )

    if len(output) > maximum_possible_pairs:
        fail(
            "Multiscale pair count exceeds the "
            "declared component-pair maximum."
        )

    return output.sort_values(
        by=[
            "source_id",
            "target_id",
            "pair_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_long_multiscale_table(
    context: ID03ExperimentContext,
) -> pd.DataFrame:
    """
    Preserve one row per original ID-02 window-relation observation.

    This table keeps all 136 frozen ID-02 statuses beside the canonical
    multiscale pair identifier.
    """
    evidence = prepare_relation_evidence(
        context
    )

    selected_columns = [
        "window_id",
        "relation_id",
        "pair_id",
        "canonical_source_id",
        "canonical_target_id",
        "classification_status",
        "estimability",
        "strength",
        "absolute_strength",
        "sign",
        "persistence",
        "estimable_block_fraction",
        "sign_preservation_fraction",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_ci_width",
        "leave_one_block_out_robust",
        "missingness_stress_robust",
        "dependency_status",
        "definitionally_constrained_flag",
    ]

    long_table = evidence.loc[
        :,
        selected_columns,
    ].copy()

    long_table = long_table.rename(
        columns={
            "canonical_source_id":
                "source_id",

            "canonical_target_id":
                "target_id",

            "classification_status":
                "id02_status",
        }
    )

    long_table.insert(
        0,
        "experiment_id",
        "CGIE3_ID_03",
    )

    long_table[
        "id02_status_modified"
    ] = False

    if len(long_table) != 136:
        fail(
            "Long multiscale table must preserve "
            f"136 ID-02 relations; observed "
            f"{len(long_table)}."
        )

    return long_table.sort_values(
        by=[
            "source_id",
            "target_id",
            "window_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_multiscale_summary(
    pair_table: pd.DataFrame,
    long_table: pd.DataFrame,
) -> dict[str, Any]:
    """Build descriptive multiscale-audit counts."""
    class_counts = (
        pair_table[
            "multiscale_class"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    support_counts = (
        pair_table[
            "supported_scale_count"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    primary_long = long_table.loc[
        long_table[
            "id02_status"
        ].isin(
            [
                "eligible",
                "candidate",
            ]
        )
    ]

    return {
        "status":
            "COMPLETED",

        "unique_pair_count":
            int(
                len(
                    pair_table
                )
            ),

        "preserved_id02_relation_count":
            int(
                len(
                    long_table
                )
            ),

        "primary_id02_relation_count":
            int(
                len(
                    primary_long
                )
            ),

        "multiscale_class_counts": {
            str(key):
                int(value)
            for key, value
            in class_counts.items()
        },

        "supported_scale_count_distribution": {
            str(key):
                int(value)
            for key, value
            in support_counts.items()
        },

        "sign_consistent_pair_count":
            int(
                pair_table[
                    "sign_consistent"
                ].sum()
            ),

        "scale_inconsistent_pair_count":
            int(
                pair_table[
                    "scale_inconsistent"
                ].sum()
            ),

        "minimum_two_scale_support_count":
            int(
                pair_table[
                    "representative_minimum_scale_count_met"
                ].sum()
            ),

        "definitionally_constrained_pair_count":
            int(
                pair_table[
                    "definitionally_constrained_flag"
                ].sum()
            ),

        "id02_statuses_modified":
            False,
    }


def audit_multiscale(
    context: ID03ExperimentContext,
) -> ID03ExperimentContext:
    """Execute the frozen CGIE3-ID-03 multiscale audit."""
    validate_context(
        context
    )

    contract = get_multiscale_contract(
        context
    )

    pair_table = build_multiscale_table(
        context,
        contract,
    )

    long_table = (
        build_long_multiscale_table(
            context
        )
    )

    summary = build_multiscale_summary(
        pair_table,
        long_table,
    )

    context.register_output(
        "multiscale_relations",
        pair_table,
    )

    context.register_output(
        "multiscale_relations_long",
        long_table,
    )

    context.register_output(
        "multiscale_audit_summary",
        summary,
    )

    context.register_runtime(
        "multiscale_audit_status",
        "COMPLETED",
    )

    return context
  
  
