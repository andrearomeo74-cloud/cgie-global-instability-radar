"""
CGIE3-ID-02 missingness-stress stage.

This module evaluates relation robustness under frozen simulated
pairwise missingness.

It does not:

- alter the original ExperimentContext feature tables;
- read files directly;
- classify relations;
- use evaluation-period or target-event information;
- infer causality or prediction.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from congruity.core import ExperimentContext


class MissingnessStageError(ValueError):
    """Raised when missingness stress violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a missingness-stage error."""
    raise MissingnessStageError(
        str(message).strip()
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like value."""
    if not isinstance(value, Mapping):
        fail(
            f"{field_name} must be a mapping."
        )

    return value


def require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """Require an integer greater than zero."""
    if isinstance(value, bool):
        fail(
            f"{field_name} must be an integer."
        )

    try:
        normalized = int(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise MissingnessStageError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
        )

    return normalized


def require_fraction(
    value: Any,
    field_name: str,
) -> float:
    """Require a numeric fraction inside [0, 1]."""
    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise MissingnessStageError(
            f"{field_name} must be numeric."
        ) from exc

    if not 0.0 <= normalized <= 1.0:
        fail(
            f"{field_name} must lie inside [0, 1]."
        )

    return normalized


def validate_context(
    context: ExperimentContext,
) -> None:
    """Validate missingness-stage prerequisites."""
    if not isinstance(
        context,
        ExperimentContext,
    ):
        fail(
            "context must be an ExperimentContext."
        )

    if context.experiment_id != "CGIE3_ID_02":
        fail(
            "Unexpected experiment ID: "
            f"{context.experiment_id}"
        )

    required_outputs = {
        "baseline_feature_tables",
        "candidate_relations",
        "candidate_relations_by_window",
        "relation_persistence",
        "leave_one_block_out",
        "bootstrap_summary",
    }

    missing_outputs = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing_outputs:
        fail(
            "Missingness prerequisites are absent: "
            + ", ".join(missing_outputs)
        )


def get_missingness_contract(
    context: ExperimentContext,
) -> dict[str, Any]:
    """Extract the frozen missingness-stress configuration."""
    robustness = require_mapping(
        context.configuration.get(
            "robustness"
        ),
        "robustness",
    )

    missingness = require_mapping(
        robustness.get(
            "missingness_stress"
        ),
        "robustness.missingness_stress",
    )

    estimability = require_mapping(
        context.configuration.get(
            "estimability"
        ),
        "estimability",
    )

    if missingness.get("enabled") is not True:
        fail(
            "Missingness stress must remain enabled."
        )

    simulated_missing_fraction = require_fraction(
        missingness.get(
            "simulated_missing_fraction"
        ),
        "robustness.missingness_stress."
        "simulated_missing_fraction",
    )

    if simulated_missing_fraction != 0.10:
        fail(
            "CGIE3-ID-02 requires simulated missingness "
            "fraction 0.10."
        )

    repetitions = require_positive_integer(
        missingness.get("repetitions"),
        "robustness.missingness_stress.repetitions",
    )

    if repetitions != 100:
        fail(
            "CGIE3-ID-02 requires exactly "
            "100 missingness repetitions."
        )

    random_seed = require_positive_integer(
        missingness.get("random_seed"),
        "robustness.missingness_stress.random_seed",
    )

    minimum_sign_fraction = require_fraction(
        missingness.get(
            "minimum_sign_preservation_fraction"
        ),
        "robustness.missingness_stress."
        "minimum_sign_preservation_fraction",
    )

    minimum_samples = require_positive_integer(
        estimability.get(
            "minimum_paired_observations"
        ),
        "estimability.minimum_paired_observations",
    )

    minimum_unique_values = require_positive_integer(
        estimability.get(
            "minimum_unique_values_per_component"
        ),
        "estimability."
        "minimum_unique_values_per_component",
    )

    return {
        "simulated_missing_fraction":
            simulated_missing_fraction,
        "repetitions":
            repetitions,
        "random_seed":
            random_seed,
        "minimum_sign_fraction":
            minimum_sign_fraction,
        "minimum_samples":
            minimum_samples,
        "minimum_unique_values":
            minimum_unique_values,
    }


def deterministic_relation_seed(
    base_seed: int,
    window_id: str,
    relation_id: str,
) -> int:
    """Derive a stable seed for one relation."""
    token = (
        f"missingness::{base_seed}::"
        f"{window_id}::{relation_id}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        token
    ).digest()

    offset = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return int(
        (
            base_seed
            + offset
        )
        % (
            2**32 - 1
        )
    )


def prepare_pair_data(
    frame: pd.DataFrame,
    source_id: str,
    target_id: str,
) -> pd.DataFrame:
    """Prepare finite paired observations without mutating the input."""
    required = {
        source_id,
        target_id,
    }

    missing_columns = sorted(
        required
        - set(frame.columns)
    )

    if missing_columns:
        fail(
            "Missingness frame lacks components: "
            + ", ".join(missing_columns)
        )

    pair = frame.loc[
        :,
        [
            source_id,
            target_id,
        ],
    ].copy()

    pair[source_id] = pd.to_numeric(
        pair[source_id],
        errors="coerce",
    )

    pair[target_id] = pd.to_numeric(
        pair[target_id],
        errors="coerce",
    )

    pair = pair.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return pair.dropna(
        subset=[
            source_id,
            target_id,
        ]
    ).reset_index(
        drop=True
    )


def estimate_stressed_strength(
    source_values: np.ndarray,
    target_values: np.ndarray,
    retained_indices: np.ndarray,
    *,
    minimum_samples: int,
    minimum_unique_values: int,
) -> tuple[float | None, str]:
    """Estimate one Spearman relation after simulated deletion."""
    if retained_indices.size < minimum_samples:
        return (
            None,
            "insufficient_observations_after_stress",
        )

    source_sample = source_values[
        retained_indices
    ]

    target_sample = target_values[
        retained_indices
    ]

    if (
        np.unique(
            source_sample
        ).size
        < minimum_unique_values
    ):
        return (
            None,
            "insufficient_unique_source_values",
        )

    if (
        np.unique(
            target_sample
        ).size
        < minimum_unique_values
    ):
        return (
            None,
            "insufficient_unique_target_values",
        )

    result = spearmanr(
        source_sample,
        target_sample,
        nan_policy="omit",
    )

    strength = float(
        result.statistic
    )

    if not math.isfinite(
        strength
    ):
        return (
            None,
            "non_finite_estimate",
        )

    return strength, "estimable"


def stress_relation(
    frame: pd.DataFrame,
    full_relation: Any,
    *,
    window_id: str,
    missing_fraction: float,
    repetitions: int,
    base_seed: int,
    minimum_samples: int,
    minimum_unique_values: int,
) -> pd.DataFrame:
    """Run the frozen missingness stress for one relation."""
    relation_id = str(
        full_relation.relation_id
    )

    source_id = str(
        full_relation.source_id
    )

    target_id = str(
        full_relation.target_id
    )

    pair = prepare_pair_data(
        frame,
        source_id,
        target_id,
    )

    observation_count = int(
        len(pair)
    )

    relation_seed = deterministic_relation_seed(
        base_seed,
        window_id,
        relation_id,
    )

    records: list[dict[str, Any]] = []

    if (
        full_relation.estimability
        != "estimable"
    ):
        for repetition in range(
            1,
            repetitions + 1,
        ):
            records.append(
                {
                    "experiment_id":
                        "CGIE3_ID_02",
                    "window_id":
                        window_id,
                    "relation_id":
                        relation_id,
                    "source_id":
                        source_id,
                    "target_id":
                        target_id,
                    "repetition":
                        repetition,
                    "relation_seed":
                        relation_seed,
                    "original_observation_count":
                        observation_count,
                    "removed_observation_count":
                        0,
                    "retained_observation_count":
                        observation_count,
                    "stressed_strength":
                        np.nan,
                    "stressed_sign":
                        np.nan,
                    "estimability":
                        "non_estimable",
                    "non_estimable_reason":
                        "full_baseline_non_estimable",
                }
            )

        return pd.DataFrame.from_records(
            records
        )

    source_values = pair[
        source_id
    ].to_numpy(
        dtype=float
    )

    target_values = pair[
        target_id
    ].to_numpy(
        dtype=float
    )

    removal_count = int(
        math.floor(
            observation_count
            * missing_fraction
        )
    )

    if (
        observation_count > minimum_samples
        and removal_count == 0
    ):
        removal_count = 1

    removal_count = min(
        removal_count,
        max(
            0,
            observation_count
            - minimum_samples,
        ),
    )

    rng = np.random.default_rng(
        relation_seed
    )

    all_indices = np.arange(
        observation_count,
        dtype=int,
    )

    for repetition in range(
        1,
        repetitions + 1,
    ):
        if removal_count > 0:
            removed_indices = rng.choice(
                all_indices,
                size=removal_count,
                replace=False,
            )

            retained_mask = np.ones(
                observation_count,
                dtype=bool,
            )

            retained_mask[
                removed_indices
            ] = False

            retained_indices = all_indices[
                retained_mask
            ]
        else:
            retained_indices = all_indices.copy()

        strength, reason = (
            estimate_stressed_strength(
                source_values,
                target_values,
                retained_indices,
                minimum_samples=minimum_samples,
                minimum_unique_values=(
                    minimum_unique_values
                ),
            )
        )

        if strength is None:
            sign: int | float = np.nan
            estimability = "non_estimable"
        else:
            sign = (
                1
                if strength > 0.0
                else (
                    -1
                    if strength < 0.0
                    else 0
                )
            )

            estimability = "estimable"

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_02",
                "window_id":
                    window_id,
                "relation_id":
                    relation_id,
                "source_id":
                    source_id,
                "target_id":
                    target_id,
                "repetition":
                    repetition,
                "relation_seed":
                    relation_seed,
                "original_observation_count":
                    observation_count,
                "removed_observation_count":
                    removal_count,
                "retained_observation_count":
                    int(
                        retained_indices.size
                    ),
                "stressed_strength":
                    (
                        strength
                        if strength is not None
                        else np.nan
                    ),
                "stressed_sign":
                    sign,
                "estimability":
                    estimability,
                "non_estimable_reason":
                    (
                        None
                        if strength is not None
                        else reason
                    ),
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def summarize_missingness(
    full_relations: pd.DataFrame,
    stress_replicates: pd.DataFrame,
    *,
    repetitions: int,
    minimum_sign_fraction: float,
) -> pd.DataFrame:
    """Summarize missingness robustness for every relation."""
    records: list[dict[str, Any]] = []

    for full_row in full_relations.itertuples(
        index=False
    ):
        relation_rows = stress_replicates.loc[
            (
                stress_replicates["window_id"]
                == full_row.window_id
            )
            & (
                stress_replicates["relation_id"]
                == full_row.relation_id
            )
        ].copy()

        estimable = relation_rows.loc[
            relation_rows["estimability"]
            == "estimable"
        ].copy()

        estimable_count = int(
            len(estimable)
        )

        estimable_fraction = float(
            estimable_count
            / repetitions
        )

        if (
            full_row.estimability
            != "estimable"
            or full_row.sign is None
            or pd.isna(full_row.sign)
            or full_row.strength is None
        ):
            sign_fraction = np.nan
            median_absolute_change = np.nan
            robust = False
        elif estimable.empty:
            sign_fraction = 0.0
            median_absolute_change = np.nan
            robust = False
        else:
            full_sign = int(
                full_row.sign
            )

            sign_fraction = float(
                (
                    estimable["stressed_sign"]
                    == full_sign
                ).mean()
            )

            median_absolute_change = float(
                (
                    estimable["stressed_strength"]
                    - float(
                        full_row.strength
                    )
                )
                .abs()
                .median()
            )

            robust = bool(
                sign_fraction
                >= minimum_sign_fraction
            )

        reasons = (
            relation_rows.loc[
                relation_rows["estimability"]
                != "estimable",
                "non_estimable_reason",
            ]
            .fillna("unspecified")
            .value_counts()
            .sort_index()
            .to_dict()
        )

        records.append(
            {
                "experiment_id":
                    "CGIE3_ID_02",
                "window_id":
                    full_row.window_id,
                "relation_id":
                    full_row.relation_id,
                "source_id":
                    full_row.source_id,
                "target_id":
                    full_row.target_id,
                "full_baseline_strength":
                    full_row.strength,
                "full_baseline_sign":
                    full_row.sign,
                "missingness_repetitions":
                    repetitions,
                "missingness_estimable_count":
                    estimable_count,
                "missingness_estimable_fraction":
                    estimable_fraction,
                "missingness_sign_preservation_fraction":
                    sign_fraction,
                "missingness_median_absolute_strength_change":
                    median_absolute_change,
                "missingness_stress_robust":
                    robust,
                "missingness_non_estimable_reasons":
                    reasons,
            }
        )

    output = pd.DataFrame.from_records(
        records
    )

    if len(output) != len(
        full_relations
    ):
        fail(
            "Missingness summary relation count "
            "does not match candidate relations."
        )

    return output.sort_values(
        by=[
            "window_id",
            "source_id",
            "target_id",
            "relation_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build_missingness_stage_summary(
    summary: pd.DataFrame,
    *,
    repetitions: int,
) -> dict[str, Any]:
    """Build descriptive missingness-stage counts."""
    by_window: dict[str, Any] = {}

    for window_id, frame in summary.groupby(
        "window_id",
        sort=True,
    ):
        fully_estimable_count = int(
            (
                frame[
                    "missingness_estimable_count"
                ]
                == repetitions
            ).sum()
        )

        robust_count = int(
            frame[
                "missingness_stress_robust"
            ]
            .fillna(False)
            .sum()
        )

        valid_changes = (
            frame[
                "missingness_median_absolute_strength_change"
            ]
            .dropna()
            .astype(float)
        )

        by_window[
            str(window_id)
        ] = {
            "relation_count":
                int(len(frame)),
            "fully_estimable_count":
                fully_estimable_count,
            "robust_count":
                robust_count,
            "median_absolute_strength_change":
                (
                    float(
                        valid_changes.median()
                    )
                    if not valid_changes.empty
                    else None
                ),
        }

    return {
        "status":
            "COMPLETED",
        "relation_count":
            int(len(summary)),
        "repetitions_per_relation":
            int(repetitions),
        "expected_replication_count":
            int(
                len(summary)
                * repetitions
            ),
        "by_window":
            by_window,
    }


def evaluate_missingness(
    context: ExperimentContext,
) -> ExperimentContext:
    """Execute frozen simulated-missingness stress."""
    validate_context(
        context
    )

    contract = get_missingness_contract(
        context
    )

    baseline_tables = require_mapping(
        context.outputs[
            "baseline_feature_tables"
        ],
        "baseline_feature_tables",
    )

    full_relations_by_window = require_mapping(
        context.outputs[
            "candidate_relations_by_window"
        ],
        "candidate_relations_by_window",
    )

    stress_frames: list[
        pd.DataFrame
    ] = []

    for window_id in sorted(
        baseline_tables
    ):
        frame = baseline_tables[
            window_id
        ]

        full_relations = (
            full_relations_by_window[
                window_id
            ]
        )

        for full_relation in full_relations.itertuples(
            index=False
        ):
            stress_frame = stress_relation(
                frame=frame,
                full_relation=full_relation,
                window_id=window_id,
                missing_fraction=contract[
                    "simulated_missing_fraction"
                ],
                repetitions=contract[
                    "repetitions"
                ],
                base_seed=contract[
                    "random_seed"
                ],
                minimum_samples=contract[
                    "minimum_samples"
                ],
                minimum_unique_values=contract[
                    "minimum_unique_values"
                ],
            )

            stress_frames.append(
                stress_frame
            )

    if not stress_frames:
        fail(
            "No missingness-stress records were produced."
        )

    stress_replicates = pd.concat(
        stress_frames,
        ignore_index=True,
    )

    expected_replication_count = (
        136
        * contract["repetitions"]
    )

    if (
        len(stress_replicates)
        != expected_replication_count
    ):
        fail(
            "Missingness replication count is "
            f"{len(stress_replicates)}; expected "
            f"{expected_replication_count}."
        )

    full_relations = context.outputs[
        "candidate_relations"
    ]

    if not isinstance(
        full_relations,
        pd.DataFrame,
    ):
        fail(
            "candidate_relations must be a DataFrame."
        )

    missingness_summary = summarize_missingness(
        full_relations,
        stress_replicates,
        repetitions=contract[
            "repetitions"
        ],
        minimum_sign_fraction=contract[
            "minimum_sign_fraction"
        ],
    )

    if len(missingness_summary) != 136:
        fail(
            "Missingness summary must contain "
            f"136 relations, observed "
            f"{len(missingness_summary)}."
        )

    stage_summary = build_missingness_stage_summary(
        missingness_summary,
        repetitions=contract[
            "repetitions"
        ],
    )

    context.register_output(
        "missingness_replicates",
        stress_replicates,
    )

    context.register_output(
        "missingness_summary",
        missingness_summary,
    )

    context.register_output(
        "missingness_stage_summary",
        stage_summary,
    )

    context.register_runtime(
        "missingness_status",
        "COMPLETED",
    )

    return context
      
