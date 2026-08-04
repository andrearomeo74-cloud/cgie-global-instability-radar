"""
CGIE3-ID-02 moving-block bootstrap stage.

This module estimates uncertainty for every full-baseline candidate
relation using the frozen moving-block bootstrap configuration.

It produces:

- one record for every bootstrap replication;
- confidence intervals for every relation;
- confidence-interval width;
- sign-support information;
- explicit non-estimable outcomes.

It does not classify relations as eligible, candidate or rejected.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from congruity.core import ExperimentContext


class BootstrapStageError(ValueError):
    """Raised when bootstrap execution violates the frozen contract."""


def fail(message: str) -> None:
    """Raise a bootstrap-stage error."""
    raise BootstrapStageError(
        str(message).strip()
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Require a mapping-like configuration value."""
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
        raise BootstrapStageError(
            f"{field_name} must be an integer."
        ) from exc

    if normalized <= 0:
        fail(
            f"{field_name} must be greater than zero."
        )

    return normalized


def require_probability(
    value: Any,
    field_name: str,
) -> float:
    """Require a probability strictly inside (0, 1)."""
    try:
        normalized = float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise BootstrapStageError(
            f"{field_name} must be numeric."
        ) from exc

    if not 0.0 < normalized < 1.0:
        fail(
            f"{field_name} must lie inside (0, 1)."
        )

    return normalized


def validate_context(
    context: ExperimentContext,
) -> None:
    """Validate bootstrap-stage prerequisites."""
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
        "components_by_window",
        "candidate_relations",
        "candidate_relations_by_window",
        "relation_persistence",
        "leave_one_block_out",
    }

    missing_outputs = sorted(
        required_outputs
        - set(context.outputs)
    )

    if missing_outputs:
        fail(
            "Bootstrap prerequisites are missing: "
            + ", ".join(missing_outputs)
        )


def get_bootstrap_contract(
    context: ExperimentContext,
) -> dict[str, Any]:
    """Extract and validate the frozen bootstrap configuration."""
    reproducibility = require_mapping(
        context.configuration.get(
            "relation_reproducibility"
        ),
        "relation_reproducibility",
    )

    bootstrap = require_mapping(
        reproducibility.get(
            "bootstrap"
        ),
        "relation_reproducibility.bootstrap",
    )

    estimability = require_mapping(
        context.configuration.get(
            "estimability"
        ),
        "estimability",
    )

    if bootstrap.get("enabled") is not True:
        fail(
            "Moving-block bootstrap must remain enabled."
        )

    method = str(
        bootstrap.get(
            "method",
            ""
        )
    ).strip()

    if method != "moving_block_bootstrap":
        fail(
            "CGIE3-ID-02 requires moving_block_bootstrap."
        )

    repetitions = require_positive_integer(
        bootstrap.get("repetitions"),
        "relation_reproducibility."
        "bootstrap.repetitions",
    )

    if repetitions != 500:
        fail(
            "CGIE3-ID-02 requires exactly "
            "500 bootstrap repetitions."
        )

    block_length = require_positive_integer(
        bootstrap.get(
            "block_length_observations"
        ),
        "relation_reproducibility.bootstrap."
        "block_length_observations",
    )

    confidence_level = require_probability(
        bootstrap.get(
            "confidence_level"
        ),
        "relation_reproducibility.bootstrap."
        "confidence_level",
    )

    random_seed = require_positive_integer(
        bootstrap.get(
            "random_seed"
        ),
        "relation_reproducibility.bootstrap.random_seed",
    )

    minimum_samples = require_positive_integer(
        estimability.get(
            "minimum_paired_observations"
        ),
        "estimability.minimum_paired_observations",
    )

    minimum_unique_values = (
        require_positive_integer(
            estimability.get(
                "minimum_unique_values_per_component"
            ),
            "estimability."
            "minimum_unique_values_per_component",
        )
    )

    return {
        "repetitions":
            repetitions,
        "block_length":
            block_length,
        "confidence_level":
            confidence_level,
        "random_seed":
            random_seed,
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
    """
    Derive a stable relation-specific seed.

    This prevents the result of one relation from depending upon the
    iteration order or estimability of another relation.
    """
    token = (
        f"{base_seed}::{window_id}::{relation_id}"
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
    """Prepare finite paired numeric observations in temporal order."""
    required = {
        source_id,
        target_id,
    }

    missing = sorted(
        required
        - set(frame.columns)
    )

    if missing:
        fail(
            "Bootstrap frame is missing components: "
            + ", ".join(missing)
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


def moving_block_indices(
    observation_count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one circular moving-block bootstrap sample.

    Circular indexing allows blocks beginning near the end of the
    sequence to wrap around without shortening the resample.
    """
    if observation_count <= 0:
        fail(
            "observation_count must be positive."
        )

    effective_block_length = min(
        block_length,
        observation_count,
    )

    block_count = int(
        math.ceil(
            observation_count
            / effective_block_length
        )
    )

    starts = rng.integers(
        low=0,
        high=observation_count,
        size=block_count,
    )

    blocks = [
        (
            start
            + np.arange(
                effective_block_length
            )
        )
        % observation_count
        for start in starts
    ]

    indices = np.concatenate(
        blocks
    )[:observation_count]

    return indices.astype(
        int,
        copy=False,
    )


def estimate_bootstrap_strength(
    source_values: np.ndarray,
    target_values: np.ndarray,
    indices: np.ndarray,
    *,
    minimum_unique_values: int,
) -> tuple[float | None, str]:
    """Estimate Spearman strength for one bootstrap resample."""
    source_sample = source_values[
        indices
    ]

    target_sample = target_values[
        indices
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


def bootstrap_relation(
    frame: pd.DataFrame,
    full_relation: Any,
    *,
    window_id: str,
    repetitions: int,
    block_length: int,
    base_seed: int,
    minimum_samples: int,
    minimum_unique_values: int,
) -> pd.DataFrame:
    """Run frozen moving-block bootstrap for one relation."""
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
                    "paired_observation_count":
                        observation_count,
                    "bootstrap_strength":
                        np.nan,
                    "bootstrap_sign":
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

    if observation_count < minimum_samples:
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
                    "paired_observation_count":
                        observation_count,
                    "bootstrap_strength":
                        np.nan,
                    "bootstrap_sign":
                        np.nan,
                    "estimability":
                        "non_estimable",
                    "non_estimable_reason":
                        "insufficient_observations",
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

    rng = np.random.default_rng(
        relation_seed
    )

    for repetition in range(
        1,
        repetitions + 1,
    ):
        indices = moving_block_indices(
            observation_count,
            block_length,
            rng,
        )

        strength, reason = (
            estimate_bootstrap_strength(
                source_values,
                target_values,
                indices,
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
                "paired_observation_count":
                    observation_count,
                "bootstrap_strength":
                    (
                        strength
                        if strength is not None
                        else np.nan
                    ),
                "bootstrap_sign":
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


def summarize_bootstrap(
    full_relations: pd.DataFrame,
    bootstrap_replicates: pd.DataFrame,
    *,
    confidence_level: float,
    repetitions: int,
) -> pd.DataFrame:
    """Summarize bootstrap uncertainty for every candidate relation."""
    alpha = 1.0 - confidence_level

    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - lower_quantile

    records: list[dict[str, Any]] = []

    for full_row in full_relations.itertuples(
        index=False
    ):
        relation_rows = bootstrap_replicates.loc[
            (
                bootstrap_replicates["window_id"]
                == full_row.window_id
            )
            & (
                bootstrap_replicates["relation_id"]
                == full_row.relation_id
            )
        ].copy()

        estimable_strengths = (
            relation_rows.loc[
                relation_rows["estimability"]
                == "estimable",
                "bootstrap_strength",
            ]
            .dropna()
            .astype(float)
        )

        estimable_count = int(
            len(estimable_strengths)
        )

        estimable_fraction = float(
            estimable_count
            / repetitions
        )

        if estimable_strengths.empty:
            lower = np.nan
            upper = np.nan
            width = np.nan
            median = np.nan
            mean = np.nan
            standard_deviation = np.nan
            sign_support = np.nan
            interval_crosses_zero = np.nan
            interval_supports_full_sign = False
        else:
            lower = float(
                estimable_strengths.quantile(
                    lower_quantile
                )
            )

            upper = float(
                estimable_strengths.quantile(
                    upper_quantile
                )
            )

            width = float(
                upper - lower
            )

            median = float(
                estimable_strengths.median()
            )

            mean = float(
                estimable_strengths.mean()
            )

            standard_deviation = float(
                estimable_strengths.std(
                    ddof=1
                )
            )

            if (
                full_row.sign is None
                or pd.isna(full_row.sign)
            ):
                sign_support = np.nan
                interval_supports_full_sign = False
            else:
                full_sign = int(
                    full_row.sign
                )

                sign_support = float(
                    (
                        np.sign(
                            estimable_strengths
                        )
                        == full_sign
                    ).mean()
                )

                if full_sign > 0:
                    interval_supports_full_sign = bool(
                        lower > 0.0
                    )
                elif full_sign < 0:
                    interval_supports_full_sign = bool(
                        upper < 0.0
                    )
                else:
                    interval_supports_full_sign = bool(
                        lower <= 0.0 <= upper
                    )

            interval_crosses_zero = bool(
                lower <= 0.0 <= upper
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
                "bootstrap_repetitions":
                    repetitions,
                "bootstrap_estimable_count":
                    estimable_count,
                "bootstrap_estimable_fraction":
                    estimable_fraction,
                "bootstrap_mean_strength":
                    mean,
                "bootstrap_median_strength":
                    median,
                "bootstrap_standard_deviation":
                    standard_deviation,
                "bootstrap_ci_lower":
                    lower,
                "bootstrap_ci_upper":
                    upper,
                "bootstrap_ci_width":
                    width,
                "bootstrap_interval_crosses_zero":
                    interval_crosses_zero,
                "bootstrap_interval_supports_full_sign":
                    interval_supports_full_sign,
                "bootstrap_sign_support_fraction":
                    sign_support,
                "bootstrap_non_estimable_reasons":
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
            "Bootstrap summary relation count "
            "does not match full-baseline relations."
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


def build_bootstrap_stage_summary(
    bootstrap_summary: pd.DataFrame,
    *,
    repetitions: int,
) -> dict[str, Any]:
    """Build descriptive bootstrap-stage counts."""
    by_window: dict[str, Any] = {}

    for window_id, frame in (
        bootstrap_summary.groupby(
            "window_id",
            sort=True,
        )
    ):
        fully_estimable_count = int(
            (
                frame[
                    "bootstrap_estimable_count"
                ]
                == repetitions
            ).sum()
        )

        interval_support_count = int(
            frame[
                "bootstrap_interval_supports_full_sign"
            ]
            .fillna(False)
            .sum()
        )

        valid_widths = (
            frame[
                "bootstrap_ci_width"
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
            "interval_supports_full_sign_count":
                interval_support_count,
            "median_confidence_interval_width":
                (
                    float(
                        valid_widths.median()
                    )
                    if not valid_widths.empty
                    else None
                ),
        }

    return {
        "status":
            "COMPLETED",
        "relation_count":
            int(len(bootstrap_summary)),
        "repetitions_per_relation":
            int(repetitions),
        "expected_replication_count":
            int(
                len(bootstrap_summary)
                * repetitions
            ),
        "by_window":
            by_window,
    }


def evaluate_bootstrap(
    context: ExperimentContext,
) -> ExperimentContext:
    """Execute frozen moving-block bootstrap evaluation."""
    validate_context(
        context
    )

    contract = get_bootstrap_contract(
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

    bootstrap_frames: list[
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
            bootstrap_frame = bootstrap_relation(
                frame=frame,
                full_relation=full_relation,
                window_id=window_id,
                repetitions=contract[
                    "repetitions"
                ],
                block_length=contract[
                    "block_length"
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

            bootstrap_frames.append(
                bootstrap_frame
            )

    if not bootstrap_frames:
        fail(
            "No bootstrap replications were produced."
        )

    bootstrap_replicates = pd.concat(
        bootstrap_frames,
        ignore_index=True,
    )

    expected_replication_count = (
        136
        * contract["repetitions"]
    )

    if (
        len(bootstrap_replicates)
        != expected_replication_count
    ):
        fail(
            "Bootstrap replication count is "
            f"{len(bootstrap_replicates)}; expected "
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

    bootstrap_summary = summarize_bootstrap(
        full_relations,
        bootstrap_replicates,
        confidence_level=contract[
            "confidence_level"
        ],
        repetitions=contract[
            "repetitions"
        ],
    )

    if len(bootstrap_summary) != 136:
        fail(
            "Bootstrap summary must contain "
            f"136 relations, observed "
            f"{len(bootstrap_summary)}."
        )

    stage_summary = (
        build_bootstrap_stage_summary(
            bootstrap_summary,
            repetitions=contract[
                "repetitions"
            ],
        )
    )

    context.register_output(
        "bootstrap_replicates",
        bootstrap_replicates,
    )

    context.register_output(
        "bootstrap_summary",
        bootstrap_summary,
    )

    context.register_output(
        "bootstrap_stage_summary",
        stage_summary,
    )

    context.register_runtime(
        "bootstrap_status",
        "COMPLETED",
    )

    return context
             
