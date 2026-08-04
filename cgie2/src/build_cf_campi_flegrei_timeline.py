from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ALERTS_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CF_RETRO_01_global_alerts.csv"
)

CATALOGUE_PATH = (
    REPOSITORY_ROOT
    / "outputs"
    / "CF_RETRO_01_catalog_frozen.csv"
)

OUTPUT_DIRECTORY = REPOSITORY_ROOT / "outputs"

TIMELINE_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "CF_CASE_01_campi_flegrei_timeline.csv"
)

TIMELINE_PNG_PATH = (
    OUTPUT_DIRECTORY
    / "CF_CASE_01_campi_flegrei_timeline.png"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "CF_CASE_01_summary.json"
)


ANALYSIS_START = pd.Timestamp(
    "2026-06-14T00:00:00Z"
)

TARGET_EVENT_TIME = pd.Timestamp(
    "2026-07-31T17:46:43Z"
)

TARGET_EVENT_MAGNITUDE = 4.7

MINIMUM_EVENT_MAGNITUDE = 2.0


STATE_ORDER = {
    "normal": 0,
    "attention": 1,
    "anomaly": 2,
    "structural": 3,
    "critical": 4,
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def find_column(
    columns: Iterable[str],
    candidates: list[str],
) -> str:
    normalized = {
        str(column).strip().lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    fail(
        "None of the expected columns were found: "
        + ", ".join(candidates)
    )


def normalize_state(value: object) -> str:
    text = str(value).strip().lower()

    aliases = {
        "stable": "normal",
        "perturbation": "attention",
        "instability": "anomaly",
        "critical_cluster": "structural",
        "transition_risk": "critical",
    }

    return aliases.get(text, text)


def strongest_state(row: pd.Series) -> str:
    states: list[str] = []

    for value in row:
        state = normalize_state(value)

        if state in STATE_ORDER:
            states.append(state)

    if not states:
        return "normal"

    return max(
        states,
        key=lambda state: STATE_ORDER[state],
    )


def load_alerts() -> pd.DataFrame:
    if not ALERTS_PATH.exists():
        fail(f"Alert file not found: {ALERTS_PATH}")

    alerts = pd.read_csv(ALERTS_PATH)

    if alerts.empty:
        fail("Global alert table is empty.")

    timestamp_column = find_column(
        alerts.columns,
        [
            "endpoint_utc",
            "time_utc",
            "timestamp_utc",
            "datetime_utc",
            "utc",
        ],
    )

    alerts["timestamp_utc"] = pd.to_datetime(
        alerts[timestamp_column],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    if alerts["timestamp_utc"].isna().any():
        fail("Invalid timestamps detected in alert table.")

    excluded_columns = {
        timestamp_column,
        "timestamp_utc",
        "alert_count",
        "confirmation_count",
        "confirmed_windows",
        "window_count",
        "episode_id",
    }

    state_columns = [
        column
        for column in alerts.columns
        if column not in excluded_columns
    ]

    if not state_columns:
        fail("No alert-state columns were detected.")

    alerts["system_state"] = alerts[
        state_columns
    ].apply(
        strongest_state,
        axis=1,
    )

    alerts["state_score"] = alerts[
        "system_state"
    ].map(STATE_ORDER)

    alerts = alerts.loc[
        (
            alerts["timestamp_utc"]
            >= ANALYSIS_START
        )
        & (
            alerts["timestamp_utc"]
            <= TARGET_EVENT_TIME
        )
    ].copy()

    if alerts.empty:
        fail(
            "No alert rows fall inside the "
            "Campi Flegrei analysis period."
        )

    return alerts.sort_values(
        "timestamp_utc",
        kind="stable",
    ).reset_index(drop=True)


def load_catalogue() -> pd.DataFrame:
    if not CATALOGUE_PATH.exists():
        fail(f"Catalogue file not found: {CATALOGUE_PATH}")

    catalogue = pd.read_csv(CATALOGUE_PATH)

    if catalogue.empty:
        fail("Frozen catalogue is empty.")

    time_column = find_column(
        catalogue.columns,
        [
            "time_utc",
            "event_time_utc",
            "datetime_utc",
            "timestamp_utc",
        ],
    )

    magnitude_column = find_column(
        catalogue.columns,
        [
            "magnitude",
            "mag",
            "event_magnitude",
        ],
    )

    catalogue["event_time_utc"] = pd.to_datetime(
        catalogue[time_column],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    catalogue["magnitude"] = pd.to_numeric(
        catalogue[magnitude_column],
        errors="coerce",
    )

    catalogue = catalogue.dropna(
        subset=[
            "event_time_utc",
            "magnitude",
        ]
    ).copy()

    catalogue = catalogue.loc[
        (
            catalogue["event_time_utc"]
            >= ANALYSIS_START
        )
        & (
            catalogue["event_time_utc"]
            <= TARGET_EVENT_TIME
        )
        & (
            catalogue["magnitude"]
            >= MINIMUM_EVENT_MAGNITUDE
        )
    ].copy()

    return catalogue.sort_values(
        "event_time_utc",
        kind="stable",
    ).reset_index(drop=True)


def identify_non_normal_episodes(
    alerts: pd.DataFrame,
) -> pd.DataFrame:
    output = alerts.copy()

    output["is_non_normal"] = (
        output["state_score"] > 0
    )

    output["episode_start_flag"] = (
        output["is_non_normal"]
        & ~output["is_non_normal"]
        .shift(fill_value=False)
    )

    output["episode_id"] = (
        output["episode_start_flag"].cumsum()
    )

    episodes = (
        output.loc[output["is_non_normal"]]
        .groupby("episode_id", as_index=False)
        .agg(
            episode_start_utc=(
                "timestamp_utc",
                "min",
            ),
            episode_end_utc=(
                "timestamp_utc",
                "max",
            ),
            maximum_state_score=(
                "state_score",
                "max",
            ),
        )
    )

    reverse_state_order = {
        value: key
        for key, value in STATE_ORDER.items()
    }

    episodes["maximum_state"] = episodes[
        "maximum_state_score"
    ].map(reverse_state_order)

    episodes["duration_hours"] = (
        (
            episodes["episode_end_utc"]
            - episodes["episode_start_utc"]
        ).dt.total_seconds()
        / 3600.0
    ) + 1.0

    return episodes


def build_summary(
    alerts: pd.DataFrame,
    catalogue: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, object]:
    non_normal = alerts.loc[
        alerts["state_score"] > 0
    ]

    if episodes.empty:
        last_episode = None
        lead_from_start = None
        lead_from_end = None
    else:
        last_row = episodes.iloc[-1]

        last_episode = {
            "start_utc": (
                last_row["episode_start_utc"]
                .isoformat()
            ),
            "end_utc": (
                last_row["episode_end_utc"]
                .isoformat()
            ),
            "maximum_state": (
                last_row["maximum_state"]
            ),
            "duration_hours": float(
                last_row["duration_hours"]
            ),
        }

        lead_from_start = (
            TARGET_EVENT_TIME
            - last_row["episode_start_utc"]
        ).total_seconds() / 3600.0

        lead_from_end = (
            TARGET_EVENT_TIME
            - last_row["episode_end_utc"]
        ).total_seconds() / 3600.0

    return {
        "case_id": "CF_CASE_01",
        "location": "Campi Flegrei",
        "analysis_start_utc": (
            ANALYSIS_START.isoformat()
        ),
        "analysis_end_utc": (
            TARGET_EVENT_TIME.isoformat()
        ),
        "target_event": {
            "time_utc": (
                TARGET_EVENT_TIME.isoformat()
            ),
            "magnitude": TARGET_EVENT_MAGNITUDE,
        },
        "alert_rows": int(len(alerts)),
        "non_normal_rows": int(len(non_normal)),
        "non_normal_fraction": float(
            len(non_normal) / len(alerts)
        ),
        "episode_count": int(len(episodes)),
        "catalogue_events_above_threshold": int(
            len(catalogue)
        ),
        "minimum_catalogue_magnitude": (
            MINIMUM_EVENT_MAGNITUDE
        ),
        "last_non_normal_episode": last_episode,
        "lead_time_from_episode_start_hours": (
            None
            if lead_from_start is None
            else float(lead_from_start)
        ),
        "lead_time_from_episode_end_hours": (
            None
            if lead_from_end is None
            else float(lead_from_end)
        ),
        "interpretation": (
            "Temporal pre-event association only. "
            "No deterministic earthquake prediction "
            "or physical causality is claimed."
        ),
    }


def write_timeline_csv(
    alerts: pd.DataFrame,
) -> None:
    columns = [
        "timestamp_utc",
        "system_state",
        "state_score",
    ]

    alerts[columns].to_csv(
        TIMELINE_CSV_PATH,
        index=False,
    )


def write_summary(
    summary: dict[str, object],
) -> None:
    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def create_plot(
    alerts: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(15, 7)
    )

    axis.step(
        alerts["timestamp_utc"],
        alerts["state_score"],
        where="post",
        linewidth=1.6,
        label="CGIE system state",
    )

    if not catalogue.empty:
        magnitude_scale = (
            catalogue["magnitude"]
            / catalogue["magnitude"].max()
        ) * 3.5

        axis.scatter(
            catalogue["event_time_utc"],
            magnitude_scale,
            marker="o",
            alpha=0.55,
            label=(
                f"Seismic events M≥"
                f"{MINIMUM_EVENT_MAGNITUDE:.1f}"
            ),
        )

    axis.axvline(
        TARGET_EVENT_TIME,
        linestyle="--",
        linewidth=1.8,
        label=(
            "Target event "
            f"M{TARGET_EVENT_MAGNITUDE:.1f}"
        ),
    )

    axis.set_yticks(
        list(STATE_ORDER.values())
    )

    axis.set_yticklabels(
        [
            "Normal",
            "Attention",
            "Anomaly",
            "Structural",
            "Critical",
        ]
    )

    axis.set_xlabel("UTC time")
    axis.set_ylabel("CGIE state / scaled magnitude")

    axis.set_title(
        "CF-CASE-01 — Campi Flegrei\n"
        "CGIE states and seismic activity before "
        "the 31 July 2026 event"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend(
        loc="upper left"
    )

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        TIMELINE_PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    alerts = load_alerts()
    catalogue = load_catalogue()

    episodes = identify_non_normal_episodes(
        alerts
    )

    summary = build_summary(
        alerts=alerts,
        catalogue=catalogue,
        episodes=episodes,
    )

    write_timeline_csv(alerts)
    write_summary(summary)

    episodes.to_csv(
        OUTPUT_DIRECTORY
        / "CF_CASE_01_non_normal_episodes.csv",
        index=False,
    )

    create_plot(
        alerts=alerts,
        catalogue=catalogue,
    )

    print("=" * 72)
    print("CF-CASE-01 — CAMPI FLEGREI TIMELINE")
    print("=" * 72)
    print(f"Alert rows: {len(alerts)}")
    print(f"Non-normal episodes: {len(episodes)}")
    print(
        "Events above threshold: "
        f"{len(catalogue)}"
    )
    print(f"Generated: {TIMELINE_CSV_PATH}")
    print(f"Generated: {TIMELINE_PNG_PATH}")
    print(f"Generated: {SUMMARY_JSON_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
