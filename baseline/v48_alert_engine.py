import pandas as pd
import numpy as np

# ================================
# LOAD DATA
# ================================
def load_events(path):
    df = pd.read_csv(path)

    df["time_utc"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df["magnitude"] = pd.to_numeric(df["MD"], errors="coerce")

    df = df.dropna(subset=["time_utc", "magnitude"]).copy()
    df["date_utc"] = df["time_utc"].dt.floor("D")

    return df


# ================================
# NORMALIZATION
# ================================
def normalize(series):
    if series.max() - series.min() == 0:
        return np.zeros(len(series))
    return (series - series.min()) / (series.max() - series.min())


# ================================
# DAILY AGGREGATION
# ================================
def build_daily(df):
    daily = df.groupby("date_utc").agg(
        n_events=("magnitude", "count"),
        max_mag=("magnitude", "max")
    ).reset_index()

    full = pd.date_range(
        daily["date_utc"].min(),
        daily["date_utc"].max(),
        freq="D",
        tz="UTC"
    )

    daily = daily.set_index("date_utc").reindex(full).reset_index()
    daily = daily.rename(columns={"index": "date_utc"})
    daily.fillna(0, inplace=True)

    daily["mag_n"] = normalize(daily["max_mag"])
    daily["cnt_n"] = normalize(daily["n_events"])

    return daily


# ================================
# CORE ENGINE
# ================================
def compute_engine(daily):
    memory = 0
    prev_prob = 0

    probs = []
    deltas = []
    memories = []

    for i in range(len(daily)):
        mag = daily.loc[i, "mag_n"]
        cnt = daily.loc[i, "cnt_n"]

        memory = 0.78 * memory + 0.22 * mag + 0.22 * cnt
        accel = 0.28 * mag + 0.26 * cnt + 0.08 * memory

        prob = 1 / (1 + np.exp(-5.5 * (accel - 0.42)))
        delta = prob - prev_prob

        prev_prob = prob

        probs.append(prob)
        deltas.append(delta)
        memories.append(memory)

    daily["prob"] = probs
    daily["prob_delta"] = deltas
    daily["memory"] = memories

    return daily


# ================================
# TEMPORAL FEATURES
# ================================
def compute_features(daily):
    daily["prob_delta_1"] = daily["prob_delta"].shift(1).fillna(0)
    daily["prob_delta_2"] = daily["prob_delta"].shift(2).fillna(0)

    daily["coherence"] = (
        (daily["prob_delta"] > 0) &
        (daily["prob_delta_1"] > 0)
    )

    daily["strong_coherence"] = (
        (daily["prob_delta"] > 0) &
        (daily["prob_delta_1"] > 0) &
        (daily["prob_delta_2"] > 0)
    )

    daily["prob_var_3"] = daily["prob"].rolling(3).var().fillna(0)
    daily["prob_var_5"] = daily["prob"].rolling(5).var().fillna(0)

    daily["stability"] = 1 / (1 + daily["prob_var_3"] * 10 + daily["prob_var_5"] * 5)

    daily["coherence_score"] = (
        0.4 * daily["prob"] +
        0.3 * daily["memory"] +
        0.2 * daily["coherence"].astype(int) +
        0.1 * daily["stability"]
    )

    daily["quality_score"] = (
        0.5 * daily["coherence_score"] +
        0.3 * daily["prob"] +
        0.2 * daily["stability"]
    )

    return daily


# ================================
# ALERT ENGINE (V48)
# ================================
def run_alert_engine(daily):
    threshold = daily["quality_score"].quantile(0.75)

    alerts = []
    prev_quality = 0
    prev_state = "quiet"

    def get_state(row):
        if row["prob"] > 0.65 and row["strong_coherence"]:
            return "strong"
        elif row["prob"] > 0.55 and row["coherence"]:
            return "confirmed"
        elif row["prob"] > 0.45:
            return "early"
        return "quiet"

    for _, row in daily.iterrows():
        state = get_state(row)
        prob = row["prob"]
        quality = row["quality_score"]

        alert = False
        reason = None

        if quality >= threshold and row["strong_coherence"]:
            alert = True
            reason = "quality_coherent"

        elif quality > prev_quality + 0.12:
            alert = True
            reason = "quality_jump"

        elif state in ["confirmed", "strong"] and state != prev_state:
            alert = True
            reason = "state_escalation"

        elif prob > 0.65 and row["strong_coherence"]:
            alert = True
            reason = "breakout"

        if alert:
            alerts.append({
                "date_utc": row["date_utc"],
                "state": state,
                "prob": prob,
                "quality": quality,
                "reason": reason
            })

        prev_quality = quality
        prev_state = state

    return pd.DataFrame(alerts)


# ================================
# PIPELINE
# ================================
def run_v48(path):
    df = load_events(path)
    daily = build_daily(df)
    daily = compute_engine(daily)
    daily = compute_features(daily)

    alerts = run_alert_engine(daily)

    return daily, alerts


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    daily, alerts = run_v48("events.csv")

    print("\n===== V48 ALERT ENGINE =====")
    print(alerts.head())

    alerts.to_csv("v48_alerts.csv", index=False)
    daily.to_csv("v48_daily.csv", index=False)
