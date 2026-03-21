import pandas as pd

# ================================
# LOAD DATA
# ================================
def load_outputs(daily_path, alerts_path):
    daily = pd.read_csv(daily_path)
    alerts = pd.read_csv(alerts_path)

    daily["date_utc"] = pd.to_datetime(daily["date_utc"], utc=True)
    alerts["date_utc"] = pd.to_datetime(alerts["date_utc"], utc=True)

    return daily, alerts


# ================================
# CURRENT STATUS
# ================================
def compute_status(daily):
    recent = daily.tail(7)

    avg_prob = recent["prob"].mean()
    max_prob = recent["prob"].max()

    if max_prob > 0.65:
        status = "strong activity"
    elif max_prob > 0.55:
        status = "elevated activity"
    elif max_prob > 0.45:
        status = "early signals"
    else:
        status = "quiet"

    return status, avg_prob, max_prob


# ================================
# TREND
# ================================
def compute_trend(daily):
    recent = daily.tail(7)

    delta = recent["prob"].iloc[-1] - recent["prob"].iloc[0]

    if delta > 0.1:
        return "increasing"
    elif delta < -0.1:
        return "decreasing"
    else:
        return "stable"


# ================================
# LAST ALERT
# ================================
def get_last_alert(alerts):
    if len(alerts) == 0:
        return None
    return alerts.iloc[-1]


# ================================
# REPORT GENERATION
# ================================
def generate_report(daily, alerts):
    status, avg_prob, max_prob = compute_status(daily)
    trend = compute_trend(daily)
    last_alert = get_last_alert(alerts)

    last_30_days = daily["date_utc"].max() - pd.Timedelta(days=30)
    recent_alerts = alerts[alerts["date_utc"] >= last_30_days]

    print("\n===== V49 FINAL REPORT =====")

    print("\n--- CURRENT STATUS ---")
    print("Status:", status)
    print("Trend:", trend)
    print("Avg prob (7d):", round(avg_prob, 3))
    print("Max prob (7d):", round(max_prob, 3))

    print("\n--- LAST ALERT ---")
    if last_alert is not None:
        print("Date:", last_alert["date_utc"])
        print("State:", last_alert["state"])
        print("Prob:", round(last_alert["prob"], 3))
        print("Reason:", last_alert["reason"])
    else:
        print("No alerts")

    print("\n--- ALERT ACTIVITY ---")
    print("Alerts last 30 days:", len(recent_alerts))
    print("Total alerts:", len(alerts))

    print("\n--- INTERPRETATION ---")

    if status == "strong activity":
        print("High probability of ongoing or imminent activity.")
    elif status == "elevated activity":
        print("Elevated activity detected. Monitoring recommended.")
    elif status == "early signals":
        print("Early signals detected. Possible activation phase.")
    else:
        print("System is currently quiet.")

    if trend == "increasing":
        print("Trend indicates increasing pressure.")
    elif trend == "decreasing":
        print("Trend indicates relaxation.")
    else:
        print("Trend is stable.")


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    daily, alerts = load_outputs("v48_daily.csv", "v48_alerts.csv")

    generate_report(daily, alerts)
