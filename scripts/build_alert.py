import json
import os
from pathlib import Path

CURRENT = Path("outputs/latest/status.json")
PREVIOUS = Path("outputs/previous/status.json")
ALERT_TXT = Path("outputs/latest/alert.txt")
ALERT_JSON = Path("outputs/latest/alert.json")

if not CURRENT.exists():
    print("Current status missing")
    raise SystemExit(1)

with open(CURRENT, "r", encoding="utf-8") as f:
    current = json.load(f)

previous = {}
if PREVIOUS.exists():
    with open(PREVIOUS, "r", encoding="utf-8") as f:
        previous = json.load(f)

changes = []

for key in current:
    old = previous.get(key, None)
    new = current.get(key, None)
    if old != new:
        changes.append(f"{key}: {old} -> {new}")

avg_prob = float(current.get("avg_prob", 0))
max_prob = float(current.get("max_prob", 0))
trend = str(current.get("trend", "")).lower()
state = str(current.get("state", "")).lower()

level = "low"
summary = "System stable"
risk_color = "green"

if changes:
    if max_prob >= 0.60 or "increasing" in trend or "elevated" in state:
        level = "high"
        summary = "Meaningful change detected, elevated instability regime"
        risk_color = "red"
    elif max_prob >= 0.35 or avg_prob >= 0.12:
        level = "medium"
        summary = "Meaningful change detected, moderate instability variation"
        risk_color = "orange"
    else:
        level = "low"
        summary = "Meaningful change detected, low intensity variation"
        risk_color = "gold"

    lines = []
    lines.append("CGIE ALERT")
    lines.append("")
    lines.append(summary)
    lines.append(f"Level: {level.upper()}")
    lines.append("")
    lines.append("Detected changes:")
    lines.extend(changes)

    ALERT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ALERT_JSON.write_text(
        json.dumps(
            {
                "alert": True,
                "level": level,
                "summary": summary,
                "risk_color": risk_color,
                "changes": changes,
                "current": current,
            },
            indent=2
        ),
        encoding="utf-8",
    )

    print("ALERT CREATED")

else:
    ALERT_JSON.write_text(
        json.dumps(
            {
                "alert": False,
                "level": "stable",
                "summary": "System stable",
                "risk_color": "green",
                "current": current,
            },
            indent=2
        ),
        encoding="utf-8",
    )

    if ALERT_TXT.exists():
        os.remove(ALERT_TXT)

    print("No meaningful changes")
