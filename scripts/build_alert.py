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

if changes:
    lines = []
    lines.append("CGIE ALERT")
    lines.append("")
    lines.append("Meaningful change detected:")
    lines.extend(changes)

    ALERT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ALERT_JSON.write_text(
        json.dumps(
            {
                "alert": True,
                "changes": changes,
                "current": current,
            },
            indent=2
        ),
        encoding="utf-8",
    )

    print("ALERT CREATED")

else:
    if ALERT_TXT.exists():
        os.remove(ALERT_TXT)

    ALERT_JSON.write_text(
        json.dumps(
            {
                "alert": False,
                "current": current,
            },
            indent=2
        ),
        encoding="utf-8",
    )

    print("No meaningful changes")
