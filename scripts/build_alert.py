import json
import os

CURRENT = "outputs/latest/status.json"
PREVIOUS = "outputs/previous/status.json"
ALERT = "outputs/latest/alert.txt"

if not os.path.exists(CURRENT):
    print("Current status missing")
    raise SystemExit(1)

with open(CURRENT) as f:
    current = json.load(f)

previous = {}
if os.path.exists(PREVIOUS):
    with open(PREVIOUS) as f:
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

    with open(ALERT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("ALERT CREATED")
else:
    if os.path.exists(ALERT):
        os.remove(ALERT)
    print("No meaningful changes")
