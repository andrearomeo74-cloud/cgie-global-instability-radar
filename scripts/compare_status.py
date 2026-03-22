import json
import os

CURRENT = "outputs/latest/status.json"
PREVIOUS = "outputs/previous/status.json"

if not os.path.exists(PREVIOUS):
    print("No previous status, initializing baseline.")
    os.makedirs("outputs/previous", exist_ok=True)
    with open(PREVIOUS, "w") as f:
        json.dump({}, f)

with open(CURRENT) as f:
    current = json.load(f)

with open(PREVIOUS) as f:
    previous = json.load(f)

changes = []

for key in current:
    if key in previous and current[key] != previous[key]:
        changes.append(f"{key}: {previous[key]} -> {current[key]}")

if changes:
    print("SIGNIFICANT CHANGE DETECTED")
    for c in changes:
        print(c)
else:
    print("No meaningful changes")
