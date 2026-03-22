import json

with open("outputs/latest/v49_report.txt") as f:
    lines = f.readlines()

status = {}
for line in lines:
    if "Status:" in line:
        status["state"] = line.split(":")[1].strip()
    if "Trend:" in line:
        status["trend"] = line.split(":")[1].strip()
    if "Avg prob" in line:
        status["avg_prob"] = float(line.split(":")[1])
    if "Max prob" in line:
        status["max_prob"] = float(line.split(":")[1])

with open("outputs/latest/status.json", "w") as f:
    json.dump(status, f, indent=2)

print("Status JSON created")
