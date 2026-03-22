# 🌍 Congruity Global Instability Engine (CGIE)

### Planetary Instability Radar · Public Demo

🔗 **Live Demo (Full Radar)**  
https://andrearomeo74-cloud.github.io/cgie-global-instability-radar/map.html  

🔗 **Live Alert (Minimal View)**  
https://andrearomeo74-cloud.github.io/cgie-global-instability-radar/alert.html  

---

## 🧠 What is CGIE

The **Congruity Global Instability Engine (CGIE)** is an experimental system designed to detect and interpret **systemic instability patterns** in complex geophysical environments.

Instead of predicting specific events, CGIE focuses on:

- detection of structural instability regimes  
- identification of temporal shifts  
- probabilistic signal aggregation  
- coherence-based filtering  
- early-phase transition signals  

---

## ⚙️ How it works

The system runs a fully automated pipeline:

1. **Data ingestion**
   - seismic event streams (global)

2. **Aggregation**
   - daily activity clustering  
   - probabilistic signal construction  

3. **Temporal analysis**
   - coherence filtering  
   - instability regime detection  

4. **State generation**
   - `status.json` (current system state)

5. **Change detection**
   - comparison vs previous state  

6. **Alert engine**
   - generates:
     - `alert.txt` (human readable)
     - `alert.json` (machine readable)

---

## 🚨 Alert system (Level 2)

CGIE includes a **graded alert system**:

| Level | Meaning |
|------|--------|
| 🟢 Stable | No significant change |
| 🟡 Low | Minor variation |
| 🟠 Medium | Moderate instability shift |
| 🔴 High | Elevated instability regime |

Alert triggers are based on:

- state transitions  
- trend changes  
- probabilistic thresholds  
- system coherence dynamics  

---

## 🌍 Public interface

### 1. Global Radar
- live alert state  
- system snapshot  
- embedded seismic visualization  

👉 `map.html`

---

### 2. Alert Monitor
- minimal real-time alert view  

👉 `alert.html`

---

## 📊 Example system state

```json
{
  "state": "elevated activity",
  "trend": "increasing",
  "avg_prob": 0.166,
  "max_prob": 0.62
}

## 🧪 Validation approach

The system is evaluated using:

- temporal proximity to real events  
- lead-time estimation  
- false positive control  
- signal coherence quality  

Evaluation focuses on **system behavior over time**, not single-event prediction accuracy.

Key aspects:

- **Temporal alignment**
  Measures how close detected instability signals are to real seismic events

- **Lead time analysis**
  Evaluates how early the system detects structural changes before events

- **False positive control**
  Ensures alerts remain sparse and meaningful rather than noisy

- **Signal coherence**
  Validates that detected patterns are structurally consistent, not random fluctuations

⚠️ CGIE is **not a deterministic prediction system**  
It is a **systemic instability detection framework**

## 🧩 Repository structure

baseline/    → core signal engine  
scripts/     → pipeline utilities  
outputs/     → generated states and alerts  
docs/        → documentation  

index.html   → radar visualization  
alert.html   → alert UI  
map.html     → full public dashboard  

---

## ⚙️ Automation

The system runs via GitHub Actions:

- daily baseline execution  
- automatic state update  
- alert generation  
- public deployment (GitHub Pages)  

---

## 📡 What makes CGIE different

- not event prediction → **system state interpretation**  
- focuses on **transitions, not outcomes**  
- integrates **temporal + probabilistic coherence**  
- designed as a **real-time observatory layer**  

---

## 🚀 Next steps

- multi-region instability mapping  
- anomaly clustering visualization  
- cross-domain extension (climate, finance, infrastructure)  
- integration with Congruity framework (ICᵀ)  

---

## 📄 License

MIT  

---

## 👤 Author

Andrea Romeo  
Congruity Framework, ICᵀ System
