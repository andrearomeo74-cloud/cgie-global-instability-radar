# Congruity Global Instability Engine  
### Planetary Seismic Instability Radar

Live site: https://andrearomeo74-cloud.github.io/cgie-global-instability-radar/

The Congruity Global Instability Engine (CGIE) is a research prototype for detecting and interpreting systemic instability patterns in global seismic activity.

The project combines spatial aggregation, probabilistic signal construction, and temporal coherence filtering to identify emerging instability regimes across the planetary seismic network.

---

## What this repository contains

This repository includes two complementary layers:

### 1. Public radar layer
A visual and exploratory interface showing global seismic instability patterns based on public earthquake data.

### 2. Baseline pipeline (reproducible)
A minimal, reproducible implementation of the core signal processing logic:

, daily aggregation of seismic activity  
, probabilistic instability signal  
, coherence-based filtering  
, alert generation  
, system-level interpretation  

The baseline pipeline is available in the `baseline/` folder and can be executed end-to-end.

---

## Overview

This project is designed as an exploratory observatory for complex system instability.

It does not attempt deterministic prediction of specific earthquake events, but instead focuses on:

, detection of elevated instability regimes  
, identification of spatial clustering patterns  
, temporal evolution of system pressure  
, interpretation of systemic transitions  

The approach is based on the idea that large-scale systems exhibit detectable structural deviations before major events.

---

## What the radar does

The radar aggregates public seismic events into spatial cells and computes a simplified instability state based on:

, seismic energy  
, event density  
, deviation from equilibrium  
, spatial clustering  
, dynamic phase evolution  

The output is a global radar map showing cells classified into instability phases.

---

## What this project is not

This project is not a deterministic earthquake prediction system.  
It is not an official warning system.  

It is a research framework for detecting and interpreting systemic instability patterns in complex geophysical systems.

---

## Baseline pipeline

See `baseline/README.md` for full details.

Pipeline:
events.csv → v48_alert_engine.py → v48_daily.csv + v48_alerts.csv → v49_final_report.py

### Pipeline flow

`events.csv`  
→ `v48_alert_engine.py`  
→ `v48_alerts.csv` and `v48_daily.csv`  
→ `v49_final_report.py`

This completes the first end to end prototype layer of the public system.

## License

MIT
