# Baseline pipeline

This folder contains the first reproducible baseline implementation of the Congruity Global Instability Engine (CGIE).

## Overview

The baseline pipeline transforms raw seismic event data into:

, structured daily signals  
, filtered alert events  
, human readable system status  

It represents the first operational layer of the public prototype.
This baseline provides a minimal, reproducible reference implementation of the public CGIE layer.

---

## Modules

### v48_alert_engine.py

Core processing engine.

Functions:

, loads raw seismic events  
, builds daily aggregated dataset  
, computes probabilistic instability signal  
, applies coherence and temporal filters  
, generates alert events  

Outputs:

, `v48_daily.csv`  
, `v48_alerts.csv`

---

### v49_final_report.py

Reporting layer.

Functions:

, reads V48 outputs  
, computes system status  
, evaluates short term trend  
, summarizes alert activity  
, produces human readable interpretation  

---

## Pipeline flow

events.csv → v48_alert_engine.py → v48_daily.csv + v48_alerts.csv → v49_final_report.py

---

## Usage (Colab or local)

Run:
python v48_alert_engine.py python v49_final_report.py
Make sure `events.csv` is present in the working directory.

---

## Notes

This baseline:

, is not a deterministic earthquake prediction system  
, is not an official warning system  
, is a research prototype for instability detection  

The deeper Congruity framework and advanced layers are not included in this public repository.
