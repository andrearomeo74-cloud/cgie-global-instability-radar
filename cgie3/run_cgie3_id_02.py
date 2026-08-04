import pandas as pd
import numpy as np

# ==========================
# CARICA RETE BASELINE
# ==========================

baseline = pd.read_csv("outputs/CGIE3_identity_edges.csv")

baseline["edge"] = (
    baseline["source"] + "->" + baseline["target"]
)

baseline = baseline.set_index("edge")

# ==========================
# CARICA RETE CORRENTE
# ==========================

current = pd.read_csv("outputs/CGIE3_identity_edges_current.csv")

current["edge"] = (
    current["source"] + "->" + current["target"]
)

current = current.set_index("edge")

# ==========================
# ARCHI COMUNI
# ==========================

common = baseline.index.intersection(current.index)

b = baseline.loc[common]["weight"]
c = current.loc[common]["weight"]

# ==========================
# IEC
# ==========================

delta = np.abs(b - c)

IEC = 1 - delta.mean()

IEC = max(0.0, min(1.0, IEC))

# ==========================
# INC
# ==========================

INC = len(common) / len(baseline)

# ==========================
# CRM-ID
# ==========================

CRM_ID = IEC * INC

# ==========================
# APS-ID
# ==========================

APS_ID = 1 - CRM_ID

# ==========================
# MFAC-ID
# ==========================

MFAC_ID = np.sqrt(APS_ID)

print()

print("IEC      :", round(IEC,4))
print("INC      :", round(INC,4))
print("CRM-ID   :", round(CRM_ID,4))
print("APS-ID   :", round(APS_ID,4))
print("MFAC-ID  :", round(MFAC_ID,4))

pd.DataFrame([{
    "IEC":IEC,
    "INC":INC,
    "CRM_ID":CRM_ID,
    "APS_ID":APS_ID,
    "MFAC_ID":MFAC_ID
}]).to_csv(
    "outputs/CGIE3_identity_metrics.csv",
    index=False
)
