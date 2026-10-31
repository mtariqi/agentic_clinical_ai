"""Clinical Data Surveillance: explainable, multivariate site-risk scoring.

Small sites produce noisy KRIs, so each z-score is shrunk toward zero with a
credibility weight n / (n + k) (n = visits at the site), an empirical-Bayes
style correction that stops tiny sites from dominating the alert list.

Two complementary detectors:
  1. Direction-aware robust z-scores (median / MAD) per KRI -> *explainable*
     flags naming exactly which indicator breached and by how much.
  2. Isolation Forest over all KRIs -> catches unusual *combinations* that no
     single threshold would flag.

The combined risk score is in [0, 1] and every flag carries its evidence, so a
human reviewer (or the oversight meta-agent) can see why a site was raised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from agentic_clinical.features import KRI_DEFINITIONS

MAD_TO_SD = 1.4826


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * MAD_TO_SD
    if not np.isfinite(mad) or mad == 0:
        sd = x.std()
        mad = sd if sd and np.isfinite(sd) else 1.0
    return (x - med) / mad


def score_sites(kri: pd.DataFrame, z_threshold: float = 3.0, n_estimators: int = 300,
                contamination: float = 0.08, seed: int = 42,
                shrinkage_k: float = 10.0) -> pd.DataFrame:
    """Return one row per site with KRI z-scores, flags, and a combined risk score."""
    out = kri.copy()
    scored = out[out["sufficient_data"]].copy()
    kri_cols = list(KRI_DEFINITIONS)

    # 1) direction-aware robust z (positive = risky direction), credibility-weighted
    weight = scored["n_visits"] / (scored["n_visits"] + shrinkage_k)
    for col, (direction, _) in KRI_DEFINITIONS.items():
        z = robust_z(scored[col].astype(float)) * weight
        scored[f"z_{col}"] = z if direction == "high" else -z

    zcols = [f"z_{c}" for c in kri_cols]
    scored["max_z"] = scored[zcols].max(axis=1)

    def _flags(row: pd.Series) -> list[dict]:
        flags = []
        for col in kri_cols:
            z = row[f"z_{col}"]
            if z >= z_threshold:
                flags.append({"kri": col, "value": round(float(row[col]), 3),
                              "z": round(float(z), 2), "description": KRI_DEFINITIONS[col][1]})
        return sorted(flags, key=lambda f: -f["z"])

    scored["flags"] = scored.apply(_flags, axis=1)

    # 2) multivariate isolation forest
    X = RobustScaler().fit_transform(scored[kri_cols].fillna(scored[kri_cols].median()))
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=seed)
    iso.fit(X)
    raw = -iso.score_samples(X)  # higher = more anomalous
    scored["if_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    scored["if_outlier"] = iso.predict(X) == -1

    # 3) combined risk in [0, 1]: saturating transform of max z, blended with IF
    z_component = 1 - np.exp(-np.clip(scored["max_z"], 0, None) / z_threshold)
    scored["risk_score"] = (0.7 * z_component + 0.3 * scored["if_score"]).round(4)
    scored["risk_tier"] = pd.cut(scored["risk_score"], [-0.01, 0.4, 0.6, 1.01],
                                 labels=["low", "medium", "high"]).astype(str)

    keep = ["site_id", *zcols, "max_z", "flags", "if_score", "if_outlier", "risk_score", "risk_tier"]
    out = out.merge(scored[keep], on="site_id", how="left")
    out["flags"] = out["flags"].apply(lambda f: f if isinstance(f, list) else [])
    out["risk_tier"] = out["risk_tier"].fillna("insufficient_data")
    return out.sort_values("risk_score", ascending=False, na_position="last").reset_index(drop=True)
