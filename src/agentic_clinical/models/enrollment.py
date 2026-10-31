"""Predictive Enrollment Intelligence.

For each country:
  * Bayesian Poisson-Gamma estimate of the *recent* weekly enrollment rate
    (Jeffreys-style prior), so short windows carry honest uncertainty.
  * Monte Carlo projection of final enrollment -> probability of missing target
    and expected shortfall, with an 80% prediction interval.
  * Rate-change test (conditional binomial test of recent vs. earlier counts)
    to separate a genuine slowdown from noise - the early-warning signal that
    surfaces competitor pressure before cumulative reports show it.
  * Headroom for countries projected above target -> reallocation candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def forecast_enrollment(enrollment: pd.DataFrame, sites: pd.DataFrame, current_week: int,
                        study_weeks: int, recent_window: int = 6, n_sims: int = 5000,
                        shortfall_threshold: float = 0.15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    remaining = study_weeks - current_week
    targets = sites.groupby("country")["planned_target"].sum()
    rows = []
    for country, g in enrollment.groupby("country"):
        weekly = g.groupby("week")["new_subjects"].sum().sort_index()
        enrolled = int(weekly.sum())
        recent = weekly[weekly.index >= current_week - recent_window]
        earlier = weekly[weekly.index < current_week - recent_window]
        w_recent = max(len(recent), 1)
        k_recent = int(recent.sum())

        # posterior: rate ~ Gamma(k + 0.5, w)  -> simulate final enrollment
        lam = rng.gamma(shape=k_recent + 0.5, scale=1.0 / w_recent, size=n_sims)
        final = enrolled + rng.poisson(lam * remaining)
        target = int(targets.get(country, 0))
        p_miss = float((final < target).mean()) if target else 0.0
        median_final = float(np.median(final))
        shortfall = max(0.0, (target - median_final) / target) if target else 0.0

        # rate-change test: are recent counts lower than the earlier rate predicts?
        rate_earlier = earlier.sum() / max(len(earlier), 1)
        rate_recent = k_recent / w_recent
        p_change = 1.0
        if len(earlier) and (k_recent + earlier.sum()) > 0:
            frac = len(recent) / (len(recent) + len(earlier))
            p_change = binomtest(k_recent, int(k_recent + earlier.sum()), frac,
                                 alternative="less").pvalue

        rows.append({
            "country": country,
            "target": target,
            "enrolled_to_date": enrolled,
            "rate_earlier": round(float(rate_earlier), 3),
            "rate_recent": round(float(rate_recent), 3),
            "rate_ratio": round(float(rate_recent / rate_earlier), 3) if rate_earlier else np.nan,
            "rate_drop_pvalue": float(p_change),
            "projected_median": median_final,
            "projected_p10": float(np.percentile(final, 10)),
            "projected_p90": float(np.percentile(final, 90)),
            "p_miss_target": round(p_miss, 4),
            "projected_shortfall": round(shortfall, 4),
            "headroom": round(median_final - target, 1),
        })
    df = pd.DataFrame(rows)
    df["at_risk"] = (df["projected_shortfall"] >= shortfall_threshold) & (df["p_miss_target"] >= 0.8)
    df["slowdown_detected"] = (df["rate_drop_pvalue"] < 0.01) & (df["rate_ratio"] < 0.7)
    return df.sort_values("projected_shortfall", ascending=False).reset_index(drop=True)
