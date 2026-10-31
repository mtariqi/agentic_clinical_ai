"""Site-level Key Risk Indicators (KRIs) for centralized monitoring.

Each KRI has a documented direction so detection logic knows which tail is
risky (e.g. *high* query rate is bad, *low* lab variance is suspicious).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# KRI name -> (direction, human-readable description)
#   direction "high": large values are risky; "low": small values are risky
KRI_DEFINITIONS: dict[str, tuple[str, str]] = {
    "query_rate": ("high", "Data queries per visit"),
    "missing_rate": ("high", "Missing CRF fields per visit"),
    "mean_entry_lag": ("high", "Mean days from visit to EDC entry"),
    "visit_window_dev": ("high", "Mean absolute deviation from visit schedule (days)"),
    "ae_rate": ("low", "Adverse events per subject (low may indicate under-reporting)"),
    "lab_variance_ratio": ("low", "Site lab SD relative to study SD (low may indicate fabrication)"),
}
MIN_SUBJECTS = 3  # below this a site is reported as "insufficient data", not scored


def build_site_kris(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sites = tables["sites"][["site_id", "country"]]
    visits, labs, aes, subjects = (tables[k] for k in ("visits", "labs", "adverse_events", "subjects"))

    v = visits.assign(window_dev=(visits["actual_day"] - visits["scheduled_day"]).abs())
    vk = v.groupby("site_id").agg(
        n_visits=("visit_no", "size"),
        query_rate=("n_queries", "mean"),
        missing_rate=("missing_fields", "mean"),
        mean_entry_lag=("entry_lag_days", "mean"),
        visit_window_dev=("window_dev", "mean"),
    )
    n_subj = subjects.groupby("site_id").size().rename("n_subjects")
    n_ae = aes.groupby("site_id").size().rename("n_ae")

    # lab variance ratio: per analyte, site SD / study SD; then average over analytes
    study_sd = labs.groupby("analyte")["value"].std()
    site_sd = labs.groupby(["site_id", "analyte"])["value"].std().unstack("analyte")
    lab_ratio = (site_sd / study_sd).mean(axis=1).rename("lab_variance_ratio")

    kri = (sites.set_index("site_id")
           .join([n_subj, vk, n_ae, lab_ratio])
           .fillna({"n_subjects": 0, "n_visits": 0, "n_ae": 0}))
    kri["n_subjects"] = kri["n_subjects"].astype(int)
    kri["n_visits"] = kri["n_visits"].astype(int)
    kri["ae_rate"] = np.where(kri["n_subjects"] > 0, kri["n_ae"] / kri["n_subjects"].clip(lower=1), np.nan)
    kri["sufficient_data"] = kri["n_subjects"] >= MIN_SUBJECTS
    cols = ["country", "n_subjects", "n_visits", "sufficient_data", *KRI_DEFINITIONS]
    return kri[cols].reset_index()
