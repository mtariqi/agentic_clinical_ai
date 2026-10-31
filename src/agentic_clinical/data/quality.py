"""Schema validation, referential integrity and lineage for input data.

"Start with clarity before you start coding": agents only run on data that has
passed explicit, codified checks. Each input file is fingerprinted (SHA-256)
so every downstream decision can be traced back to the exact data it used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# table -> {column: pandas dtype kind}  (i = integer, f = float, O = object/str, b = bool)
SCHEMAS: dict[str, dict[str, str]] = {
    "sites": {"site_id": "O", "country": "O", "activation_week": "i", "planned_target": "i"},
    "enrollment_weekly": {"site_id": "O", "country": "O", "week": "i", "new_subjects": "i"},
    "subjects": {"subject_id": "O", "site_id": "O", "country": "O", "enroll_week": "i"},
    "visits": {"subject_id": "O", "site_id": "O", "visit_no": "i", "scheduled_day": "i",
               "actual_day": "i", "entry_lag_days": "f", "missing_fields": "i", "n_queries": "i"},
    "labs": {"subject_id": "O", "site_id": "O", "visit_no": "i", "analyte": "O", "value": "f"},
    "adverse_events": {"subject_id": "O", "site_id": "O", "ae_term": "O", "serious": "b"},
    "site_comments": {"site_id": "O", "week": "i", "comment": "O"},
}
PRIMARY_KEYS = {
    "sites": ["site_id"],
    "enrollment_weekly": ["site_id", "week"],
    "subjects": ["subject_id"],
    "visits": ["subject_id", "visit_no"],
    "labs": ["subject_id", "visit_no", "analyte"],
}
RANGES = {  # (table, column): (min, max)
    ("enrollment_weekly", "new_subjects"): (0, 100),
    ("visits", "entry_lag_days"): (0, 365),
    ("visits", "missing_fields"): (0, 200),
    ("visits", "n_queries"): (0, 200),
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lineage: dict[str, dict] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"passed": self.passed, "errors": self.errors,
                "warnings": self.warnings, "lineage": self.lineage}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _kind_ok(series: pd.Series, kind: str) -> bool:
    actual = series.dtype.kind
    if kind == "f":
        return actual in "fi"  # integers are acceptable where floats are expected
    if kind == "O":
        return actual in "OSU" or str(series.dtype).startswith("string")
    return actual == kind


def load_tables(datadir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(datadir)
    return {name: pd.read_csv(d / f"{name}.csv") for name in SCHEMAS}


def validate_tables(datadir: str | Path) -> ValidationReport:
    d = Path(datadir)
    rep = ValidationReport()
    tables: dict[str, pd.DataFrame] = {}

    for name, schema in SCHEMAS.items():
        path = d / f"{name}.csv"
        if not path.exists():
            rep.errors.append(f"{name}: file missing ({path})")
            continue
        df = pd.read_csv(path)
        tables[name] = df
        rep.lineage[name] = {"file": str(path), "sha256": sha256_file(path), "rows": len(df)}
        missing = set(schema) - set(df.columns)
        if missing:
            rep.errors.append(f"{name}: missing columns {sorted(missing)}")
            continue
        for col, kind in schema.items():
            if len(df) and not _kind_ok(df[col], kind):
                rep.errors.append(f"{name}.{col}: expected dtype kind '{kind}', got {df[col].dtype}")
            nulls = int(df[col].isna().sum())
            if nulls:
                rep.errors.append(f"{name}.{col}: {nulls} null values")
        pk = PRIMARY_KEYS.get(name)
        if pk and df.duplicated(pk).any():
            rep.errors.append(f"{name}: duplicate primary key {pk} ({int(df.duplicated(pk).sum())} rows)")

    for (name, col), (lo, hi) in RANGES.items():
        if name in tables and col in tables[name]:
            bad = ~tables[name][col].between(lo, hi)
            if bad.any():
                rep.errors.append(f"{name}.{col}: {int(bad.sum())} values outside [{lo}, {hi}]")

    # referential integrity
    if "sites" in tables:
        known_sites = set(tables["sites"]["site_id"])
        for name in ("enrollment_weekly", "subjects", "visits", "labs", "adverse_events",
                     "site_comments"):
            if name in tables and "site_id" in tables[name]:
                orphans = set(tables[name]["site_id"]) - known_sites
                if orphans:
                    rep.errors.append(f"{name}: {len(orphans)} unknown site_id values")
    if "subjects" in tables:
        known_subj = set(tables["subjects"]["subject_id"])
        for name in ("visits", "labs", "adverse_events"):
            if name in tables:
                orphans = set(tables[name]["subject_id"]) - known_subj
                if orphans:
                    rep.errors.append(f"{name}: {len(orphans)} unknown subject_id values")

    # soft checks
    if "visits" in tables and len(tables["visits"]):
        late = (tables["visits"]["actual_day"] < tables["visits"]["scheduled_day"] - 14).mean()
        if late > 0.05:
            rep.warnings.append(f"visits: {late:.1%} of visits occur >14 days before schedule")
    if "site_comments" in tables:
        rep.warnings.append("site_comments: free text is UNTRUSTED input and is screened by guardrails")
    return rep
