import pandas as pd

from agentic_clinical.data.quality import validate_tables
from agentic_clinical.data.synthetic import StudyConfig, generate_study, write_study


def test_generation_is_deterministic():
    a, ta = generate_study(StudyConfig(), seed=3)
    b, tb = generate_study(StudyConfig(), seed=3)
    assert ta == tb
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_injected_anomaly_sets_are_disjoint(study):
    _, truth = study
    groups = ["fabrication_sites", "high_query_sites", "late_entry_sites", "prompt_injection_sites"]
    seen = set()
    for g in groups:
        assert not seen & set(truth[g])
        seen |= set(truth[g])


def test_true_rates_are_hidden_from_models(study):
    tables, _ = study
    assert "base_rate" not in tables["sites"].columns


def test_validation_passes_on_clean_data(tmp_path, study):
    write_study(*study, tmp_path)
    rep = validate_tables(tmp_path)
    assert rep.passed, rep.errors
    assert all(len(v["sha256"]) == 64 for v in rep.lineage.values())


def test_validation_catches_orphans_and_ranges(tmp_path, study):
    tables, truth = study
    bad = {k: v.copy() for k, v in tables.items()}
    bad["visits"].loc[0, "site_id"] = "XX-999"
    bad["enrollment_weekly"].loc[0, "new_subjects"] = -5
    write_study(bad, truth, tmp_path)
    rep = validate_tables(tmp_path)
    assert not rep.passed
    joined = " ".join(rep.errors)
    assert "unknown site_id" in joined and "outside" in joined


def test_validation_catches_missing_file(tmp_path, study):
    write_study(*study, tmp_path)
    (tmp_path / "labs.csv").unlink()
    assert not validate_tables(tmp_path).passed
