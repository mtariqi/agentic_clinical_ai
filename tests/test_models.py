from agentic_clinical.evaluation import evaluate


def test_surveillance_finds_injected_site_anomalies(study, model_outputs):
    _, truth = study
    scores, fc = model_outputs
    m = evaluate(truth, scores, fc, [])
    assert m["surveillance"]["any_site_anomaly"]["recall"] >= 0.8
    assert m["surveillance"]["any_site_anomaly"]["precision"] >= 0.8


def test_scores_are_bounded_and_explained(model_outputs):
    scores, _ = model_outputs
    s = scores.dropna(subset=["risk_score"])
    assert s["risk_score"].between(0, 1).all()
    # every flag names its KRI and carries evidence
    for flags in s["flags"]:
        for f in flags:
            assert {"kri", "value", "z", "description"} <= set(f)


def test_enrollment_detects_competitor_shock(study, model_outputs):
    _, truth = study
    _, fc = model_outputs
    flagged = set(fc.loc[fc["at_risk"] | fc["slowdown_detected"], "country"])
    assert set(truth["competitor_shock_countries"]) <= flagged


def test_forecast_intervals_are_ordered(model_outputs):
    _, fc = model_outputs
    assert (fc["projected_p10"] <= fc["projected_median"]).all()
    assert (fc["projected_median"] <= fc["projected_p90"]).all()
    assert fc["p_miss_target"].between(0, 1).all()
