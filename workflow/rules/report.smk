# ---- Evaluation and reporting -----------------------------------------------

rule evaluate:
    input:
        truth=f"{DATA}/ground_truth.json",
        scores=f"{INT}/site_scores.csv",
        forecast=f"{INT}/enrollment_forecast.csv",
        results=f"{AG}/agent_results.json",
        triage=f"{AG}/review_queue.json",
    output:
        f"{REP}/metrics.json",
    log:
        f"{LOGS}/evaluate.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} evaluate --truth {input.truth} --scores {input.scores} --forecast {input.forecast} "
        "--results {input.results} --triage {input.triage} --out {output} > {log} 2>&1"


rule report:
    input:
        truth=f"{DATA}/ground_truth.json",
        scores=f"{INT}/site_scores.csv",
        forecast=f"{INT}/enrollment_forecast.csv",
        results=f"{AG}/agent_results.json",
        triage=f"{AG}/review_queue.json",
        audit=f"{AG}/audit_verification.json",
        metrics=f"{REP}/metrics.json",
        validation=f"{INT}/validation.json",
        policy=config["agents"]["policy_file"],
    output:
        f"{REP}/report.html",
    log:
        f"{LOGS}/report.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} report --config {RUNCFG} --policy {input.policy} --truth {input.truth} "
        "--scores {input.scores} --forecast {input.forecast} --results {input.results} "
        "--triage {input.triage} --audit {input.audit} --metrics {input.metrics} "
        "--validation {input.validation} --out {output} > {log} 2>&1"
