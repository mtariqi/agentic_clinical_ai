# ---- Agents: governed autonomy, oversight triage, audit verification ---------

rule run_agents:
    input:
        scores=f"{INT}/site_scores.csv",
        forecast=f"{INT}/enrollment_forecast.csv",
        policy=config["agents"]["policy_file"],
    output:
        results=f"{AG}/agent_results.json",
        audit=f"{AG}/audit_log.jsonl",
        outbox=f"{AG}/outbox.jsonl",
        approvals=f"{AG}/approvals.jsonl",
    params:
        datadir=DATA,
        outdir=AG,
        memory=MEMORY,
    log:
        f"{LOGS}/run_agents.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} agents --config {RUNCFG} --policy {input.policy} "
        "--datadir {params.datadir} --scores {input.scores} --forecast {input.forecast} "
        "--memory {params.memory} --outdir {params.outdir} > {log} 2>&1"


rule oversight_triage:
    """Meta-agent: consolidate signals into a short, ranked human review queue."""
    input:
        f"{AG}/agent_results.json",
    output:
        f"{AG}/review_queue.json",
    log:
        f"{LOGS}/oversight_triage.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} oversight --config {RUNCFG} --results {input} --out {output} > {log} 2>&1"


rule verify_audit:
    """Fails the run if the hash-chained audit trail has been altered."""
    input:
        f"{AG}/audit_log.jsonl",
    output:
        f"{AG}/audit_verification.json",
    log:
        f"{LOGS}/verify_audit.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} verify-audit --log {input} --out {output} > {log} 2>&1"
