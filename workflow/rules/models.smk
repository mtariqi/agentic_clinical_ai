# ---- Data science: KRIs, surveillance scoring, enrollment forecast -----------

rule site_kris:
    input:
        validation=f"{INT}/validation.json",   # gate: only runs on validated data
    output:
        f"{INT}/site_kris.csv",
    params:
        datadir=DATA,
    log:
        f"{LOGS}/site_kris.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} features --datadir {params.datadir} --out {output} > {log} 2>&1"


rule surveillance_scores:
    input:
        kri=f"{INT}/site_kris.csv",
    output:
        f"{INT}/site_scores.csv",
    log:
        f"{LOGS}/surveillance_scores.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} surveillance --config {RUNCFG} --kri {input.kri} --out {output} > {log} 2>&1"


rule enrollment_forecast:
    input:
        validation=f"{INT}/validation.json",
    output:
        f"{INT}/enrollment_forecast.csv",
    params:
        datadir=DATA,
    log:
        f"{LOGS}/enrollment_forecast.log",
    conda:
        "../../envs/environment.yml"
    shell:
        "{CLI} forecast --config {RUNCFG} --datadir {params.datadir} --out {output} > {log} 2>&1"
