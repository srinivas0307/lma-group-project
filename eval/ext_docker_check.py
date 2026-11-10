"""Extension #5 (docker): compare the Docker sandbox with the subprocess sandbox (CPU; needs Docker and the image
`docker build -t igt-sandbox:docker1 sandbox/docker`).

    .venv/bin/python eval/ext_docker_check.py                 # red-team set + 40 pool candidates
    .venv/bin/python eval/ext_docker_check.py --pool-sample 0 # red-team set only

1. Red-team set (benchmark/redteam/samples.yaml) through docker_safe_run, and with the guard bypassed (docker_run) for
   samples with bypass_check: each must be blocked or contained (same rule as eval/safety_table.py). Extra probes
   only the container can stop with the guard bypassed: reading a host file, and opening a network connection.
2. Equivalence: the reference solutions of a sample of tasks and a sample of pool candidates run in both sandboxes;
   status and output (eval/equivalence.outputs_equal) must be the same.
Writes results/safety_ext/docker_check.json and .md. results/safety/ (the main safety table) is not touched.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.equivalence import outputs_equal  # noqa: E402
from eval.safety_table import DF, LIMITS, ok_for  # noqa: E402
from igt.paths import POOL, REDTEAM, RESULTS, TASKS  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402
from igt.sandbox.docker_executor import IMAGE, docker_available, docker_run, docker_safe_run  # noqa: E402

OUT = RESULTS / "safety_ext"
PROBES = [
    dict(id="probe_host_file", code='def transform(df):\n    df["x"] = open("%s").read()[:10]\n    return df\n'
         % (ROOT / "README.md"), ok_status={"error"}),
    dict(id="probe_network", code=('def transform(df):\n    import _socket   # the C module: bypasses the socket stub of the '
                                   'child script\n    s = _socket.socket()\n    s.settimeout(3)\n    s.connect(("1.1.1.1", 53))\n'
                                   '    return df\n'), ok_status={"error"}),
    dict(id="probe_write_root", code='def transform(df):\n    open("/pwned", "w").write("x")\n    return df\n',
         ok_status={"error"}),
]


def same(a, b):
    if a.status != b.status:
        return False
    if a.ran_ok and b.ran_ok:
        return bool(outputs_equal(a.df_out, b.df_out)[0])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-sample", type=int, default=40)
    ap.add_argument("--task-sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not docker_available():
        sys.exit(f"Docker or the image {IMAGE} is missing: docker build -t {IMAGE} sandbox/docker")
    rep = dict(image=IMAGE, redteam=[], probes=[], equivalence=[])
    for s in yaml.safe_load((REDTEAM / "samples.yaml").read_text())["samples"]:
        r = docker_safe_run(s["code"], DF, **LIMITS)
        row = dict(id=s["id"], expect=s["expect"], status=r.status, ok=ok_for(s, r.status, r.df_out))
        if s.get("bypass_check"):
            b = docker_run(s["code"], DF, **LIMITS)
            row.update(bypass_status=b.status, bypass_ok=b.status in s["contained_as"] and b.df_out is None)
        rep["redteam"].append(row)
        print(f"{s['id']:<32} {r.status:<8} {row.get('bypass_status', ''):<8} {'ok' if row['ok'] and row.get('bypass_ok', True) else 'FAIL'}",
              flush=True)
    for p in PROBES:                                 # the guard would block these; bypass it to test the container
        d, sub = docker_run(p["code"], DF, **LIMITS), None
        from igt.sandbox import run as sub_run
        sub = sub_run(p["code"], DF, **LIMITS)
        rep["probes"].append(dict(id=p["id"], docker=d.status, docker_error=(d.error or "")[:200],
                                  subprocess=sub.status, subprocess_error=(sub.error or "")[:200],
                                  ok=d.status in p["ok_status"] and d.df_out is None))
        print(f"{p['id']:<32} docker={d.status} ({(d.error or '')[:60]}) | subprocess={sub.status}", flush=True)
    rng = random.Random(args.seed)
    tasks = [yaml.safe_load(f.read_text()) for f in sorted(TASKS.glob("*/*.yaml"))]
    pool = [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()]
    data = {}
    items = [(f"ref:{t['id']}", t["reference_code"], t["data_file"]) for t in rng.sample(tasks, min(args.task_sample, len(tasks)))]
    tfile = {t["id"]: t["data_file"] for t in tasks}
    items += [(f"pool:{c['candidate_id']}", c["code"], tfile[c["task_id"]])
              for c in rng.sample(pool, min(args.pool_sample, len(pool))) if c.get("code")]
    for name, code, f in items:
        df = data.setdefault(f, pd.read_csv(ROOT / f))
        t0 = time.time()
        a, b = safe_run(code, df, **LIMITS), docker_safe_run(code, df, **LIMITS)
        rep["equivalence"].append(dict(item=name, subprocess=a.status, docker=b.status, same=same(a, b),
                                       seconds=round(time.time() - t0, 1)))
        print(f"{name:<40} {a.status:<8} {b.status:<8} {'same' if same(a, b) else 'DIFFERENT'}", flush=True)
    rt_ok = sum(r["ok"] and r.get("bypass_ok", True) for r in rep["redteam"])
    rep["summary"] = dict(redteam_passed=rt_ok, redteam_n=len(rep["redteam"]),
                          probes_contained=sum(p["ok"] for p in rep["probes"]), probes_n=len(rep["probes"]),
                          equivalent=sum(e["same"] for e in rep["equivalence"]), equivalence_n=len(rep["equivalence"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "docker_check.json").write_text(json.dumps(rep, indent=2))
    s = rep["summary"]
    md = ["# Docker sandbox check (extension #5)", "", f"Image `{IMAGE}`. Generated by `eval/ext_docker_check.py`.", "",
          f"- Red-team set: {s['redteam_passed']}/{s['redteam_n']} blocked or contained (guard on; guard bypassed where set).",
          f"- Probes with the guard bypassed (host file, network, write to /): {s['probes_contained']}/{s['probes_n']} "
          "stopped by the container.",
          f"- Same status and output as the subprocess sandbox: {s['equivalent']}/{s['equivalence_n']} "
          "(reference solutions + pool candidates).", "", "| Probe | Docker | Subprocess sandbox |", "|---|---|---|"]
    md += [f"| {p['id']} | {p['docker']}: {p['docker_error'][:80]} | {p['subprocess']}: {p['subprocess_error'][:80]} |"
           for p in rep["probes"]]
    md += ["", "Different items: " + (", ".join(e["item"] for e in rep["equivalence"] if not e["same"]) or "none")]
    (OUT / "docker_check.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md[4:7]))
    return 0 if rt_ok == s["redteam_n"] and s["equivalent"] == s["equivalence_n"] else 1


if __name__ == "__main__":
    sys.exit(main())
