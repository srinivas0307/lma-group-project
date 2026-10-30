"""Score every downloaded parser run in one go (CPU): contract metrics, verdicts, and the Stage 1 table.

    .venv/bin/python eval/score_parsers.py                       # all outputs/parse_*/ folders
    .venv/bin/python eval/score_parsers.py outputs/parse_qwen25-7b-instruct
    .venv/bin/python eval/score_parsers.py --pool-version v3         # same parses, judged on pool v3

Each parser folder needs parses.jsonl and system.txt (written by notebooks/baseline3_parse_kaggle.ipynb), e.g.
`intent_prompted_7b`. Writes:
    <folder>/contract_metrics.json
    results/<system>_v2/verdicts_<system>[_change_only|_preserve_only].jsonl
    results/parsers_v2/metrics_stage1.json and contract_summary.json  (all systems side by side)
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
def base_verdicts(v):
    return [ROOT / f"eval/pool_{v}/verdicts_baseline1.jsonl", ROOT / f"results/baseline2_{v}/verdicts_baseline2.jsonl",
            ROOT / f"results/intent_oracle_{v}/verdicts_intent_oracle.jsonl"]
ORDER = [f"intent_prompted_{m}{v}" for v in ("", "_p1v2", "_p1v3") for m in ("7b", "3b", "1_5b")] + \
    ["intent_ft_3b", "intent_ft_1_5b"]
VERSION_SUFFIX = {"": "p1-v1", "_p1v2": "p1-v2", "_p1v3": "p1-v3"}


def select_on_dev(summary):
    """Baseline 3 prompt selection rule (decided 2026-10-30, before the p1-v3 run): per prompted model, pick the prompt
    version with the lowest DEV invalid-output rate; ties -> higher DEV mean of change/preserve F1 -> earlier version.
    Test results are never used to choose."""
    groups = {}
    for system, info in summary.items():
        if not system.startswith("intent_prompted_"):
            continue
        base, suffix = system, ""
        for suf in ("_p1v2", "_p1v3"):
            if system.endswith(suf):
                base, suffix = system[: -len(suf)], suf
        cm = json.loads((Path(info["folder"]) / "contract_metrics.json").read_text())["by_split"]["dev"]
        f1 = [cm[s]["f1"] or 0.0 for s in ("must_change", "must_preserve")]
        groups.setdefault(base, []).append(dict(system=system, version=VERSION_SUFFIX[suffix],
                                                dev_invalid=cm["invalid_rate"], dev_mean_f1=round(sum(f1) / 2, 3)))
    out = {}
    for base, cands in groups.items():
        order = ["p1-v1", "p1-v2", "p1-v3"]
        best = min(cands, key=lambda c: (c["dev_invalid"], -c["dev_mean_f1"], order.index(c["version"])))
        out[base] = dict(selected=best["system"], version=best["version"], candidates=cands)
    return out


def run(*cmd):
    p = subprocess.run([PY, *map(str, cmd)], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout


def main():
    argv = sys.argv[1:]
    pv = "v2"
    if "--pool-version" in argv:
        i = argv.index("--pool-version")
        pv = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    dirs = [Path(a).resolve() for a in argv] or sorted((ROOT / "outputs").glob("parse_*"))
    dirs = [d for d in dirs if (d / "parses.jsonl").exists()]
    if not dirs:
        sys.exit("no parser folders with parses.jsonl under outputs/ (unzip the Kaggle outputs to outputs/parse_<model>/)")
    systems, summary = [], {}
    for d in dirs:
        st = d / "system.txt"
        if not st.exists():
            sys.exit(f"{d}: missing system.txt (one line, e.g. intent_prompted_7b)")
        system = st.read_text().strip()
        print(f"== {system}  ({d.name})")
        print(run("eval/contract_metrics.py", "--parses", d / "parses.jsonl"))
        out = ROOT / "results" / f"{system}_{pv}"
        run("eval/run_intent_oracle.py", "--parses", d / "parses.jsonl", "--system", system, "--out", out,
            "--pool", ROOT / f"eval/pool_{pv}/pool_{pv}.jsonl")
        cm = json.loads((d / "contract_metrics.json").read_text())
        summary[system] = dict(folder=str(d), overall=cm["overall"], test=cm["by_split"].get("test"))
        systems.append(system)
    systems.sort(key=lambda s: ORDER.index(s) if s in ORDER else len(ORDER))
    verdicts = base_verdicts(pv) + [ROOT / "results" / f"{s}_{pv}" / f"verdicts_{s}.jsonl" for s in systems]
    out = ROOT / "results" / f"parsers_{pv}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "contract_summary.json").write_text(json.dumps(summary, indent=2))
    selection = select_on_dev(summary)
    (out / "baseline3_selection.json").write_text(json.dumps(selection, indent=2))
    table = run("eval/metrics.py", "--pool", f"eval/pool_{pv}/pool_{pv}.jsonl", "--verdicts", *verdicts,
                "--out", out / "metrics_stage1.json")
    keep = [l for l in table.splitlines() if l.startswith(("===", "system")) or " overall " in l or l.startswith("McNemar baseline2 vs")
            or ("McNemar intent_oracle vs" in l)]
    print("\n".join(keep))
    print("\nPrompt selection on DEV only (lowest invalid rate, then F1):")
    for base, sel in selection.items():
        print(f"  {base:<22} -> {sel['selected']:<28} " +
              " | ".join(f"{c['version']}: invalid {c['dev_invalid']:.3f}, F1 {c['dev_mean_f1']:.3f}" for c in sel["candidates"]))
    print(f"\nfull table: {out / 'metrics_stage1.json'}")


if __name__ == "__main__":
    main()
