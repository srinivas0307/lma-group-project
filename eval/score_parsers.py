"""Score every downloaded parser run in one go (CPU): contract metrics, verdicts, and the Stage 1 table.

    .venv/bin/python eval/score_parsers.py                       # all outputs/parse_*/ folders
    .venv/bin/python eval/score_parsers.py outputs/parse_qwen25-7b-instruct

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
BASE = [ROOT / "eval/pool_v2/verdicts_baseline1.jsonl", ROOT / "results/baseline2_v2/verdicts_baseline2.jsonl",
        ROOT / "results/intent_oracle_v2/verdicts_intent_oracle.jsonl"]
ORDER = ["intent_prompted_7b", "intent_prompted_3b", "intent_prompted_1_5b", "intent_ft_3b", "intent_ft_1_5b"]


def run(*cmd):
    p = subprocess.run([PY, *map(str, cmd)], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout


def main():
    dirs = [Path(a).resolve() for a in sys.argv[1:]] or sorted((ROOT / "outputs").glob("parse_*"))
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
        out = ROOT / "results" / f"{system}_v2"
        run("eval/run_intent_oracle.py", "--parses", d / "parses.jsonl", "--system", system, "--out", out)
        cm = json.loads((d / "contract_metrics.json").read_text())
        summary[system] = dict(folder=str(d), overall=cm["overall"], test=cm["by_split"].get("test"))
        systems.append(system)
    systems.sort(key=lambda s: ORDER.index(s) if s in ORDER else len(ORDER))
    verdicts = BASE + [ROOT / "results" / f"{s}_v2" / f"verdicts_{s}.jsonl" for s in systems]
    out = ROOT / "results" / "parsers_v2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "contract_summary.json").write_text(json.dumps(summary, indent=2))
    table = run("eval/metrics.py", "--pool", "eval/pool_v2/pool_v2.jsonl", "--verdicts", *verdicts,
                "--out", out / "metrics_stage1.json")
    keep = [l for l in table.splitlines() if l.startswith(("===", "system")) or " overall " in l or l.startswith("McNemar baseline2 vs")
            or ("McNemar intent_oracle vs" in l)]
    print("\n".join(keep))
    print(f"\nfull table: {out / 'metrics_stage1.json'}")


if __name__ == "__main__":
    main()
