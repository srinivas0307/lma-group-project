"""Score Stage 1 extension systems (CPU) without touching results/stage1/summary/ (eval/score_parsers.py rewrites that
folder, so it is never run on extension folders).

    .venv/bin/python eval/ext_score.py results/stage1/extensions/intent_prompted_7b_p1v3_pc1   # one system
    .venv/bin/python eval/ext_score.py                                                       # every extension system

An extension folder holds parses.jsonl + system.txt (from eval/ext_precheck.py, eval/ext_reask.py or
eval/parse_contracts_tools_ft.py) or only verdicts_<system>.jsonl (eval/run_llm_verifier.py). For a parse folder this
writes contract_metrics.json and verdicts_<system>[_change_only|_preserve_only].jsonl into the same folder. Then all
extension systems are scored side by side with Baseline 2, Baseline 3 and the oracle into
results/stage1/extensions/summary/metrics_extensions.json (+ .txt, the printed table).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ext_common import EXT_STAGE1, EXT_SUMMARY, ROOT  # noqa: E402
from igt.paths import POOL, rel, stage1_dir  # noqa: E402

PY = sys.executable
BASE = [stage1_dir("baseline2") / "verdicts_baseline2.jsonl",
        stage1_dir("intent_prompted_7b_p1v3") / "verdicts_intent_prompted_7b_p1v3.jsonl",
        stage1_dir("intent_oracle") / "verdicts_intent_oracle.jsonl"]


def run(*cmd):
    p = subprocess.run([PY, *map(str, cmd)], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout


def score_folder(d):
    d = Path(d).resolve()
    if not str(d).startswith(str(EXT_STAGE1.resolve())):
        sys.exit(f"{d}: not under {rel(EXT_STAGE1)}/ (existing result folders are scored by eval/score_parsers.py)")
    system = (d / "system.txt").read_text().strip() if (d / "system.txt").exists() else d.name
    if (d / "parses.jsonl").exists():
        print(run("eval/contract_metrics.py", "--parses", d / "parses.jsonl", "--out", d / "contract_metrics.json")
              .splitlines()[1])
        run("eval/run_intent_oracle.py", "--parses", d / "parses.jsonl", "--system", system, "--out", d, "--pool", POOL)
    return system, d / f"verdicts_{system}.jsonl"


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in EXT_STAGE1.glob("*") if p.is_dir() and p.name != "summary")
    for d in dirs:
        score_folder(d)
    verdicts = []
    for d in sorted(p for p in EXT_STAGE1.glob("*") if p.is_dir() and p.name != "summary"):
        system = (d / "system.txt").read_text().strip() if (d / "system.txt").exists() else d.name
        if (d / f"verdicts_{system}.jsonl").exists():
            verdicts.append(d / f"verdicts_{system}.jsonl")
    if not verdicts:
        sys.exit(f"no extension verdicts under {rel(EXT_STAGE1)}/")
    EXT_SUMMARY.mkdir(parents=True, exist_ok=True)
    table = run("eval/metrics.py", "--pool", POOL, "--verdicts", *BASE, *verdicts,
                "--out", EXT_SUMMARY / "metrics_extensions.json")
    (EXT_SUMMARY / "metrics_extensions.txt").write_text(table)
    keep = [l for l in table.splitlines() if l.startswith(("===", "system")) or "overall " in l]
    print("\n".join(keep))
    print(f"\nfull table: {rel(EXT_SUMMARY / 'metrics_extensions.json')}")


if __name__ == "__main__":
    main()
