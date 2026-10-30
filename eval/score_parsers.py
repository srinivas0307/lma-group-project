"""Score every downloaded parser run in one go (CPU): contract metrics, verdicts, and the Stage 1 table (pool v3).

    .venv/bin/python eval/score_parsers.py                                   # all kaggle_outputs/parsing/*/ folders
    .venv/bin/python eval/score_parsers.py kaggle_outputs/parsing/7b_p1-v3    # one run

Each parser folder needs parses.jsonl and system.txt (written by notebooks/kaggle/baseline3_parse_kaggle.ipynb), e.g.
`intent_prompted_7b_p1v3`. Unzip a new Kaggle run to kaggle_outputs/parsing/<model>_<prompt version>/. Writes, per
system, into its Stage 1 results folder (igt.paths.stage1_dir: baseline3/, size_ablation/<system>/,
other_prompts/<system>/ or finetuned/<system>/):
    contract_metrics.json
    verdicts_<system>[_change_only|_preserve_only].jsonl
and for all systems side by side, in results/stage1/summary/:
    metrics_stage1.json, contract_summary.json, baseline3_selection.json
Pool v1/v2 runs are in archive/ and are not rescored.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.paths import PARSING, POOL, STAGE1_SUMMARY, rel, stage1_dir  # noqa: E402

PY = sys.executable
BASE_VERDICTS = [stage1_dir("baseline1") / "verdicts_baseline1.jsonl",
                 stage1_dir("baseline2") / "verdicts_baseline2.jsonl",
                 stage1_dir("intent_oracle") / "verdicts_intent_oracle.jsonl"]
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
        cm = json.loads((ROOT / info["contract_metrics"]).read_text())["by_split"]["dev"]
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
    if "--pool-version" in argv:                    # old flag: only pool v3 is scored now
        i = argv.index("--pool-version")
        if argv[i + 1] != "v3":
            sys.exit("only pool v3 is scored; pool v1/v2 results are kept unchanged in archive/")
        argv = argv[:i] + argv[i + 2:]
    dirs = [Path(a).resolve() for a in argv] or sorted(PARSING.glob("*"))
    dirs = [d for d in dirs if (d / "parses.jsonl").exists()]
    if not dirs:
        sys.exit(f"no parser folders with parses.jsonl under {rel(PARSING)}/ (unzip the Kaggle outputs there)")
    systems, summary = [], {}
    for d in dirs:
        st = d / "system.txt"
        if not st.exists():
            sys.exit(f"{d}: missing system.txt (one line, e.g. intent_prompted_7b)")
        system = st.read_text().strip()
        out = stage1_dir(system)
        print(f"== {system}  ({rel(d)} -> {rel(out)})")
        print(run("eval/contract_metrics.py", "--parses", d / "parses.jsonl", "--out", out / "contract_metrics.json"))
        run("eval/run_intent_oracle.py", "--parses", d / "parses.jsonl", "--system", system, "--out", out, "--pool", POOL)
        cm = json.loads((out / "contract_metrics.json").read_text())
        summary[system] = dict(folder=rel(d), results=rel(out), contract_metrics=rel(out / "contract_metrics.json"),
                               overall=cm["overall"], test=cm["by_split"].get("test"))
        systems.append(system)
    systems.sort(key=lambda s: ORDER.index(s) if s in ORDER else len(ORDER))
    verdicts = BASE_VERDICTS + [stage1_dir(s) / f"verdicts_{s}.jsonl" for s in systems]
    STAGE1_SUMMARY.mkdir(parents=True, exist_ok=True)
    (STAGE1_SUMMARY / "contract_summary.json").write_text(json.dumps(summary, indent=2))
    selection = select_on_dev(summary)
    (STAGE1_SUMMARY / "baseline3_selection.json").write_text(json.dumps(selection, indent=2))
    table = run("eval/metrics.py", "--pool", POOL, "--verdicts", *verdicts, "--out", STAGE1_SUMMARY / "metrics_stage1.json")
    keep = [l for l in table.splitlines() if l.startswith(("===", "system")) or " overall " in l or l.startswith("McNemar baseline2 vs")
            or ("McNemar intent_oracle vs" in l)]
    print("\n".join(keep))
    print("\nPrompt selection on DEV only (lowest invalid rate, then F1):")
    for base, sel in selection.items():
        print(f"  {base:<22} -> {sel['selected']:<28} " +
              " | ".join(f"{c['version']}: invalid {c['dev_invalid']:.3f}, F1 {c['dev_mean_f1']:.3f}" for c in sel["candidates"]))
    print(f"\nfull table: {rel(STAGE1_SUMMARY / 'metrics_stage1.json')}")


if __name__ == "__main__":
    main()
