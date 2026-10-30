"""One-time move of the project into the clean layout (2026-10-30). Moves only; nothing with results is deleted.

    python tools/restructure.py --dry-run     # show what would move
    python tools/restructure.py               # do it (uses `git mv` inside a git repo, so history is kept)

Safe to run twice: moves whose source is gone and whose target exists are skipped.
Deleted (no results in them): the .xls files in data/finetune/ that are byte-identical to the .csv files,
notebooks/baseline1_outputs/jobs/ (Kaggle scratch files) and finetune/runs/smoke_cpu/ (CPU smoke-test adapter).
"""
import argparse
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MOVES = [
    # course material
    ("handout.txt", "docs/course/handout.txt"),
    ("proposal.txt", "docs/course/proposal.txt"),
    ("SSSS_LMA_PROJECT_PROPOSAL.pdf", "docs/course/SSSS_LMA_PROJECT_PROPOSAL.pdf"),
    ("LMA_Group_Project_v1 (1).pdf", "docs/course/LMA_Group_Project_v1.pdf"),
    # frozen benchmark inputs
    ("eval/tasks", "benchmark/tasks"),
    ("eval/prompts", "benchmark/prompts"),
    ("eval/redteam", "benchmark/redteam"),
    ("eval/annotation", "benchmark/annotation"),
    ("eval/agent_scenarios/rct_v1.jsonl", "archive/agent_scenarios/rct_v1.jsonl"),
    ("eval/agent_scenarios", "benchmark/agent_scenarios"),
    ("eval/pool_v3/verdicts_baseline1.jsonl", "results/stage1/baseline1/verdicts_baseline1.jsonl"),
    ("eval/pool_v3", "benchmark/pool"),
    ("eval/pool_v1", "archive/pools/pool_v1"),
    ("eval/pool_v2", "archive/pools/pool_v2"),
    # raw Kaggle downloads
    ("notebooks/baseline1_outputs", "kaggle_outputs/generation/3b_adult_airbnb"),
    ("outputs/gen_qwen15b", "kaggle_outputs/generation/1_5b_adult_airbnb"),
    ("outputs/gen_qwen25-3b-instruct_taxi_titanic", "kaggle_outputs/generation/3b_taxi_titanic"),
    ("outputs/gen_qwen25-15b-instruct_taxi_titanic", "kaggle_outputs/generation/1_5b_taxi_titanic"),
] + [(f"outputs/parse_qwen25-{m}-instruct_p1-v{v}", f"kaggle_outputs/parsing/{n}_p1-v{v}")
     for m, n in (("7b", "7b"), ("3b", "3b"), ("15b", "1_5b")) for v in (1, 2, 3)] + [
    # Stage 1 results (pool v3 = main)
    ("results/baseline1_v3/by_dataset.json", "results/stage1/baseline1/by_dataset.json"),
    ("results/baseline2_v3", "results/stage1/baseline2"),
    ("results/intent_oracle_v3", "results/stage1/oracle"),
    ("results/intent_prompted_7b_p1v3_v3", "results/stage1/baseline3"),
    ("results/intent_prompted_3b_p1v2_v3", "results/stage1/size_ablation/intent_prompted_3b_p1v2"),
    ("results/intent_prompted_1_5b_p1v3_v3", "results/stage1/size_ablation/intent_prompted_1_5b_p1v3"),
] + [(f"results/intent_prompted_{s}_v3", f"results/stage1/other_prompts/intent_prompted_{s}")
     for s in ("7b", "7b_p1v2", "3b", "3b_p1v3", "1_5b", "1_5b_p1v2")] + [
    ("results/parsers_v3", "results/stage1/summary"),
    # superseded results (pool v1 / v2), kept unchanged
    ("results/baseline2", "archive/results/baseline2_v1"),
    ("results/baseline2_v2", "archive/results/baseline2_v2"),
    ("results/intent_oracle_v1", "archive/results/intent_oracle_v1"),
    ("results/intent_oracle_v2", "archive/results/intent_oracle_v2"),
    ("results/parsers_v2", "archive/results/parsers_v2"),
    ("results/agent_v1", "archive/results/agent_v1"),
] + [(f"results/intent_prompted_{s}_v2", f"archive/results/intent_prompted_{s}_v2")
     for s in ("7b", "7b_p1v2", "7b_p1v3", "3b", "3b_p1v2", "3b_p1v3", "1_5b", "1_5b_p1v2", "1_5b_p1v3")] + [
    ("configs/baseline2.yaml", "archive/configs/baseline2_v1.yaml"),
    ("configs/baseline2_v2.yaml", "archive/configs/baseline2_v2.yaml"),
    # notebooks: the ones you run vs the executed copies downloaded from Kaggle
] + [(f"notebooks/{n}.ipynb", f"notebooks/kaggle/{n}.ipynb")
     for n in ("gate1_generate_kaggle", "baseline3_parse_kaggle", "finetune_qlora_kaggle", "agent_eval_kaggle")] + [
    ("notebooks/baseline1_kaggle.ipynb", "archive/notebooks/baseline1_kaggle.ipynb"),     # superseded by gate1_generate
    ("notebooks/kaggle/baseline1_kaggle.ipynb", "archive/notebooks/baseline1_kaggle.ipynb"),
] + [(f"notebooks/{n}.ipynb", f"notebooks/executed/{n}.ipynb")
     for n in ("baseline-1_ran-notebook-v1", "baseline-3_ran-notebook", "baseline-3_ran-notebook-v2",
               "baseline-3_ran-notebook-v3", "gate-1-new-datasets_qewn-1.5B", "gate-1-new-datasets_qewn-3B",
               "agent-eval_ran-notebook", "new-baseline-1", "new-baseline-1_complete")]

DELETE_DIRS = ["notebooks/baseline1_outputs/jobs", "kaggle_outputs/generation/3b_adult_airbnb/jobs",
               "finetune/runs/smoke_cpu"]
DUPLICATE_XLS = ["data/finetune/imdb_top_1000.xls", "data/finetune/student-mat.xls", "data/finetune/Walmart.xls"]
EMPTY_AFTER = ["outputs", "eval/agent_scenarios", "results/baseline1_v3"]


def in_git(root):
    return subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True).returncode == 0


def tracked(root, p):
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=root, capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root, git = args.root.resolve(), in_git(args.root)
    moved = skipped = 0
    for src, dst in MOVES:
        s, d = root / src, root / dst
        if not s.exists():
            skipped += 1
            continue
        if d.exists():
            sys.exit(f"both {src} and {dst} exist; stop (nothing else moved after this point)")
        print(f"move  {src}  ->  {dst}")
        moved += 1
        if args.dry_run:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if git and tracked(root, s):
            subprocess.run(["git", "mv", str(s), str(d)], cwd=root, check=True)
            if s.exists():                       # untracked files left behind in a tracked folder
                shutil.copytree(s, d, dirs_exist_ok=True)
                shutil.rmtree(s)
        else:
            shutil.move(str(s), str(d))
    for x in DUPLICATE_XLS:
        p, csv = root / x, (root / x).with_suffix(".csv")
        if p.exists() and csv.exists() and filecmp.cmp(p, csv, shallow=False):
            print(f"delete {x}  (identical to {csv.name})")
            if not args.dry_run:
                (subprocess.run(["git", "rm", "-q", x], cwd=root, check=True) if git and tracked(root, p) else p.unlink())
    for x in DELETE_DIRS:
        p = root / x
        if p.exists():
            print(f"delete {x}/")
            if not args.dry_run:
                shutil.rmtree(p)
    for x in EMPTY_AFTER:
        p = root / x
        if p.is_dir() and not any(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts):
            print(f"remove empty {x}/")
            if not args.dry_run:
                shutil.rmtree(p)
    print(f"{moved} moved, {skipped} already done or not present" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
