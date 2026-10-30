"""Baseline 1 (direct LLM, no verification) per dataset and generator, from the pool v3 generations. No GPU, no new run.

Baseline 1 has no verifier: every output that runs is accepted. So its numbers come straight from the generations
already made on Kaggle (both generators, all four datasets) and the pool's correctness labels:
- greedy accuracy: share of tasks whose greedy (temperature 0) output equals the reference output;
- sample accuracy: the same over all 6 generations per task (1 greedy + 5 samples at T = 0.7);
- silent error rate: wrong among accepted (= ran) unique pool candidates, the Stage 1 Baseline 1 metric.
Each generation's label is the label of its pool candidate (same task + normalized code), i.e. the label re-checked
locally with the pinned pandas, not the Kaggle-side label.

    .venv/bin/python eval/baseline1_by_dataset.py            # -> results/baseline1_v3/by_dataset.json + printed table
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "eval" / "pool_v3" / "pool_v3.jsonl"
OUT = ROOT / "results" / "baseline1_v3"


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    manifest = json.loads((POOL.parent / "manifest.json").read_text())
    pool = load_jsonl(POOL)
    label = {(c["task_id"], c["code_hash"]): bool(c["is_correct"]) for c in pool if c["code_hash"]}
    gens = []
    for src in manifest["sources"]:
        p = Path(src["path"])
        p = p if p.exists() else ROOT / p.relative_to(p.anchor).as_posix().split("intent-parser/", 1)[-1]
        gens += load_jsonl(p / "candidates.jsonl")       # one line per generation, with its code_hash
    acc = defaultdict(lambda: [0, 0])                     # (split, model, dataset, kind) -> [correct, n]
    for g in gens:
        ok = bool(g.get("code_hash")) and label.get((g["task_id"], g["code_hash"]), False)
        if g.get("code_hash") and (g["task_id"], g["code_hash"]) not in label:
            sys.exit(f"generation {g['gen_id']} ({g['model']}) has no pool candidate")
        model = g["model"].split("/")[-1].replace("-Instruct", "")
        for ds in (g["dataset"], "all"):
            for kind in (["greedy"] if g["mode"] == "greedy" else []) + ["samples"]:
                a = acc[(g["split"], model, ds, kind)]
                a[0] += ok
                a[1] += 1
    sil = defaultdict(lambda: [0, 0])                     # (split, dataset) -> [wrong among ran, ran]
    for c in pool:
        if c["ran_ok"]:
            for ds in (c["dataset"], "all"):
                s = sil[(c["split"], ds)]
                s[0] += not c["is_correct"]
                s[1] += 1
    out = dict(pool="v3", n_generations=len(gens),
               accuracy={"|".join(k): dict(correct=v[0], n=v[1], rate=round(v[0] / v[1], 3)) for k, v in sorted(acc.items())},
               silent_error_rate={"|".join(k): dict(wrong=v[0], ran=v[1], rate=round(v[0] / v[1], 3)) for k, v in sorted(sil.items())})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "by_dataset.json").write_text(json.dumps(out, indent=2))

    dss = ["adult", "airbnb", "taxi", "titanic", "all"]
    for split in ("test", "dev"):
        print(f"\n{split}: accuracy (correct / generations)")
        print(f"{'generator':<16}{'kind':<9}" + "".join(f"{d:>15}" for d in dss))
        for model in sorted({k[1] for k in acc}):
            for kind in ("greedy", "samples"):
                cells = [acc.get((split, model, d, kind)) for d in dss]
                print(f"{model:<16}{kind:<9}" + "".join(f"{c[0]:>4}/{c[1]:<4}{c[0] / c[1]:>6.3f}" for c in cells))
        cells = [sil[(split, d)] for d in dss]
        print(f"{'silent error rate (unique candidates that ran)':<25}")
        print(f"{'':<25}" + "".join(f"{c[0]:>4}/{c[1]:<4}{c[0] / c[1]:>6.3f}" for c in cells))
    print(f"\n-> {OUT / 'by_dataset.json'}")


if __name__ == "__main__":
    main()
