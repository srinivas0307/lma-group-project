"""Step 6 of the fine-tuning data (docs/finetune_data_guide.md §8): the 100-example hand review sheet.

    .venv/bin/python finetune/data_gen/review_sample.py      # -> finetune/data/review.csv

Draws 20 random checked examples per transformation type (seed 0). A reviewer fills `verdict` with
OK / wrong_contract / unnatural_request and adds a note. More than ~5% not OK -> fix the templates, regenerate.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = [json.loads(l) for l in (ROOT / "finetune/data/checked.jsonl").read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["transformation_type"]].append(r)
    rng = random.Random(0)
    pick = [r for t in sorted(by) for r in rng.sample(by[t], min(20, len(by[t])))]
    out = ROOT / "finetune/data/review.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "type", "pattern", "style", "request", "contract", "reference_code", "verdict", "note"])
        for r in pick:
            w.writerow([r["id"], r["transformation_type"], r["pattern"], r["style"], r["request"],
                        json.dumps(r["contract"]), r["reference_code"], "", ""])
    print(f"{len(pick)} examples -> {out}")


if __name__ == "__main__":
    main()
