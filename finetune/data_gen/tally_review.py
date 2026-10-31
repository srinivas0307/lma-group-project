"""Tally the 100-example hand review of the fine-tuning data (docs/finetune_data_guide.md §8).

    .venv/bin/python finetune/data_gen/tally_review.py                 # reads finetune/data/review.csv
    .venv/bin/python finetune/data_gen/tally_review.py --csv other.csv

Read-only: it never writes to review.csv, so it can run while the review is still being filled in.
Writes finetune/data/review_summary.json and prints a short report.

Verdicts (case and spaces/dashes don't matter):
    OK                  the example is right
    wrong_contract      the contract doesn't match the request (or breaks the annotation rules)
    unnatural_request   the request reads badly or is ambiguous
Anything else is reported as "unrecognised" (fix the spelling in the sheet); an empty cell is "not reviewed yet".

Rule (guide §8): if more than 5% of the reviewed examples are not OK, fix the templates and regenerate.
The rule is only applied once all rows are reviewed; before that the report says "incomplete".
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_NOT_OK = 0.05
ALIASES = {
    "ok": "OK", "okay": "OK", "correct": "OK", "yes": "OK",
    "wrong_contract": "wrong_contract", "wrongcontract": "wrong_contract", "wrong": "wrong_contract",
    "unnatural_request": "unnatural_request", "unnaturalrequest": "unnatural_request", "unnatural": "unnatural_request",
}


def normalise(v):
    v = (v or "").strip()
    if not v:
        return None
    key = v.lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, ALIASES.get(key.replace("_", ""), "unrecognised"))


def tally(rows):
    reviewed = [r for r in rows if r["verdict_norm"] is not None]
    counts = Counter(r["verdict_norm"] for r in reviewed)
    not_ok = [r for r in reviewed if r["verdict_norm"] != "OK"]

    def group(key):
        g = defaultdict(lambda: Counter())
        for r in rows:
            g[r[key]]["n"] += 1
            if r["verdict_norm"] is not None:
                g[r[key]]["reviewed"] += 1
                g[r[key]]["not_ok"] += r["verdict_norm"] != "OK"
        return {k: dict(n=v["n"], reviewed=v["reviewed"], not_ok=v["not_ok"],
                        not_ok_rate=round(v["not_ok"] / v["reviewed"], 3) if v["reviewed"] else None)
                for k, v in sorted(g.items())}

    complete = len(reviewed) == len(rows) and counts.get("unrecognised", 0) == 0
    rate = round(len(not_ok) / len(reviewed), 3) if reviewed else None
    decision = ("incomplete" if not complete else
                "pass: train on this data" if rate <= MAX_NOT_OK else
                "fail: fix the templates of the patterns below, regenerate, re-check and re-review")
    return dict(
        n_rows=len(rows), n_reviewed=len(reviewed), n_not_reviewed=len(rows) - len(reviewed),
        verdicts=dict(counts), not_ok=len(not_ok), not_ok_rate=rate, threshold=MAX_NOT_OK,
        complete=complete, decision=decision,
        by_type=group("type"), by_pattern=group("pattern"), by_style=group("style"),
        not_ok_examples=[dict(id=r["id"], type=r["type"], pattern=r["pattern"], verdict=r["verdict_norm"],
                              raw_verdict=r["verdict"], note=r.get("note", "")) for r in not_ok],
        patterns_to_fix=sorted({r["pattern"] for r in not_ok}),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "finetune/data/review.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "finetune/data/review_summary.json")
    args = ap.parse_args(argv)
    with open(args.csv, newline="", encoding="utf-8-sig") as fh:     # utf-8-sig: sheets saved by Excel add a BOM
        rows = list(csv.DictReader(fh))
    missing = {"id", "type", "pattern", "style", "verdict"} - set(rows[0] if rows else {})
    if missing:
        sys.exit(f"{args.csv} lacks columns {sorted(missing)}")
    for r in rows:
        r["verdict_norm"] = normalise(r.get("verdict"))
    s = tally(rows)
    args.out.write_text(json.dumps(s, indent=2))

    print(f"reviewed {s['n_reviewed']}/{s['n_rows']}  verdicts {s['verdicts']}")
    if s["n_reviewed"]:
        print(f"not OK: {s['not_ok']}/{s['n_reviewed']} = {s['not_ok_rate']:.1%} (limit {MAX_NOT_OK:.0%})")
    if s["verdicts"].get("unrecognised"):
        bad = [r["id"] for r in rows if r["verdict_norm"] == "unrecognised"]
        print(f"unrecognised verdicts (use OK / wrong_contract / unnatural_request): {bad}")
    print("by type:")
    for k, v in s["by_type"].items():
        print(f"  {k:<24} reviewed {v['reviewed']:>3}/{v['n']:<3} not OK {v['not_ok']}")
    if s["not_ok_examples"]:
        print("not OK examples:")
        for e in s["not_ok_examples"]:
            print(f"  {e['id']:<26} {e['pattern']:<32} {e['verdict']:<18} {e['note']}")
    print(f"decision: {s['decision']}")
    print(f"summary -> {args.out}")
    return s


if __name__ == "__main__":
    main()
