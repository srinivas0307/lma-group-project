"""Fix the template problems found in the hand review (finetune/data/review.csv, 9/100 not OK), without regenerating.

    .venv/bin/python finetune/data_gen/fix_review_issues.py
    # then, on the fixed raw file:
    .venv/bin/python finetune/data_gen/check_examples.py finetune/data/v2/raw.jsonl --out finetune/data/v2/checked.jsonl
    .venv/bin/python finetune/data_gen/build_sft.py finetune/data/v2/checked.jsonl --val-datasets student --out-dir finetune/data/v2

Input: finetune/data/checked.jsonl (data v1, unchanged). Output: finetune/data/v2/raw.jsonl + fix_log.json + review_changed.csv.
Regenerating with an edited generator would change the random stream (every example would differ and the 100
reviewed ones would no longer exist), so the fixes are applied here, per example, and logged.

Fixes (one per review finding; nothing else is touched):
  F1 dedup without a keep rule (7 wrong_contract): `duplicate_count eq 0` without `keep` only checks uniqueness, so
     an empty or one-row output passes. Every such example's reference code already keeps the first row
     (`drop_duplicates(..., keep="first")`), and every evaluation gold contract states a keep rule. Fix: add
     "keep": "first" to the predicate and state the rule in the request ("keep the first one").
  F2 dedup_on_key, single key, "Each combination of X should appear only once" (1 unnatural_request):
     -> "Each value of X should appear only once".
  F3 filter_with_nulls (nulls kept), "Discard <E> whose <col> is less than <t>. <E> without a <col> stay in."
     (1 unnatural_request). Second review (v2/review_changed.csv): "with a missing <col> are kept" was still unnatural,
     so the reviewer's wording is used: -> "Discard <e> with <col> below <t>. Keep <e> with missing <col> values."
"""
import csv
import os
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "finetune/data/checked.jsonl"
OUT_DIR = ROOT / "finetune/data/v2"
REVIEW = ROOT / "finetune/data/review.csv"
NO_KEEP_PHRASES = (" (one copy is enough)", " (keep one of each)")
KEEP_FIRST = " (keep the first one)"


def fix_dedup_keep(e):
    pred = e["contract"]["must_change"][0]
    if e["transformation_type"] != "deduplication" or pred.get("type") != "duplicate_count" or "keep" in pred:
        return None
    if 'keep="first"' not in e["reference_code"]:
        raise SystemExit(f"{e['id']}: reference code doesn't keep the first row; fix by hand")
    old = e["request"]
    new = old
    for p in NO_KEEP_PHRASES:
        if p in new:
            new = new.replace(p, KEEP_FIRST)
            break
    else:                                   # the template's empty variant: append the rule before the final period
        new = new[:-1] + KEEP_FIRST + "." if new.endswith(".") else new + KEEP_FIRST
    pred["keep"] = "first"
    e["request"] = new
    return dict(fix="F1", field="contract+request", old=old, new=new)


def fix_single_key_combination(e):
    if e["pattern"] != "dedup_on_key":
        return None
    tgt = e["contract"]["must_change"][0].get("target")
    if not (isinstance(tgt, list) and len(tgt) == 1 and e["request"].startswith("Each combination of ")):
        return None
    old = e["request"]
    e["request"] = "Each value of " + old[len("Each combination of "):]
    return dict(fix="F2", field="request", old=old, new=e["request"])


_F3 = re.compile(r"^Discard (.+?) whose (.+?) is less than (.+?)\. \S.*? without an? .+? stay in\.$")


def fix_filter_nulls_wording(e):
    if e["pattern"] != "filter_with_nulls":
        return None
    m = _F3.match(e["request"])
    if not m:
        return None
    old = e["request"]
    ent, col, t = m.groups()
    e["request"] = f"Discard {ent} with {col} below {t}. Keep {ent} with missing {col} values."
    return dict(fix="F3", field="request", old=old, new=e["request"])


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    for e in rows:
        for f in (fix_dedup_keep, fix_single_key_combination, fix_filter_nulls_wording):
            ch = f(e)
            if ch:
                log.append(dict(id=e["id"], pattern=e["pattern"], **ch))
    (OUT_DIR / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_fix = Counter(c["fix"] for c in log)
    changed_ids = sorted({c["id"] for c in log})
    (OUT_DIR / "fix_log.json").write_text(json.dumps(dict(
        source=str(SRC.relative_to(ROOT)), n_examples=len(rows), n_changed=len(changed_ids), by_fix=dict(by_fix),
        changes=log), indent=2))

    # Re-review sheet: every example that failed the first review, plus a random sample of the other changed ones.
    reviewed_bad = set()
    if REVIEW.exists():
        with open(REVIEW, newline="", encoding="utf-8-sig") as fh:
            reviewed_bad = {r["id"] for r in csv.DictReader(fh) if (r.get("verdict") or "").strip().lower() not in ("", "ok")}
    by_id = {e["id"]: e for e in rows}
    rng = random.Random(0)
    others = [i for i in changed_ids if i not in reviewed_bad]
    pick = sorted(i for i in reviewed_bad if i in by_id) + sorted(rng.sample(others, min(16, len(others))))
    sheet = OUT_DIR / "review_changed.csv"
    if sheet.exists():                       # never overwrite a sheet that may already hold review verdicts
        print(f"kept existing {sheet} (already reviewed); not rewritten")
        pick = []
    with open(os.devnull if sheet.exists() else sheet, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "type", "pattern", "style", "request", "contract", "reference_code", "was_in_first_review",
                    "verdict", "note"])
        for i in pick:
            e = by_id[i]
            w.writerow([i, e["transformation_type"], e["pattern"], e["style"], e["request"], json.dumps(e["contract"]),
                        e["reference_code"], "yes" if i in reviewed_bad else "no", "", ""])
    print(f"{len(rows)} examples, {len(changed_ids)} changed {dict(by_fix)} -> {OUT_DIR / 'raw.jsonl'}")
    print(f"log {OUT_DIR / 'fix_log.json'}; re-review sheet ({len(pick)} rows) {OUT_DIR / 'review_changed.csv'}")
    missing = reviewed_bad - {c["id"] for c in log}
    if missing:
        print(f"WARNING: reviewed-not-OK examples not changed by any fix: {sorted(missing)}")


if __name__ == "__main__":
    main()
