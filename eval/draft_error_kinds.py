"""Draft error-kind labels for every runnable wrong candidate of pool v4 (README Phase 5 task 3, Phase 12 failure
taxonomy). CPU only. The labels are DRAFTS for a person to review (column `reviewed_by`).

    .venv/bin/python eval/draft_error_kinds.py     # -> benchmark/annotation/error_kinds_pool_v4.csv

Source of each label (`labelled_by`):
  kiro-draft (pool v1)   copied from benchmark/annotation/error_kinds_pool_v1.csv when the candidate id and its code
                         are unchanged (all 49 pool v1 labels qualify)
  rule:<name>            a fixed rule over the code and the output compared with the reference output (below)
  unclassified           no rule applies: a person must label it
Kinds: docs/error_taxonomy.md (v1 adds wrong_null_handling, wrong_keep_rule, wrong_target_values,
groupby_first_mixes_rows, removes_too_many_rows, removes_too_few_rows). Rules, in order (the first that applies wins):
  columns differ        extra new columns only -> helper_column_left; a new column and a missing one ->
                        wrong_output_column; a column missing -> unrequested_change
  category order        -> wrong_category_order
  dtype differs         on a column the contract targets -> wrong_target_type, otherwise -> unrequested_change
  deduplication         keep=False -> removes_all_copies; selects .duplicated() rows -> inverted_selection;
                        groupby().first()/last() -> groupby_first_mixes_rows; dropna before deduplicating ->
                        removes_too_many_rows; sort before drop_duplicates -> order_changed_before_keep;
                        drop_duplicates key smaller than the gold key -> incomplete_key, other key -> wrong_key;
                        right key, other keep rule (missing keep = "first") -> wrong_keep_rule
  filtering / missing   the code never names a column the request conditions on -> wrong_column
  row differences       kept = exactly the rows the reference removes -> inverted_selection; the wrong rows mostly
                        have a missing value in a condition column -> wrong_null_handling; rows wrong both ways ->
                        wrong_boolean_logic (code with & or |) or wrong_proxy_condition; one way ->
                        removes_too_many_rows / removes_too_few_rows (judged by row count if the index was lost)
  values differ         (aligned by index, or by position if the index was reset) in a column the contract doesn't
                        target -> unrequested_change, else wrong_target_values
Existing reviewed labels are never overwritten: if the CSV exists, rows with `reviewed_by` filled are kept as they are.
"""
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.equivalence import outputs_equal  # noqa: E402
from igt.contracts.schema import expand_columns  # noqa: E402
from igt.paths import ARCHIVE, POOL, STAGE1, TASKS  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402

OUT = ROOT / "benchmark" / "annotation" / "error_kinds_pool_v4.csv"
V1 = ROOT / "benchmark" / "annotation" / "error_kinds_pool_v1.csv"
FIELDS = ["candidate_id", "task_id", "split", "transformation_type", "primary_kind", "secondary_kind", "note",
          "answer_key_reason", "baseline2_outcome", "baseline3_outcome", "oracle_outcome", "labelled_by", "reviewed_by"]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def contract_columns(contract, cols):
    """Columns the must_change part targets or conditions on."""
    out = set()

    def cond(c):
        if not c:
            return
        if "column" in c:
            out.add(c["column"])
        for k in ("all", "any"):
            for x in c.get(k, []):
                cond(x)
        cond(c.get("not"))
    for p in contract.get("must_change") or []:
        t = p.get("target")
        if isinstance(t, str) and t != "*":
            out.add(t)
        elif isinstance(t, list):
            out |= set(t)
        cond(p.get("condition"))
        cond(p.get("where"))
    return {c for c in out if c in cols} | {c for c in out if c not in cols}


def gold_key(contract, cols):
    for p in contract.get("must_change") or []:
        if p.get("type") == "duplicate_count":
            return set(expand_columns(p["target"], p.get("except") or [], cols)), p.get("keep")
    return None, None


def code_key(code, cols):
    """Key columns of the first drop_duplicates call: subset=... anywhere in the call, else the first positional list."""
    m = re.search(r"drop_duplicates\(([^)]*)\)", code, re.S)
    if not m:
        return None
    args = m.group(1)
    s = re.search(r"subset\s*=\s*(\[[^\]]*\]|'[^']*'|\"[^\"]*\")", args) or re.match(r"\s*(\[[^\]]*\]|'[^']*'|\"[^\"]*\")", args)
    if s:
        return set(re.findall(r"['\"]([^'\"]+)['\"]", s.group(1)))
    return None if "subset" in args or "columns" in args else set(cols)


def code_keep(code):
    if "drop_duplicates" not in code:
        return None
    m = re.search(r"keep\s*=\s*(False|['\"](first|last)['\"])", code)
    return "first" if not m else ("none" if m.group(1) == "False" else m.group(2))


def code_columns(code, cols):
    return {c for c in cols if re.search(r"['\"]" + re.escape(str(c)) + r"['\"]", code)}


def classify(t, cand, before, ref, out):
    cols = list(before.columns)
    targets = contract_columns(t["gold_contract"], cols)
    reason = str(outputs_equal(ref, out)[1])
    rc, oc = set(map(str, ref.columns)), set(map(str, out.columns))
    if rc != oc:
        extra, missing = oc - rc, rc - oc
        if extra and not missing:
            return "helper_column_left", f"extra columns {sorted(extra)}"
        if extra and missing:
            return "wrong_output_column", f"writes {sorted(extra)}, drops {sorted(missing)}"
        return "unrequested_change", f"drops columns {sorted(missing)}"
    if "category order" in reason:
        return "wrong_category_order", reason
    if "dtype family differs in" in reason:
        col = reason.split("dtype family differs in", 1)[1].strip().split(" ")[0]
        return ("wrong_target_type" if col in targets else "unrequested_change"), reason
    code = cand["code"]
    ttype = t["transformation_type"]
    if ttype == "deduplication":
        gk, gkeep = gold_key(t["gold_contract"], cols)
        ck, keep = code_key(code, cols), code_keep(code)
        if keep == "none":
            return "removes_all_copies", "keep=False"
        if re.search(r"\.duplicated\(", code) and "drop_duplicates" not in code:
            return "inverted_selection", "selects rows with .duplicated() instead of dropping them"
        if re.search(r"groupby\([^)]*\)\s*\.\s*(first|last)\(\)", code):
            return "groupby_first_mixes_rows", "groupby().first()/last() takes each column's first non-missing value separately"
        if "dropna" in code and len(out) < len(ref):
            return "removes_too_many_rows", "drops rows with missing values before deduplicating"
        sv, dd = code.find("sort_values"), code.find("drop_duplicates")
        if sv != -1 and (dd == -1 or sv < dd):
            return "order_changed_before_keep", "sorts before keeping the first/last row"
        if ck is not None and gk is not None and ck != gk:
            return ("incomplete_key" if ck < gk else "wrong_key"), f"key {sorted(ck)[:6]} vs gold {sorted(gk)[:6]}"
        if gkeep and keep and keep != gkeep:
            return "wrong_keep_rule", f"keep={keep}, request: {gkeep}"
    cond_cols = [c for c in targets if c in before.columns]
    if ttype in ("filtering", "missing_values") and cond_cols and not (code_columns(code, cols) & set(cond_cols)):
        return "wrong_column", f"the code never uses {cond_cols}"
    if len(ref) != len(out) or "row count" in reason:
        if out.index.is_unique and set(out.index) <= set(before.index) and set(ref.index) <= set(before.index):
            kept, rkept = set(out.index), set(ref.index)
            if kept == set(before.index) - rkept:
                return "inverted_selection", "keeps exactly the rows the reference removes"
            wrong_removed, wrong_kept = rkept - kept, kept - rkept
            diff = list(wrong_removed | wrong_kept)
            if diff and cond_cols and before.loc[diff, cond_cols].isna().any(axis=1).mean() > 0.5:
                return "wrong_null_handling", f"{len(wrong_removed)} rows wrongly removed, {len(wrong_kept)} wrongly kept, mostly with missing {cond_cols}"
            if wrong_removed and wrong_kept:
                return ("wrong_boolean_logic" if re.search(r"[&|]", code) else "wrong_proxy_condition"), \
                    f"{len(wrong_removed)} rows wrongly removed and {len(wrong_kept)} wrongly kept"
            if wrong_removed:
                return "removes_too_many_rows", f"{len(wrong_removed)} rows removed that the reference keeps"
            if wrong_kept:
                return "removes_too_few_rows", f"{len(wrong_kept)} rows kept that the reference removes"
        if "dropna" in code and len(out) < len(ref):
            return "wrong_null_handling", "drops rows with missing values (index not kept)"
        return ("removes_too_many_rows" if len(out) < len(ref) else "removes_too_few_rows"), \
            f"{len(out)} rows vs {len(ref)} in the reference (index not kept, judged by row count)"
    if "values differ" in reason:
        common = [c for c in ref.columns if c in out.columns]
        if set(out.index) == set(ref.index):
            o = out.loc[ref.index]
        elif len(out) == len(ref):
            o = out.set_axis(ref.index)                   # index was reset: compare by position
        else:
            return "unclassified", reason
        changed = [c for c in common if not ref[c].astype(object).where(ref[c].notna(), None).equals(
            o[c].astype(object).where(o[c].notna(), None))]
        if changed and not set(changed) <= targets:
            return "unrequested_change", f"changes {sorted(set(changed) - targets)}"
        if changed:
            return "wrong_target_values", f"wrong values in {changed}"
    return "unclassified", reason


def main():
    pool = load_jsonl(POOL)
    tasks = {t["id"]: t for t in (yaml.safe_load(f.read_text()) for f in sorted(TASKS.glob("*/*.yaml")))}
    verdicts = {}
    for name, f in (("baseline2", STAGE1 / "baseline2" / "verdicts_baseline2.jsonl"),
                    ("baseline3", STAGE1 / "baseline3" / "verdicts_intent_prompted_7b_p1v3.jsonl"),
                    ("oracle", STAGE1 / "oracle" / "verdicts_intent_oracle.jsonl")):
        verdicts[name] = {r["candidate_id"]: r["outcome"] for r in load_jsonl(f)}
    v1_hash = {c["candidate_id"]: c["code_hash"] for c in load_jsonl(ARCHIVE / "pools" / "pool_v1" / "pool_v1.jsonl")}
    with open(V1, newline="") as fh:
        v1 = {r["candidate_id"]: r for r in csv.DictReader(fh)}
    kept = {}
    if OUT.exists():
        with open(OUT, newline="") as fh:
            kept = {r["candidate_id"]: r for r in csv.DictReader(fh) if (r.get("reviewed_by") or "").strip()}
    data, refs, rows = {}, {}, []
    for c in pool:
        if not c["ran_ok"] or c["is_correct"]:
            continue
        cid, t = c["candidate_id"], tasks[c["task_id"]]
        if cid in kept:
            rows.append(kept[cid])
            continue
        if t["data_file"] not in data:
            data[t["data_file"]] = pd.read_csv(ROOT / t["data_file"])
        before = data[t["data_file"]]
        if t["id"] not in refs:
            refs[t["id"]] = safe_run(t["reference_code"], before).df_out
        with open(POOL.parent / c["output_file"], "rb") as fh:
            out = pickle.load(fh)
        reason = str(outputs_equal(refs[t["id"]], out)[1])
        if cid in v1 and v1_hash.get(cid) == c["code_hash"]:
            kind, second, note, by = v1[cid]["primary_kind"], v1[cid]["secondary_kind"], v1[cid]["note"], "kiro-draft (pool v1)"
        else:
            kind, note = classify(t, c, before, refs[t["id"]], out)
            second, by = "", "unclassified" if kind == "unclassified" else "rule:" + kind
        rows.append(dict(candidate_id=cid, task_id=t["id"], split=c["split"], transformation_type=t["transformation_type"],
                         primary_kind=kind, secondary_kind=second, note=note, answer_key_reason=reason,
                         baseline2_outcome=verdicts["baseline2"].get(cid), baseline3_outcome=verdicts["baseline3"].get(cid),
                         oracle_outcome=verdicts["oracle"].get(cid), labelled_by=by, reviewed_by=""))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} wrong candidates -> {OUT}")
    print("by source:", dict(Counter(r["labelled_by"].split(":")[0] for r in rows)))
    print("by kind:", dict(Counter(r["primary_kind"] for r in rows).most_common()))
    return rows


if __name__ == "__main__":
    main()
