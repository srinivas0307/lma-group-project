"""One check per predicate type (§5.5). Each returns a PredicateResult: holds / violated / uncertain.

All checks receive the before frame, the after frame and the row alignment; evidence rows are BEFORE ids.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from igt.contracts.schema import expand_columns

from .conditions import ConditionError, evaluate
from .functions import expected_values
from .values import DEFAULT_ATOL, as_objects, dtype_family, equal_mask, hashable_key, row_keys

MAX_EVIDENCE = 5


@dataclass
class PredicateResult:
    status: str                       # holds | violated | uncertain
    expected: str = ""
    observed: object = None
    detail: str = ""
    evidence_rows: list = field(default_factory=list)
    reason: str | None = None         # for uncertain: uncertain reason code

    @property
    def violated(self):
        return self.status == "violated"

    @property
    def uncertain(self):
        return self.status == "uncertain"


def _ids(xs):
    return [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in list(xs)[:MAX_EVIDENCE]]


def holds(expected="", observed=None):
    return PredicateResult("holds", expected, observed)


def violated(expected, observed, detail="", rows=()):
    return PredicateResult("violated", expected, observed, detail, _ids(rows))


def uncertain(reason, detail):
    return PredicateResult("uncertain", detail=detail, reason=reason)


def _count_ok(op, value, after_n, before_n):
    if op == "unchanged":
        return after_n == before_n
    if op == "decreased":
        return after_n < before_n
    if op == "increased":
        return after_n > before_n
    return {"eq": after_n == value, "lt": after_n < value, "le": after_n <= value, "gt": after_n > value,
            "ge": after_n >= value}[op]


def _count_expect(op, value, before_n):
    rel = {"unchanged": f"== {before_n} (unchanged)", "decreased": f"< {before_n} (decreased)",
           "increased": f"> {before_n} (increased)"}
    return rel.get(op) or f"{op} {value}"


def _mask(cond, before):
    try:
        return evaluate(cond, before), None
    except ConditionError as e:
        return None, uncertain("predicate_not_evaluable", str(e))


def _pairs(al, before, after, cols, row_mask=None):
    """Aligned (before values, after values, before ids) for surviving rows, optionally only where row_mask."""
    pos_of = {bid: i for i, bid in enumerate(before.index)}
    keep = [(b, a) for b, a in zip(al.before_ids, al.after_pos) if row_mask is None or row_mask[pos_of[b]]]
    bids = [b for b, _ in keep]
    apos = [a for _, a in keep]
    out = {}
    for c in cols:
        bv = as_objects(before.loc[bids, c]) if c in before.columns else [None] * len(bids)
        av = as_objects(after.iloc[apos][c]) if c in after.columns else None
        out[c] = (bv, av)
    return out, bids


# ---------------- row-level ----------------
def row_count(p, before, after, al):
    nb, na = len(before), len(after)
    exp = _count_expect(p.op, p.value, nb)
    return holds(exp, na) if _count_ok(p.op, p.value, na, nb) else violated(exp, na, f"{na} rows after, {nb} before")


def rows_removed_where(p, before, after, al):
    m, err = _mask(p.condition, before)
    if err:
        return err
    target = list(before.index[m])
    if not target:
        return uncertain("predicate_not_evaluable", "condition matches 0 rows in the input")
    left = [i for i in target if i in al.surviving]
    if left:
        return violated(f"all {len(target)} matching rows removed", f"{len(left)} still present",
                        f"{len(left)} of {len(target)} rows that should be removed are still there", left)
    return holds(f"all {len(target)} matching rows removed", 0)


def rows_retained_where(p, before, after, al):
    m, err = _mask(p.condition, before)
    if err:
        return err
    target = list(before.index[m])
    gone = [i for i in target if i not in al.surviving]
    if gone:
        return violated(f"all {len(target)} matching rows kept", f"{len(gone)} removed",
                        f"{len(gone)} of {len(target)} rows that should be kept were removed", gone)
    return holds(f"all {len(target)} matching rows kept", 0)


def duplicate_count(p, before, after, al):
    keys = expand_columns(p.target, p.except_, before.columns)
    missing = [c for c in keys if c not in after.columns]
    if missing:
        return uncertain("predicate_not_evaluable", f"key column(s) {missing} missing in the output")
    nb = int(pd.Series(row_keys(before, keys)).duplicated().sum()) if len(before) else 0
    ak = pd.Series(row_keys(after, keys))
    dup_mask = ak.duplicated().to_numpy() if len(after) else np.zeros(0, dtype=bool)
    na = int(dup_mask.sum())
    exp = _count_expect(p.op, p.value, nb)
    if not _count_ok(p.op, p.value, na, nb):
        pos_to_bid = dict(zip(al.after_pos, al.before_ids))
        rows = [pos_to_bid.get(i, f"after#{i}") for i in np.flatnonzero(dup_mask)]
        return violated(exp, na, f"{na} rows still duplicate an earlier row on {keys if len(keys) < 6 else 'the key'}",
                        rows)
    if p.keep is None:
        return holds(exp, na)
    # keep rule: every before key group keeps exactly one row, the first / last by index
    groups = defaultdict(list)
    for bid, k in zip(before.index, row_keys(before, keys)):
        groups[k].append(bid)
    surv = al.surviving
    bad = []
    for ids in groups.values():
        alive = [i for i in ids if i in surv]
        want = min(ids) if p.keep == "first" else max(ids)
        ok_ids = al.equiv.get(want, frozenset([want])) if al.method == "content" else {want}
        if len(alive) != 1 or alive[0] not in ok_ids:
            bad.append(want)
    if bad:
        return violated(f"one row per key group, the {p.keep} one", f"{len(bad)} groups wrong",
                        f"{len(bad)} of {len(groups)} key groups don't keep exactly their {p.keep} row", bad)
    return holds(exp, na)


# ---------------- column-level ----------------
def null_count(p, before, after, al):
    if p.target not in after.columns:
        return uncertain("predicate_not_evaluable", f"column '{p.target}' missing in the output")
    nb, na = int(before[p.target].isna().sum()), int(after[p.target].isna().sum())
    exp = _count_expect(p.op, p.value, nb)
    if _count_ok(p.op, p.value, na, nb):
        return holds(exp, na)
    return violated(exp, na, f"{na} missing values in {p.target} after, {nb} before")


def _changed_rows(bv, av, bids, atol=DEFAULT_ATOL):
    eq = equal_mask(bv, av, atol)
    return [bids[i] for i in np.flatnonzero(~eq)]


def column_values_unchanged(p, before, after, al):
    cols = expand_columns(p.target, p.except_, before.columns)
    missing = [c for c in cols if c not in after.columns]
    if missing:
        return uncertain("predicate_not_evaluable", f"column(s) {missing} missing in the output")
    mask = None
    if p.where is not None:
        mask, err = _mask(p.where, before)
        if err:
            return err
    pairs, bids = _pairs(al, before, after, cols, mask)
    bad = {c: _changed_rows(bv, av, bids) for c, (bv, av) in pairs.items()}
    bad = {c: r for c, r in bad.items() if r}
    if bad:
        n = sum(len(r) for r in bad.values())
        return violated("values unchanged", f"{n} changed cells in {sorted(bad)}",
                        "; ".join(f"{c}: {len(r)} rows changed" for c, r in bad.items()),
                        [r for rs in bad.values() for r in rs])
    return holds("values unchanged", 0)


def columns_unchanged(p, before, after, al):
    cols = expand_columns(p.target, p.except_, before.columns)
    missing = [c for c in cols if c not in after.columns]
    if missing:
        return violated("columns present and unchanged", f"missing {missing}", f"column(s) {missing} were removed")
    fam = [c for c in cols if dtype_family(before[c]) != dtype_family(after[c])]
    if fam:
        return violated("same dtype family", {c: f"{dtype_family(before[c])}->{dtype_family(after[c])}" for c in fam},
                        f"dtype family changed in {fam}")
    pairs, bids = _pairs(al, before, after, cols)
    bad = {c: _changed_rows(bv, av, bids) for c, (bv, av) in pairs.items()}
    bad = {c: r for c, r in bad.items() if r}
    if bad:
        n = sum(len(r) for r in bad.values())
        return violated("values unchanged", f"{n} changed cells in {sorted(bad)}",
                        "; ".join(f"{c}: {len(r)} rows changed" for c, r in bad.items()),
                        [r for rs in bad.values() for r in rs])
    return holds("columns present and unchanged", 0)


def column_dtype(p, before, after, al):
    if p.target not in after.columns:
        return uncertain("predicate_not_evaluable", f"column '{p.target}' missing in the output")
    s = after[p.target]
    fam = dtype_family(s)
    want = p.dtype + (f", ordered={p.ordered}" if p.ordered is not None else "") + \
        (", categories given" if p.categories is not None else "")
    if fam != p.dtype:
        return violated(want, fam, f"{p.target} is {fam}, expected {p.dtype}")
    if p.dtype == "category":
        if p.ordered is not None and bool(s.cat.ordered) != p.ordered:
            return violated(want, f"ordered={s.cat.ordered}", f"{p.target} ordered flag is {s.cat.ordered}")
        if p.categories is not None:
            got = list(s.cat.categories)
            same = got == list(p.categories) if (p.ordered or s.cat.ordered) else set(got) == set(p.categories)
            if not same:
                return violated(want, got[:20], f"{p.target} categories differ (order matters: {bool(p.ordered)})")
    return holds(want, fam)


def column_value_relation(p, before, after, al):
    if p.target not in after.columns:
        return uncertain("predicate_not_evaluable", f"target column '{p.target}' missing in the output")
    mask = None
    if p.where is not None:
        mask, err = _mask(p.where, before)
        if err:
            return err
    try:
        exp_all = expected_values(p, before)
    except Exception as e:
        return uncertain("predicate_not_evaluable", f"can't compute {p.fn}: {type(e).__name__}: {e}")
    pos_of = {bid: i for i, bid in enumerate(before.index)}
    pairs = [(b, a) for b, a in zip(al.before_ids, al.after_pos) if mask is None or mask[pos_of[b]]]
    bids = [b for b, _ in pairs]
    expv = [exp_all[pos_of[b]] for b in bids]
    got = as_objects(after[p.target].iloc[[a for _, a in pairs]])
    bad = _changed_rows(expv, got, bids, p.atol if p.atol is not None else DEFAULT_ATOL)
    exp = f"{p.target} = {p.fn}(...)" + (" where condition" if p.where is not None else "")
    if bad:
        pos = {b: i for i, b in enumerate(bids)}
        ex = bad[0]
        return violated(exp, f"{len(bad)} of {len(bids)} rows differ",
                        f"e.g. row {ex}: expected {expv[pos[ex]]!r}, got {got[pos[ex]]!r}", bad)
    return holds(exp, 0)


def columns_added(p, before, after, al):
    pre = [c for c in p.target if c in before.columns]
    if pre:
        return uncertain("predicate_not_evaluable", f"column(s) {pre} already exist in the input")
    missing = [c for c in p.target if c not in after.columns]
    if missing:
        return violated(f"columns {p.target} added", f"missing {missing}", f"column(s) {missing} not added")
    return holds(f"columns {p.target} added", p.target)


def columns_removed(p, before, after, al):
    still = [c for c in p.target if c in after.columns]
    if still:
        return violated(f"columns {p.target} removed", f"still present {still}", f"column(s) {still} not removed")
    return holds(f"columns {p.target} removed", p.target)


CHECKS = {
    "row_count": row_count, "rows_removed_where": rows_removed_where, "rows_retained_where": rows_retained_where,
    "duplicate_count": duplicate_count, "null_count": null_count, "column_values_unchanged": column_values_unchanged,
    "columns_unchanged": columns_unchanged, "column_dtype": column_dtype,
    "column_value_relation": column_value_relation, "columns_added": columns_added, "columns_removed": columns_removed,
}
