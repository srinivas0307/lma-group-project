"""Match after-rows to before-rows (§5.2, README G9).

The before index (0..n-1 as loaded) is the row id.
1. Index alignment when the after index can be trusted: unique, same number of levels, every label is a
   before label, and not "renumbered" (exactly 0..m-1 with m < n while the rows don't agree with those labels).
2. Otherwise content alignment on the key columns (shared columns the contract says are unchanged; if none,
   all shared columns). Before rows that are identical in *every* before column are interchangeable, so any
   of them may be picked. If after rows match several before rows that differ elsewhere -> ambiguous.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .values import equal_mask, row_keys

AGREE_ROW = 0.5        # a row agrees if more than half of the key columns are equal
AGREE_FRAC = 0.9       # renumbered index is trusted only if this share of rows agree


@dataclass
class Alignment:
    method: str                                   # "index" | "content"
    before_ids: list                              # surviving before ids, one per matched after row
    after_pos: list                               # positional index into `after`, same order
    removed: list = field(default_factory=list)   # before ids with no match
    added_pos: list = field(default_factory=list)  # after positions with no match
    ambiguous: int = 0                            # after rows that match several distinguishable before rows
    equiv: dict = field(default_factory=dict)     # before id -> frozenset of interchangeable before ids

    @property
    def surviving(self) -> set:
        return set(self.before_ids)

    def survives_any(self, ids) -> bool:
        s = self.surviving
        return any(i in s for i in ids)

    def ok(self) -> bool:
        return self.ambiguous == 0


def _row_agreement(before, after, key_cols) -> float:
    if not key_cols or len(after) == 0:
        return 1.0
    b = before.loc[after.index, key_cols]
    agree = [equal_mask(b[c].astype(object).tolist(), after[c].astype(object).tolist()) for c in key_cols]
    per_row = sum(a.astype(int) for a in agree) / len(key_cols)
    return float((per_row > AGREE_ROW).mean())


def _index_trusted(before, after, key_cols) -> bool:
    """Labels must be unique, same depth, all before labels. An index that is exactly 0..m-1 may be a
    reset_index() after removing or reordering rows, so it is trusted only if the rows agree with the before
    rows those labels point to. Original (non-renumbered) labels are always trusted, so in-place damage to
    values shows up as a violation instead of being hidden by re-matching."""
    ai, bi = after.index, before.index
    if ai.nlevels != bi.nlevels or not ai.is_unique:
        return False
    try:
        if not ai.isin(bi).all():
            return False
    except (TypeError, ValueError):
        return False
    renumbered = len(after) > 0 and list(ai) == list(range(len(after)))
    if renumbered:
        return _row_agreement(before, after, key_cols) >= AGREE_FRAC
    return True


def align(before: pd.DataFrame, after: pd.DataFrame, key_cols=None) -> Alignment:
    shared = [c for c in before.columns if c in after.columns]
    keys = [c for c in (key_cols or []) if c in shared] or shared
    if _index_trusted(before, after, keys):
        ids = list(after.index)
        return Alignment("index", ids, list(range(len(after))),
                         removed=[i for i in before.index if i not in set(ids)])
    return _content_align(before, after, keys)


def _content_align(before, after, keys) -> Alignment:
    bkeys = row_keys(before, keys)
    akeys = row_keys(after, keys)
    full = row_keys(before, list(before.columns))
    groups = defaultdict(list)                     # key -> before ids in index order
    for bid, k in zip(before.index, bkeys):
        groups[k].append(bid)
    full_of = dict(zip(before.index, full))
    equiv = {}
    for ids in groups.values():
        by_full = defaultdict(list)
        for i in ids:
            by_full[full_of[i]].append(i)
        for same in by_full.values():
            fs = frozenset(same)
            for i in same:
                equiv[i] = fs
    taken = defaultdict(int)
    before_ids, after_pos, added, ambiguous = [], [], [], 0
    for pos, k in enumerate(akeys):
        ids = groups.get(k)
        if not ids or taken[k] >= len(ids):
            added.append(pos)
            continue
        distinct = {full_of[i] for i in ids}
        if len(distinct) > 1 and sum(1 for p in akeys if p == k) < len(ids):
            ambiguous += 1                         # some, not all, of several different rows survived
        before_ids.append(ids[taken[k]])
        after_pos.append(pos)
        taken[k] += 1
    matched = set(before_ids)
    return Alignment("content", before_ids, after_pos, removed=[i for i in before.index if i not in matched],
                     added_pos=added, ambiguous=ambiguous, equiv=equiv)
