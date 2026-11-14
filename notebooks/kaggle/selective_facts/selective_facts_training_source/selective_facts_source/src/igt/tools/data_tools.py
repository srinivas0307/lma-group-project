"""The 5 agent tools (README §4.5): inspect_dataset, profile_column, diff_datasets, run_transformation, run_verifier.

All tools work on a ToolContext that holds the READ-ONLY before snapshot and contract (README §4.8 invariant 1):
tools only ever get copies. Code runs only through the safety checker + sandbox (invariant 4).

    ctx = ToolContext(before=df, verify_fn=lambda after: verify(contract, df, after)[0], sandbox_limits=...)
    reg = build_registry(ctx)
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from igt.sandbox import check, safe_run
from igt.verifier.alignment import align
from igt.verifier.values import as_objects, dtype_family, equal_mask

from .registry import Tool, ToolRegistry

MAX_ROWS = 20


def _jsonable(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v if isinstance(v, (str, int, float, bool)) else str(v)


def _records(df, n):
    return [{str(k): _jsonable(v) for k, v in row.items()} | {"_row_id": _jsonable(i)}
            for i, row in df.head(n).iterrows()]


@dataclass
class ToolContext:
    before: pd.DataFrame
    verify_fn: Callable                       # after DataFrame -> Verdict (the system's configured verifier)
    sandbox_limits: dict = field(default_factory=lambda: dict(timeout_s=30, cpu_s=20, mem_gb=2))
    func_name: str = "transform"
    last_output: pd.DataFrame | None = None   # output of the latest run_transformation
    last_code: str | None = None
    safety_log: list = field(default_factory=list)   # every blocked run (safety metrics)

    def __post_init__(self):
        self._before = self.before.copy()       # private copy: the caller's frame can't be changed through tools

    def frame(self, which):
        if which == "before":
            return self._before.copy()
        if self.last_output is None:
            raise ValueError("no transformation has been run yet; call run_transformation first")
        return self.last_output.copy()


def build_registry(ctx: ToolContext) -> ToolRegistry:
    which = {"type": "string", "enum": ["before", "last_output"]}

    def inspect_dataset(which="before", n_rows=5, columns=None):
        df = ctx.frame(which)
        cols = columns or [str(c) for c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"unknown columns {missing}")
        return dict(which=which, n_rows=len(df), n_cols=len(df.columns),
                    columns=[dict(name=str(c), dtype=str(df[c].dtype), family=dtype_family(df[c]),
                                  nulls=int(df[c].isna().sum())) for c in df.columns],
                    head=_records(df[cols], min(n_rows, MAX_ROWS)))

    def profile_column(column, which="before", top=10):
        df = ctx.frame(which)
        if column not in df.columns:
            raise KeyError(f"column '{column}' not in {which} (columns: {[str(c) for c in df.columns][:30]})")
        s = df[column]
        out = dict(which=which, column=column, dtype=str(s.dtype), family=dtype_family(s), n=len(s),
                   nulls=int(s.isna().sum()), distinct=int(s.nunique(dropna=True)))
        vc = s.astype(object).value_counts(dropna=True).head(min(top, MAX_ROWS))
        out["top_values"] = [dict(value=_jsonable(k), count=int(v)) for k, v in vc.items()]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.dropna()
            if len(d):
                out.update(min=_jsonable(d.min()), max=_jsonable(d.max()), mean=round(float(d.mean()), 6),
                           median=_jsonable(d.median()))
        if isinstance(s.dtype, pd.CategoricalDtype):
            out.update(ordered=bool(s.cat.ordered), categories=[_jsonable(c) for c in s.cat.categories][:50])
        return out

    def diff_datasets(max_examples=5):
        if ctx.last_output is None:
            raise ValueError("nothing to compare; call run_transformation first")
        b, a = ctx._before, ctx.last_output
        al = align(b, a, [c for c in b.columns if c in a.columns])
        shared = [c for c in b.columns if c in a.columns]
        pos = dict(zip(al.after_pos, al.before_ids))
        changed, dtype_changes = {}, {}
        for c in shared:
            bv = as_objects(b.loc[al.before_ids, c])
            av = as_objects(a.iloc[al.after_pos][c])
            neq = ~equal_mask(bv, av)
            if neq.any():
                ids = [al.before_ids[i] for i in np.flatnonzero(neq)][:max_examples]
                changed[str(c)] = dict(n_cells=int(neq.sum()), examples=[
                    dict(row_id=_jsonable(i), before=_jsonable(b.at[i, c]),
                         after=_jsonable(a.iloc[al.after_pos[al.before_ids.index(i)]][c])) for i in ids])
            if dtype_family(b[c]) != dtype_family(a[c]):
                dtype_changes[str(c)] = f"{dtype_family(b[c])} -> {dtype_family(a[c])}"
        return dict(alignment=al.method, rows_before=len(b), rows_after=len(a),
                    rows_removed=len(al.removed), removed_examples=[_jsonable(i) for i in al.removed[:max_examples]],
                    rows_added=len(al.added_pos), ambiguous_rows=al.ambiguous,
                    columns_added=[str(c) for c in a.columns if c not in b.columns],
                    columns_removed=[str(c) for c in b.columns if c not in a.columns],
                    changed_cells=changed, dtype_changes=dtype_changes)

    def run_transformation(code):
        g = check(code)
        if not g.ok:
            ctx.safety_log.append(dict(tool="run_transformation", reason=g.reason))
            return dict(status="blocked", error=f"blocked by the safety checker: {g.reason}")
        r = safe_run(code, ctx._before, ctx.func_name, **ctx.sandbox_limits)
        if not r.ran_ok:
            return dict(status=r.status, error=r.error)
        ctx.last_output, ctx.last_code = r.df_out, code
        return dict(status="ok", n_rows=len(r.df_out), n_cols=len(r.df_out.columns),
                    columns=[str(c) for c in r.df_out.columns], head=_records(r.df_out, 3))

    def run_verifier():
        if ctx.last_output is None:
            raise ValueError("nothing to verify; call run_transformation first")
        v = ctx.verify_fn(ctx.last_output.copy())
        return dict(status=v.status, report=v.explain(), n_failures=len(v.failures))

    return ToolRegistry([
        Tool("inspect_dataset", "Shape, columns (dtype, missing values) and the first rows of the input data "
             "('before') or of the last transformation output ('last_output').",
             {"type": "object", "properties": {"which": which, "n_rows": {"type": "integer", "minimum": 0, "maximum": MAX_ROWS},
                                               "columns": {"type": "array", "items": {"type": "string"}}}, "required": []},
             inspect_dataset),
        Tool("profile_column", "Statistics of one column: dtype, missing values, distinct count, most common values, "
             "min/max/mean for numbers, categories for category columns.",
             {"type": "object", "properties": {"column": {"type": "string"}, "which": which,
                                               "top": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS}},
              "required": ["column"]}, profile_column),
        Tool("diff_datasets", "Compare the input with the last transformation output: rows removed/added, columns "
             "added/removed, changed cells per column with examples, dtype changes.",
             {"type": "object", "properties": {"max_examples": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS}},
              "required": []}, diff_datasets),
        Tool("run_transformation", "Run Python code defining transform(df) on a copy of the input (safety checker + "
             "sandbox). Stores the result as 'last_output'. Does not verify it.",
             {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}, run_transformation),
        Tool("run_verifier", "Check the last transformation output with the system's verifier and return its report.",
             {"type": "object", "properties": {}, "required": []}, run_verifier),
    ])
