"""Intent verifier: check every predicate of a contract against before/after data -> Verdict.

    verdict, info = verify(contract_dict_or_Contract, before_df, after_df)

- Every predicate is checked (no early stop); results are combined with Verdict.combine:
  any violation -> VIOLATION, else any uncertain -> UNCERTAIN, else PASS.
- `directions` selects must_change and/or must_preserve (change-only / preserve-only ablations, README G4).
- UNCERTAIN reasons: invalid_contract, unknown_column, predicate_not_evaluable, rows_unmatched.
- Deterministic; never sees the reference solution.
"""
import pandas as pd

from igt.contracts.schema import Contract, ContractError, expand_columns, load_contract, predicate_columns
from igt.types import Verdict, predicate_failure, uncertain_failure

from .alignment import align
from .predicates import CHECKS

ROW_LEVEL = {"rows_removed_where", "rows_retained_where", "duplicate_count", "column_values_unchanged",
             "columns_unchanged", "column_value_relation"}       # need matched rows


def _key_columns(contract: Contract, before_cols) -> list:
    """Columns the contract says are unchanged (used to match rows by content when the index was lost)."""
    cols = []
    for _, _, p in contract.predicates():
        if p.type == "columns_unchanged" or (p.type == "column_values_unchanged" and p.where is None):
            cols += expand_columns(p.target, p.except_, before_cols)
    return list(dict.fromkeys(cols))


def verify(contract, before: pd.DataFrame, after: pd.DataFrame,
           directions=("must_change", "must_preserve")) -> tuple[Verdict, dict]:
    if not isinstance(contract, Contract):
        try:
            contract = load_contract(contract)
        except ContractError as e:
            return Verdict.combine([uncertain_failure("invalid_contract", str(e))]), dict(alignment=None)
    if not isinstance(after, pd.DataFrame):
        return Verdict.combine([uncertain_failure("predicate_not_evaluable", "output is not a DataFrame")]), {}
    after = after.loc[:, ~after.columns.duplicated()] if after.columns.duplicated().any() else after

    al = align(before, after, _key_columns(contract, before.columns))
    info = dict(alignment=al.method, n_predicates=0, ambiguous_rows=al.ambiguous, results=[])
    failures = []
    for direction, pid, p in contract.predicates(directions):
        info["n_predicates"] += 1
        unknown = [c for c in predicate_columns(p, before.columns) if c not in before.columns]
        if unknown:
            failures.append(uncertain_failure("unknown_column", f"{unknown} not in the input data", pid))
            info["results"].append((pid, "uncertain"))
            continue
        if p.type in ROW_LEVEL and not al.ok():
            failures.append(uncertain_failure(
                "rows_unmatched", f"{al.ambiguous} output rows match several different input rows", pid))
            info["results"].append((pid, "uncertain"))
            continue
        r = CHECKS[p.type](p, before, after, al)
        info["results"].append((pid, r.status))
        if r.violated:
            target = p.target if not hasattr(p, "except_") or not p.except_ else {"cols": p.target, "except": p.except_}
            failures.append(predicate_failure(pid, p.type, direction, target, r.expected, r.observed, r.detail,
                                              r.evidence_rows))
        elif r.uncertain:
            failures.append(uncertain_failure(r.reason, r.detail, pid))
    return Verdict.combine(failures), info
