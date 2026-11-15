"""Input-only contract diagnostics, version contract-check-v1.

Receives only the request text, the input DataFrame and the predicted contract. It never sees reference
code, reference outputs, gold contracts or scores.

Every issue has a ``kind``:
- ``proof``: the contract is internally inconsistent with the input data. A contract that still has a proof issue
  after the single correction is withheld.
- ``evidence``: suspicious, but a legitimate request could produce it (for example an empty filter). Evidence
  triggers the correction prompt but never causes withholding by itself.
"""
import difflib
import re

import numpy as np
import pandas as pd

from finetune.selective_facts_correction.checks import diagnose as base_diagnose
from igt.contracts.schema import ContractError, expand_columns, load_contract
from igt.verifier.conditions import ConditionError, evaluate
from igt.verifier.functions import expected_values
from igt.verifier.predicates import _count_ok
from igt.verifier.values import DEFAULT_ATOL, dtype_family, is_null, row_keys, value_equal

VERSION = 'contract-check-v1'
MAX_LISTED = 8
ORDER_OPS = {'<', '<=', '>', '>='}
BASE_PROOF = {'schema', 'unknown_column', 'existing_column_added', 'change_preserve_conflict'}
# Input-family -> declared-family pairs whose cell values cannot stay equal under the verifier's value comparison.
VALUE_CHANGING = {('string', 'datetime'), ('object', 'datetime'), ('int', 'datetime'), ('float', 'datetime'),
                  ('int', 'bool'), ('float', 'bool'), ('string', 'bool'), ('object', 'bool'),
                  ('int', 'string'), ('float', 'string'), ('bool', 'string'), ('datetime', 'string'),
                  ('string', 'int'), ('string', 'float'), ('object', 'int'), ('object', 'float')}


def _short(values, n=MAX_LISTED):
    values = list(values)
    return values[:n] + (['...'] if len(values) > n else [])


def _leaves(cond, path):
    if cond is None:
        return
    if cond.column is not None:
        yield path, cond
        return
    for key, kids in (('all', cond.all), ('any', cond.any)):
        for i, kid in enumerate(kids or []):
            yield from _leaves(kid, f'{path}.{key}[{i}]')
    if cond.not_ is not None:
        yield from _leaves(cond.not_, path + '.not')


def _conditions(pid, p):
    for key in ('condition', 'where'):
        cond = getattr(p, key, None)
        if cond is not None:
            yield f'{pid}.{key}', cond


def _is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _textual(s):
    return dtype_family(s) in ('string', 'object', 'category')


def _present(value, series_values):
    return any(value_equal(value, x) for x in series_values)


def _columns_containing(values, before, skip):
    """Other text columns in which every requested literal occurs exactly."""
    hits = []
    for c in before.columns:
        if c == skip or not _textual(before[c]):
            continue
        uniq = set(before[c].dropna().astype(str))
        if all(str(v) in uniq for v in values):
            hits.append(c)
    return hits


def _request_columns(request, before):
    """Input columns whose name or name words appear in the request (hints only)."""
    words = set(re.findall(r'[a-z0-9]+', request.lower()))
    out = []
    for c in before.columns:
        parts = [w for w in re.split(r'[^a-z0-9]+', str(c).lower()) if w and w not in {'id', 'num', 'count'}]
        if str(c).lower() in request.lower() or (parts and any(w in words for w in parts)):
            out.append(c)
    return out


def diagnose(request, extracted, before):
    """List of issues {code, kind, detail}. Empty list means no check fired."""
    issues = []

    def issue(code, kind, detail):
        item = dict(code=code, kind=kind, detail=detail)
        if item not in issues:
            issues.append(item)

    for item in base_diagnose(extracted, before):
        issue(item['code'], 'proof', item['detail'])
    if extracted.get('status') != 'ok' or any(x['code'] in ('schema', 'unknown_column') for x in issues):
        return issues
    try:
        contract = load_contract(extracted['contract'])
    except ContractError as exc:
        issue('schema', 'proof', str(exc)[:1600])
        return issues
    preds = contract.predicates()
    change = [(pid, p) for d, pid, p in preds if d == 'must_change']
    removal = any(p.type in ('rows_removed_where', 'duplicate_count') for _, _, p in preds)
    rows_fixed = not removal and any(p.type == 'row_count' and p.op == 'unchanged' for _, _, p in preds)
    must_decrease = any(p.type == 'row_count' and p.op in ('decreased', 'lt') for _, p in change)

    # 1. Conditions: evaluability, literal types and literal presence.
    for _, pid, p in preds:
        for where, cond in _conditions(pid, p):
            try:
                evaluate(cond, before)
            except ConditionError as exc:
                issue('condition_not_evaluable', 'proof', f'{where}: {exc}. Use values that exist in this column, '
                      'or a different column whose type matches the comparison.')
                continue
            for path, leaf in _leaves(cond, where):
                s = before[leaf.column]
                values = leaf.value if leaf.op in ('in', 'not_in') else [leaf.value]
                if leaf.op in ('is_null', 'not_null'):
                    continue
                if _textual(s) and leaf.op not in ORDER_OPS and any(_is_number(v) for v in values):
                    issue('literal_type_mismatch', 'evidence', f'{path}: text column {leaf.column!r} is compared with '
                          f'number(s) {_short(values)}. Text cells never equal numbers.')
                    continue
                strings = [v for v in values if isinstance(v, str)]
                # A threshold in an ordering comparison need not occur in the data; only equality/membership.
                if leaf.op in ORDER_OPS or not strings or not _textual(s):
                    continue
                uniq = s.dropna().astype(object).unique().tolist()
                absent = [v for v in strings if not _present(v, uniq)]
                if not absent:
                    continue
                elsewhere = _columns_containing(absent, before, leaf.column)
                near = sorted({m for v in absent for m in difflib.get_close_matches(v, [str(u) for u in uniq[:5000]],
                                                                                     n=3, cutoff=.6)})
                issue('literal_absent', 'evidence', f'{path}: value(s) {_short(absent)} never occur in column '
                      f'{leaf.column!r}. Columns that contain all of them: {elsewhere}. Similar values in '
                      f'{leaf.column!r}: {_short(near)}. Check which column and values the request refers to.')

    # 2. Removal predicates that select nothing.
    for _, pid, p in preds:
        if p.type != 'rows_removed_where':
            continue
        try:
            n = int(evaluate(p.condition, before).sum())
        except ConditionError:
            continue
        if n == 0:
            kind = 'proof' if must_decrease else 'evidence'
            issue('removal_matches_nothing', kind, f'{pid}: the removal condition matches 0 of {len(before)} input rows'
                  + (', but the contract also requires the row count to decrease.' if must_decrease else '.'))

    # 3. Deduplication keys that have no duplicates.
    for pid, p in change:
        if p.type != 'duplicate_count':
            continue
        keys = expand_columns(p.target, p.except_, before.columns)
        dups = int(pd.Series(row_keys(before, keys)).duplicated().sum()) if len(before) else 0
        if dups == 0:
            hint = _request_columns(request, before)
            issue('dedup_key_has_no_duplicates', 'proof' if must_decrease else 'evidence',
                  f'{pid}: the input has no duplicate rows on key {keys if len(keys) < 6 else "all columns"}, so '
                  'deduplicating on it removes nothing' + (' although the row count must decrease' if must_decrease
                                                           else '') +
                  f'. Input columns named in the request (suggestions only): {hint}.')

    # 4. Mapping literals that do not match the input values.
    for pid, p in change:
        if p.type != 'column_value_relation' or p.fn != 'map' or p.args[0].col not in before.columns:
            continue
        src = p.args[0].col
        uniq = before[src].dropna().astype(object).unique().tolist()
        keys = list(p.args[1].lit)
        unmatched = [k for k in keys if not any(value_equal(k, v) or str(v) == str(k) for v in uniq)]
        unmapped = [v for v in uniq if v not in p.args[1].lit and str(v) not in p.args[1].lit]
        # An extra key for a value absent from this sample is harmless when every input value is covered.
        if unmatched and unmapped:
            issue('map_key_absent', 'evidence', f'{pid}: mapping key(s) {_short(unmatched)} never occur in input '
                  f'column {src!r}. Keys are matched literally, so they never apply (there is no default branch). '
                  f'Input values not covered by the mapping keep their original value: {_short(unmapped)}.')

    # 5. Change predicates that change nothing.
    for pid, p in change:
        if p.type != 'column_value_relation' or p.target not in before.columns:
            continue
        try:
            active = evaluate(p.where, before) if p.where is not None else np.ones(len(before), dtype=bool)
            expected = expected_values(p, before)
        except (ConditionError, ValueError, TypeError, KeyError, OverflowError):
            continue
        original = before[p.target].astype(object).tolist()
        tol = DEFAULT_ATOL if p.atol is None else p.atol
        changed = [i for i in np.flatnonzero(active) if not value_equal(expected[i], original[i], atol=tol)]
        if active.any() and not changed:
            issue('change_is_noop', 'evidence', f'{pid}: this must_change relation leaves every one of the '
                  f'{int(active.sum())} selected {p.target!r} values unchanged. Check the function and literals.')

    # 6. Declared type changes that contradict exact value preservation.
    for pid, p in change:
        if p.type != 'column_dtype' or p.target not in before.columns:
            continue
        pair = (dtype_family(before[p.target]), p.dtype)
        if pair not in VALUE_CHANGING:
            continue
        for _, qid, q in preds:
            if q.type in ('column_values_unchanged', 'columns_unchanged') and \
                    p.target in expand_columns(q.target, q.except_, before.columns):
                issue('dtype_vs_value_preservation', 'proof', f'{pid} converts {p.target!r} from {pair[0]} to '
                      f'{pair[1]}, which changes every cell value, but {qid} requires the same cells unchanged. '
                      'Express the conversion itself (for example a cast relation) instead of value preservation.')

    # 7. Null-count requirements that the contract's own value requirements contradict.
    if rows_fixed:
        for _, pid, p in preds:
            if p.type != 'null_count' or p.target not in before.columns:
                continue
            result = _projected_nulls(p.target, preds, before)
            if result is None:
                continue
            after, fully = result
            nb = int(before[p.target].isna().sum())
            if not _count_ok(p.op, p.value, after, nb):
                expect = p.op if p.value is None else f'{p.op} {p.value}'
                issue('null_count_contradiction', 'proof' if fully else 'evidence',
                      f'{pid} requires null_count({p.target}) {expect} (input has {nb}), but the contract\'s value '
                      f'requirements give {after} missing values' + ('' if fully else
                                                                     ' when unspecified cells keep their values') +
                      '. Make the null requirement agree with the requested fill/replacement.')
    return issues


def _projected_nulls(target, preds, before):
    """(null count after, fully_determined) implied by relations and preservation on an unchanged row set."""
    n = len(before)
    known = np.zeros(n, dtype=bool)
    nulls = before[target].isna().to_numpy().copy()
    try:
        for _, _, q in preds:
            if q.type == 'columns_unchanged' and target in expand_columns(q.target, q.except_, before.columns):
                known[:] = True
            elif q.type == 'column_values_unchanged' and target in expand_columns(q.target, q.except_, before.columns):
                known |= evaluate(q.where, before) if q.where is not None else np.ones(n, dtype=bool)
        for _, _, q in preds:
            if q.type == 'column_value_relation' and q.target == target:
                mask = evaluate(q.where, before) if q.where is not None else np.ones(n, dtype=bool)
                values = expected_values(q, before)
                for i in np.flatnonzero(mask):
                    nulls[i] = is_null(values[i])
                known |= mask
    except (ConditionError, ValueError, TypeError, KeyError, OverflowError):
        return None
    return int(nulls.sum()), bool(known.all())


def summary_line(issues):
    return ', '.join(f"{x['code']}({x['kind']})" for x in issues) or 'none'
