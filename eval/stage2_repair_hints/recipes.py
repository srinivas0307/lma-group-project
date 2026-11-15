"""Deterministic pandas recipes derived only from the predicted contract and the input schema.

Each recipe states, in pandas, what one contract predicate means under the verifier's semantics
(src/igt/verifier): conditions and relation inputs are evaluated on the ORIGINAL input rows, a null
cell makes every comparison false, keep='first'/'last' refers to input order, and round is
Series.round. Recipes never see reference code, reference output or gold contracts.

`assemble(contract, columns)` composes all recipes into one transform(df). It is used offline to
prove recipe semantics (it must pass the verifier on contracts it fully covers); the runtime only shows
per-predicate recipes as hints.
"""
import json

from igt.contracts.schema import ColArg, expand_columns, load_contract

VERSION = 'contract-recipes-v1'
ORDER_OPS = ('<', '<=', '>', '>=')


class Unsupported(ValueError):
    """The predicate has no faithful single-statement recipe (for example an unspecified fill rule)."""


def lit(value):
    """Python source for a JSON literal (None/True/False instead of null/true/false)."""
    return repr(value)


def col(name, frame='df'):
    return f'{frame}[{name!r}]'


def mask(cond, frame='df'):
    """Boolean Series source equal to igt.verifier.conditions.evaluate on `frame`."""
    if cond.all is not None:
        return '(' + ' & '.join(mask(k, frame) for k in cond.all) + ')'
    if cond.any is not None:
        return '(' + ' | '.join(mask(k, frame) for k in cond.any) + ')'
    if cond.not_ is not None:
        return '~' + mask(cond.not_, frame)
    c, op, v = col(cond.column, frame), cond.op, cond.value
    if op == 'is_null':
        return f'{c}.isna()'
    if op == 'not_null':
        return f'{c}.notna()'
    if op == 'in':
        return f'{c}.isin({lit(v)})'
    if op == 'not_in':
        # pandas would keep nulls in ~isin; the contract treats a null cell as not matching.
        return f'({c}.notna() & ~{c}.isin({lit(v)}))'
    if op == '!=':
        return f'({c}.notna() & ({c} != {lit(v)}))'
    return f'({c} {op} {lit(v)})'


def _arg(a, frame='df'):
    return col(a.col, frame) if isinstance(a, ColArg) else lit(a.lit)


def relation_expr(p, frame='df'):
    """Series (or scalar) source for column_value_relation values computed from `frame` (the input rows)."""
    fn, args = p.fn, p.args
    a = [_arg(x, frame) for x in args]
    if fn == 'identity':
        return a[0]
    if fn == 'const':
        return a[0]
    if fn == 'null':
        return 'None'
    if fn in ('add', 'sub', 'mul'):
        return f'{a[0]} {dict(add="+", sub="-", mul="*")[fn]} {a[1]}'
    if fn == 'div':
        if isinstance(args[1], ColArg):
            return f'({a[0]} / {a[1]}).where({a[1]} != 0)'
        if args[1].lit == 0:
            raise Unsupported('division by literal zero')
        return f'{a[0]} / {a[1]}'
    if fn == 'round':
        return f'{a[0]}.round({int(args[1].lit)})'
    if fn == 'clip':
        lo, hi = args[1].lit, args[2].lit
        bounds = ', '.join(x for x in [f'lower={lit(lo)}' if lo is not None else '',
                                       f'upper={lit(hi)}' if hi is not None else ''] if x)
        return f'{a[0]}.clip({bounds})'
    if fn in ('lower', 'upper', 'strip'):
        return f'{a[0]}.map(lambda v: v.{fn}() if isinstance(v, str) else v)'
    if fn == 'map':
        mapping = args[1].lit
        return (f'{a[0]}.map(lambda v, m={lit(mapping)}: v if pd.isna(v) else m.get(v, m.get(str(v), v)))')
    if fn == 'cast':
        dtype = args[1].lit
        fmt = args[2].lit if len(args) > 2 else None
        if dtype == 'int':
            return f'pd.to_numeric({a[0]}, errors="coerce").astype("Int64")'
        if dtype == 'float':
            return f'pd.to_numeric({a[0]}, errors="coerce").astype(float)'
        if dtype == 'string':
            return f'{a[0]}.map(lambda v: v if pd.isna(v) or isinstance(v, str) else str(v))'
        if dtype == 'datetime':
            return f'pd.to_datetime({a[0]}, format={lit(fmt or "ISO8601")}, errors="coerce")'
        if dtype == 'category':
            return f'{a[0]}.astype("category")'
        if dtype == 'bool':
            return (f'{a[0]}.map(lambda v: v if pd.isna(v) else ({{"true": True, "1": True, "yes": True, '
                    f'"false": False, "0": False, "no": False}}.get(v.strip().lower()) if isinstance(v, str) else bool(v)))')
    raise Unsupported(f'function {fn}')


def dtype_stmt(p, out='out'):
    t = col(p.target, out)
    if p.dtype == 'category':
        if p.categories is not None or p.ordered is not None:
            parts = [t]
            if p.categories is not None:
                parts.append(f'categories={lit(p.categories)}')
            if p.ordered is not None:
                parts.append(f'ordered={p.ordered}')
            return f'{t} = pd.Categorical({", ".join(parts)})'
        return f'{t} = {t}.astype("category")'
    return {'int': f'{t} = pd.to_numeric({t}, errors="coerce").astype("Int64")',
            'float': f'{t} = pd.to_numeric({t}, errors="coerce").astype(float)',
            'string': f'{t} = {t}.map(lambda v: v if pd.isna(v) or isinstance(v, str) else str(v))',
            'datetime': f'{t} = pd.to_datetime({t}, errors="coerce")',
            'bool': f'{t} = {t}.astype("boolean")'}[p.dtype]


def _keys(p, columns):
    return expand_columns(p.target, p.except_, columns)


def recipe(direction, p, columns):
    """One short instruction plus pandas lines for a single predicate. `df` is the unmodified input,
    `out` the frame being built (`out = df.copy()` first)."""
    t = p.type
    if direction == 'must_change':
        if t == 'rows_removed_where':
            return dict(note='Remove exactly the input rows matching the condition (null cells never match). '
                             'Build the mask from the ORIGINAL df; do not convert or overwrite columns to filter.',
                        code=[f'remove = {mask(p.condition)}', 'out = out[~remove]'])
        if t == 'duplicate_count' and p.op == 'eq' and p.value == 0:
            keys = _keys(p, columns)
            keep = p.keep or 'first'
            return dict(note=f"Keep one row per {keys} group: the {keep} one in ORIGINAL input order. "
                             'Do not sort, dropna or reset_index before deduplicating.',
                        code=[f'out = out[~df.duplicated(subset={lit(keys)}, keep={keep!r})]'])
        if t == 'column_value_relation':
            expr = relation_expr(p)
            target = col(p.target, 'out')
            if p.where is None:
                return dict(note=f'Set {p.target} from the input values.', code=[f'{target} = {expr}'])
            where = mask(p.where)
            value = expr if p.fn in ('const', 'null') else f'({expr})[where]'
            return dict(note=f'Change {p.target} only on input rows matching the condition; leave other rows as they were.',
                        code=[f'where = {where}', f'out.loc[where, {p.target!r}] = {value}'])
        if t == 'column_dtype':
            return dict(note=f'Assign the converted column back (category methods return a new object).',
                        code=[dtype_stmt(p)])
        if t == 'columns_removed':
            return dict(note='Drop only these columns.', code=[f'out = out.drop(columns={lit(list(p.target))})'])
        if t == 'columns_added':
            return dict(note=f'Create {list(p.target)} and keep every input column; no helper columns in the output.', code=[])
        if t == 'null_count':
            want = 'no missing values' if p.op == 'eq' and p.value == 0 else f'missing-value count {p.op} {p.value if p.value is not None else "input"}'
            return dict(note=f'{p.target} must end with {want}; fill using the rule in the request, '
                             'computed from the original input rows, and do not drop rows to achieve it.', code=[])
        if t == 'row_count':
            return dict(note=f'Row count must be {p.op}{"" if p.value is None else " " + str(p.value)}; '
                             'only the removals/deduplication stated in the contract may change it.', code=[])
    else:
        if t in ('columns_unchanged', 'column_values_unchanged'):
            keys = _keys(p, columns)
            scope = 'all other columns' if p.target == '*' else str(keys)
            extra = ' Keep each column\'s dtype: compare against a temporary converted Series instead of reassigning.' \
                if t == 'columns_unchanged' else ''
            return dict(note=f'Leave {scope} exactly as in the input (no drop, rename, fillna or conversion).{extra}', code=[])
        if t == 'row_count' and p.op == 'unchanged':
            return dict(note='Keep every input row: no dropna, drop_duplicates or filtering.', code=[])
        if t == 'rows_retained_where':
            return dict(note='These input rows must remain (rows with nulls included unless the condition says is_null):',
                        code=[f'keep = {mask(p.condition)}'])
        if t == 'null_count' and p.op == 'unchanged':
            return dict(note=f'Do not fill or create missing values in {p.target}.', code=[])
        if t == 'column_dtype':
            return dict(note=f'{p.target} must stay dtype {p.dtype}.', code=[])
        if t == 'column_value_relation':
            return dict(note=f'{p.target} must equal this after the transformation:',
                        code=[f'{col(p.target, "out")} == {relation_expr(p)}'])
    raise Unsupported(f'{direction} {t}')


def recipes(contract, columns):
    """{predicate_id: recipe or {'unsupported': reason}} for every predicate."""
    parsed = load_contract(contract) if isinstance(contract, dict) else contract
    value_rules = _conversion_value_rules(parsed)
    result = {}
    for direction, pid, p in parsed.predicates():
        try:
            r = recipe(direction, p, columns)
            if direction == 'must_change' and p.type == 'column_dtype' and p.target in value_rules:
                r['code'] = [f'{col(p.target, "out")} = {relation_expr(value_rules[p.target])}'] + r['code']
                r['note'] = 'Convert the values with the contract mapping first, then set the dtype; assign both back.'
            result[pid] = dict(direction=direction, type=p.type, **r)
        except Unsupported as exc:
            result[pid] = dict(direction=direction, type=p.type, unsupported=str(exc))
    return result


def _conversion_value_rules(parsed):
    """must_preserve value relations on a column whose dtype must change: the conversion's value rule."""
    converted = {p.target for d, _, p in parsed.predicates(('must_change',)) if p.type == 'column_dtype'}
    return {p.target: p for d, _, p in parsed.predicates(('must_preserve',))
            if p.type == 'column_value_relation' and p.target in converted and p.where is None}


def render(items, limit=2500):
    lines = []
    for pid, r in items.items():
        if 'unsupported' in r:
            continue
        lines.append(f'- {pid} ({r["type"]}): {r["note"]}')
        lines += [f'    {c}' for c in r['code']]
    text = '\n'.join(lines)
    return text if len(text) <= limit else text[:limit].rsplit('\n', 1)[0] + '\n    ...'


# ---------------- offline semantics proof ----------------
def assemble(contract, columns):
    """A full transform(df) composed from the recipes, or raise Unsupported.

    Order: column changes and dtypes on `out` (values from the input df), then removals and deduplication
    using masks computed on the input df, then dropped columns. Raises Unsupported when a must_change
    predicate is only a constraint (null_count with no relation, columns_added with no relation).
    """
    parsed = load_contract(contract) if isinstance(contract, dict) else contract
    changes = [p for d, _, p in parsed.predicates(('must_change',))]
    related = {p.target for p in changes if p.type == 'column_value_relation'}
    body, row_masks, drops = ['out = df.copy()'], [], []
    for p in changes:
        if p.type == 'column_value_relation':
            body += recipe('must_change', p, columns)['code']
    value_rules = _conversion_value_rules(parsed)
    for p in changes:
        if p.type == 'column_dtype':
            if p.target in value_rules and p.target not in related:
                body.append(f'{col(p.target, "out")} = {relation_expr(value_rules[p.target])}')
            body.append(dtype_stmt(p))
    for i, p in enumerate(changes):
        if p.type == 'rows_removed_where':
            row_masks.append(f'~{mask(p.condition)}')
        elif p.type == 'duplicate_count':
            if not (p.op == 'eq' and p.value == 0):
                raise Unsupported('duplicate_count without eq 0')
            row_masks.append(f'~df.duplicated(subset={lit(_keys(p, columns))}, keep={(p.keep or "first")!r})')
        elif p.type == 'columns_removed':
            drops += list(p.target)
        elif p.type == 'columns_added':
            missing = [c for c in p.target if c not in related]
            if missing:
                raise Unsupported(f'added columns without a value rule: {missing}')
        elif p.type == 'null_count' and p.target not in related:
            raise Unsupported(f'null_count on {p.target} without a value rule')
    if row_masks:
        body.append('out = out[' + ' & '.join(f'({m})' for m in row_masks) + ']')
    if drops:
        body.append(f'out = out.drop(columns={lit(drops)})')
    body.append('return out')
    return 'import pandas as pd\n\ndef transform(df):\n' + '\n'.join('    ' + line for line in body) + '\n'


def describe(contract):
    return json.dumps(contract, sort_keys=True)
