"""Static pandas pitfalls in a failing candidate, reported only when the contract makes them relevant.

Inputs: candidate code, the predicted contract and the input columns. Each hint names the code pattern
and the contract predicate it conflicts with; it never states a reference answer.
"""
import ast

from igt.contracts.schema import expand_columns, load_contract

VERSION = 'repair-lint-v1'
RESULT_METHODS = {'set_categories', 'add_categories', 'remove_categories', 'rename_categories',
                  'reorder_categories', 'as_ordered', 'as_unordered', 'astype', 'fillna', 'drop_duplicates',
                  'dropna', 'round', 'replace', 'drop', 'rename', 'sort_values', 'reset_index', 'map', 'clip'}
CATEGORY_KWARGS = {'add_categories': set(), 'remove_categories': set(), 'rename_categories': set(),
                   'set_categories': {'ordered', 'rename'}, 'reorder_categories': {'ordered'}}


def _name(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _calls(tree):
    """(line, method name, call node) in source order."""
    return sorted(((n.lineno, _name(n.func), n) for n in ast.walk(tree) if isinstance(n, ast.Call)),
                  key=lambda x: (x[0], getattr(x[2], 'col_offset', 0)))


def _str_items(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _assigned_columns(tree):
    """Columns assigned with df['c'] = ... (any frame name), with the value node."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                    out.append((t.slice.value, n.value, n.lineno))
    return out


def _contains_call(node, names):
    return any(isinstance(n, ast.Call) and _name(n.func) in names for n in ast.walk(node))


def hints(code, contract, columns):
    """List of {code, line, message} for contract-relevant pitfalls in `code` (empty if it does not parse)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [dict(code='syntax_error', line=exc.lineno,
                     message=f'The code does not parse ({exc.msg}). Check brackets and write boolean masks as '
                             "df[(df['a'] == x) & (df['b'] == y)] with each comparison in parentheses.")]
    parsed = load_contract(contract) if isinstance(contract, dict) else contract
    preds = [(d, p) for d, _, p in parsed.predicates()]
    change = [p for d, p in preds if d == 'must_change']
    preserve = [p for d, p in preds if d == 'must_preserve']
    types = {p.type for p in change}
    keep_rule = [p for p in change if p.type == 'duplicate_count' and p.keep]
    removes_nulls = any(p.type == 'rows_removed_where' and 'is_null' in str(p.condition) for p in change)
    fills = {p.target for p in change if p.type in ('column_value_relation', 'null_count')}
    removed = {c for p in change if p.type == 'columns_removed' for c in p.target}
    added = {c for p in change if p.type == 'columns_added' for c in p.target}
    changed_cols = {p.target for p in change if isinstance(getattr(p, 'target', None), str)} | removed | added
    protected = set()
    for p in preserve:
        if p.type in ('columns_unchanged', 'column_values_unchanged'):
            protected |= set(expand_columns(p.target, p.except_, columns))
    protected -= changed_cols
    calls = _calls(tree)
    names = [c[1] for c in calls]
    found = []

    def add(kind, line, message):
        if not any(h['code'] == kind for h in found):
            found.append(dict(code=kind, line=line, message=message))

    if keep_rule:
        dedup_lines = [l for l, n, _ in calls if n == 'drop_duplicates']
        sort_lines = [l for l, n, _ in calls if n in ('sort_values', 'sort_index')]
        if dedup_lines and any(l <= max(dedup_lines) for l in sort_lines):
            add('sort_before_dedup', min(sort_lines),
                f"keep='{keep_rule[0].keep}' means the {keep_rule[0].keep} row in ORIGINAL input order. Sorting "
                'before drop_duplicates changes which row survives; remove the sort.')
    if 'dropna' in names and not removes_nulls:
        add('unrequested_dropna', next(l for l, n, _ in calls if n == 'dropna'),
            'dropna removes rows the contract does not ask to remove; rows with missing values must stay.')
    if any(n == 'notna' for n in names) and 'rows_removed_where' in types and not removes_nulls:
        add('null_filter', next(l for l, n, _ in calls if n == 'notna'),
            'The removal condition never matches null cells, so rows with missing values must be kept; '
            'do not add notna() to the filter.')
    # Discarded results: a bare expression statement calling a method that returns a new object.
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            meth = _name(n.value.func)
            inplace = any(k.arg == 'inplace' for k in n.value.keywords)
            if meth in RESULT_METHODS and not inplace:
                add('discarded_result', n.lineno,
                    f'.{meth}(...) returns a new object and its result is discarded on line {n.lineno}; assign it '
                    "back, e.g. df['c'] = pd.Categorical(df['c'], categories=[...], ordered=True).")
    for line, meth, call in calls:
        if meth in CATEGORY_KWARGS:
            bad = [k.arg for k in call.keywords if k.arg and k.arg not in CATEGORY_KWARGS[meth] and k.arg != 'inplace']
            if bad:
                add('bad_category_kwargs', line,
                    f'.cat.{meth}() does not accept {bad}. Use pd.Categorical(values, categories=[...], ordered=...).')
    if any(p.type == 'column_value_relation' and p.fn == 'round' for p in change):
        uses_builtin = any(isinstance(c.func, ast.Name) and c.func.id == 'round' for _, _, c in calls)
        uses_np = any(isinstance(c.func, ast.Attribute) and c.func.attr == 'round' and _name(c.func.value) in ('np', 'numpy')
                      for _, _, c in calls)
        uses_apply = any(n == 'apply' and any(_name(a) == 'round' for a in c.args) for _, n, c in calls)
        no_digits = any(n == 'round' and not c.args and not c.keywords for _, n, c in calls)
        if uses_builtin or uses_np or uses_apply or no_digits:
            digits = next(int(p.args[1].lit) for p in change if p.type == 'column_value_relation' and p.fn == 'round')
            add('round_style', next(l for l, n, _ in calls if n in ('round', 'apply')),
                f'Use Series.round({digits}) on the column (NumPy rounding with {digits} decimals), not round()/np.round.')
    for colname, value, line in _assigned_columns(tree):
        if colname in protected and _contains_call(value, {'to_datetime', 'astype', 'to_numeric'}):
            add('overwrites_preserved_column', line,
                f"'{colname}' must keep its values and dtype. Convert into a temporary Series for comparisons "
                f"instead of assigning back to '{colname}'.")
    recreated = set()  # columns restored by rename (drop + rename back is legitimate)
    for _, meth, call in calls:
        if meth == 'rename':
            for k in call.keywords:
                if k.arg == 'columns' and isinstance(k.value, ast.Dict):
                    recreated |= {v.value for v in k.value.values if isinstance(v, ast.Constant)}
    if keep_rule:
        for line, meth, call in calls:
            if meth in ('first', 'last', 'agg', 'aggregate') and isinstance(call.func, ast.Attribute) \
                    and _contains_call(call.func.value, {'groupby'}):
                add('groupby_for_dedup', line,
                    'groupby(...).first()/last()/agg take the first non-null value per column, so with missing '
                    'values the result mixes rows; to keep whole original rows use '
                    "df[~df.duplicated(subset=[...], keep=...)].")
    for line, meth, call in calls:
        if meth == 'drop':
            dropped = [c for k in call.keywords if k.arg == 'columns' for c in _str_items(k.value)]
            dropped += [c for a in call.args for c in _str_items(a)] if any(k.arg == 'axis' for k in call.keywords) else []
            keep = [c for c in dropped if c in columns and c not in removed and c not in recreated]
            if keep:
                add('drops_required_column', line,
                    f'{keep} must remain in the output (only {sorted(removed) or "no columns"} may be removed).')
        if meth in ('merge', 'join') and fills:
            add('merge_for_fill', line,
                'Avoid merge/join to fill values: it can duplicate rows and rename columns (_x/_y). Compute the '
                "statistic with groupby(...).transform(...) and use fillna on the column.")
    return found


def render(found):
    return '\n'.join(f'- line {h["line"]}: {h["message"]}' for h in found)
