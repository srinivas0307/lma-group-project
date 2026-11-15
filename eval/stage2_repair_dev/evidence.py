"""Bounded diagnostics from input, candidate output and the predicted contract only."""
import json
import pandas as pd
from igt.contracts.schema import load_contract, expand_columns
from igt.verifier.alignment import align
from igt.verifier.intent_verifier import _key_columns
from igt.verifier.functions import expected_values
from igt.verifier.values import row_keys, value_equal


SEMANTICS = ('Contract operator conventions: round uses NumPy rounding (pandas Series.round), '
             'including its floating-point tie behavior. Deduplication keeps an entire original '
             'record; first/last refers to input order. Category order is metadata, not a remapping of values.')


def scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    if hasattr(value, 'item'):
        value = value.item()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def row(frame, position, columns):
    return {str(c): scalar(frame.iloc[position][c]) for c in columns if c in frame.columns}


def diagnose(contract, before, after, report, max_examples=3):
    result = dict(source='input + submitted candidate + contract; no reference answer',
                  output_available=after is not None,
                  failures=report['failures'], examples=[], notes=[])
    if after is None:
        result['notes'].append('This candidate produced no usable output. Inspect before, not last_output.')
        return result
    result['schema_delta'] = dict(missing=[c for c in before if c not in after],
                                  added=[c for c in after if c not in before],
                                  duplicate=list(after.columns[after.columns.duplicated()]))
    # Do not guess row identity or use .loc on duplicate indices.
    if not before.index.is_unique or not before.columns.is_unique or not after.columns.is_unique:
        result['notes'].append('Ambiguous indices/columns: paired cell examples omitted.')
        return result
    try:
        parsed = load_contract(contract)
        alignment = align(before, after, _key_columns(parsed, before.columns))
        result['alignment'] = dict(method=alignment.method, ambiguous=alignment.ambiguous)
        if not alignment.ok():
            result['notes'].append('Ambiguous row alignment: paired cell examples omitted.')
            return result
        before_pos = {bid: i for i, bid in enumerate(before.index)}
        output_pos = dict(zip(alignment.before_ids, alignment.after_pos))
        predicates = {pid: p for _, pid, p in parsed.predicates()}
        for failure in report['failures']:
            predicate = predicates.get(failure.get('predicate_id'))
            if predicate is None:
                continue
            if predicate.type == 'column_dtype' and predicate.target in after:
                series = after[predicate.target]
                result['examples'].append(dict(predicate=failure['predicate_id'], column=predicate.target,
                    observed_dtype=str(series.dtype), observed_ordered=bool(series.cat.ordered) if isinstance(series.dtype, pd.CategoricalDtype) else None,
                    observed_categories=list(map(scalar,series.cat.categories[:20])) if isinstance(series.dtype,pd.CategoricalDtype) else None,
                    required_ordered=predicate.ordered, required_categories=predicate.categories))
                continue
            expected = expected_values(predicate, before) if predicate.type == 'column_value_relation' else None
            if predicate.type == 'duplicate_count':
                keys = expand_columns(predicate.target, predicate.except_, before.columns)
                groups = {}
                for bid,key in zip(before.index,row_keys(before,keys)):
                    groups.setdefault(key,[]).append(bid)
                group_of = {bid: ids for ids in groups.values() for bid in ids}
            for bid in failure.get('evidence_rows', [])[:max_examples]:
                if len(result['examples']) >= max_examples:
                    break
                if bid not in before_pos:
                    continue
                pos = before_pos[bid]
                columns = list(before.columns)[:8]
                if isinstance(predicate.target,str) and predicate.target in before:
                    columns = [predicate.target]
                if predicate.type in ('columns_unchanged','column_values_unchanged'):
                    protected = expand_columns(predicate.target,predicate.except_,before.columns)
                    changed = []
                    if bid in output_pos:
                        outpos = output_pos[bid]
                        changed = [c for c in protected if c in after and not value_equal(before.iloc[pos][c],after.iloc[outpos][c])]
                    columns = (changed or protected)[:8]
                item = dict(predicate=failure['predicate_id'], input_position=pos,
                            input_row_id=scalar(bid), input=row(before,pos,columns),
                            retained=bid in output_pos)
                if bid in output_pos:
                    outpos = output_pos[bid]
                    item['output_position'] = outpos
                    item['output'] = row(after,outpos,columns)
                    if expected is not None:
                        item.update(column=predicate.target, expected_from_contract=scalar(expected[pos]),
                                    observed=scalar(after.iloc[outpos][predicate.target]))
                if predicate.type == 'duplicate_count':
                    ids = group_of[bid]
                    item.update(key_columns=keys, key_values=row(before,pos,keys), keep=predicate.keep,
                        group_input_positions=[before_pos[b] for b in ids[:5]],
                        retained_input_positions=[before_pos[b] for b in ids if b in output_pos][:5])
                result['examples'].append(item)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        result['notes'].append('Detailed examples unavailable: '+type(exc).__name__+': '+str(exc)[:200])
    result['examples'] = result['examples'][:max_examples]
    return result


def render(value, limit=6500):
    text = json.dumps(value,ensure_ascii=False,default=str)
    if len(text) <= limit:
        return text
    # Preserve valid JSON and label omitted detail instead of cutting a JSON value.
    return json.dumps(dict(output_available=value.get('output_available'),
                           failures=value.get('failures',[])[:2],
                           schema_delta=value.get('schema_delta'),
                           notes=['Long diagnostic: showing first two failures; use tools for more evidence.']),default=str)
