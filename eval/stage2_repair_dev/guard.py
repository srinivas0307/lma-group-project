"""Closed output-column set derived only from explicit contract declarations."""
from igt.contracts.schema import load_contract, ContractError
from igt.types import Verdict, check_failure, uncertain_failure


def schema_failures(contract, before, after):
    try:
        parsed = load_contract(contract)
    except ContractError as exc:
        return [uncertain_failure('invalid_contract', str(exc))]
    if not before.columns.is_unique:
        return [uncertain_failure('predicate_not_evaluable', 'Schema guard requires unique input columns')]
    added, removed = set(), set()
    for direction, _, predicate in parsed.predicates():
        if direction != 'must_change':
            continue
        if predicate.type == 'columns_added':
            added.update(predicate.target)
        elif predicate.type == 'columns_removed':
            removed.update(predicate.target)
    expected = (set(before.columns) - removed) | added
    actual = set(after.columns)
    missing = sorted(expected - actual, key=str)
    extra = sorted(actual - expected, key=str)
    duplicate = list(after.columns[after.columns.duplicated()])
    if missing or extra or duplicate:
        return [check_failure('declared_output_schema',
                              dict(missing=missing, unexpected=extra, duplicate=duplicate),
                              dict(columns=sorted(expected, key=str)))]
    return []


def guarded_verifier(verify_fn, contract, before):
    def check(after):
        verdict = verify_fn(after)
        return Verdict.combine(verdict.failures + schema_failures(contract, before, after))
    return check
