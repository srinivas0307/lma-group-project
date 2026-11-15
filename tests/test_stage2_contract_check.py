"""contract-check-v1: diagnostics on synthetic data, gold-contract safety, correction policy and pipeline smoke."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from igt.agents.intent_parser import contract_to_text
from eval.stage2_contract_check.diagnostics import diagnose
from eval.stage2_contract_check.correction import check_and_correct
from eval.stage2_selective.runner import load_tasks

def ok(contract):
    return dict(status='ok', contract=contract)


def codes(issues, kind=None):
    return {i['code'] for i in issues if kind is None or i['kind'] == kind}


@pytest.fixture
def shops():
    # Synthetic data with names unrelated to the benchmark datasets.
    return pd.DataFrame(dict(
        store_id=[1, 2, 3, 4, 5, 6],
        region=['North', 'South', 'North', 'East', 'South', 'West'],
        district=['Elm', 'Oak', 'Pine', 'Elm', 'Birch', 'Oak'],
        grade=['A', 'B', 'C', 'B', 'A', 'C'],
        staff=[3, 0, 5, 0, 2, 4],
        rating=[4.5, None, 3.0, None, 2.5, 4.0],
        opened=['2024-01-05 10:00:00', '2024-02-11 09:30:00', '2024-03-01 08:00:00',
                '2024-03-15 12:00:00', '2024-04-02 07:45:00', '2024-05-20 11:15:00'],
        owner=['Kim', 'Lee', 'Kim', 'Ng', 'Ng', 'Ali'],
    ))


KEEP_ALL = [{"type": "columns_unchanged", "target": "*", "except": []}]


def filt(column, op, value, decrease=True):
    change = [{"type": "rows_removed_where", "condition": {"column": column, "op": op, "value": value}}]
    if decrease:
        change.append({"type": "row_count", "op": "decreased"})
    return {"must_change": change, "must_preserve": KEEP_ALL}


def test_value_in_wrong_column_is_flagged_with_the_right_column(shops):
    issues = diagnose('Keep only stores in the North and South regions.',
                      ok(filt('district', 'not_in', ['North', 'South'])), shops)
    hit = [i for i in issues if i['code'] == 'literal_absent']
    assert hit and "'region'" in hit[0]['detail'] and hit[0]['kind'] == 'evidence'
    assert not diagnose('Keep only stores in the North and South regions.',
                        ok(filt('region', 'not_in', ['North', 'South'])), shops)


def test_empty_removal_is_proof_only_when_rows_must_decrease(shops):
    issues = diagnose('Remove grade Z stores.', ok(filt('grade', '==', 'Z')), shops)
    assert 'removal_matches_nothing' in codes(issues, 'proof')
    issues = diagnose('Remove grade Z stores if any.', ok(filt('grade', '==', 'Z', decrease=False)), shops)
    assert 'removal_matches_nothing' in codes(issues, 'evidence') and not codes(issues, 'proof')


def test_ordering_on_text_is_proof_but_threshold_literal_need_not_exist(shops):
    issues = diagnose('Remove low grades.', ok(filt('grade', '<', 2)), shops)
    assert 'condition_not_evaluable' in codes(issues, 'proof')
    # A date threshold that does not occur as a cell value is legitimate.
    assert not diagnose('Remove stores opened before March.', ok(filt('opened', '<', '2024-03-01')), shops)


def test_dedup_key_without_duplicates(shops):
    bad = {"must_change": [{"type": "duplicate_count", "target": "*", "op": "eq", "value": 0, "keep": "first"},
                           {"type": "row_count", "op": "decreased"}], "must_preserve": KEEP_ALL}
    issues = diagnose('Keep one store per owner, the first one.', ok(bad), shops)
    hit = [i for i in issues if i['code'] == 'dedup_key_has_no_duplicates']
    assert hit and hit[0]['kind'] == 'proof' and "'owner'" in hit[0]['detail']
    good = copy.deepcopy(bad)
    good['must_change'][0]['target'] = ['owner']
    assert not diagnose('Keep one store per owner, the first one.', ok(good), shops)


def test_map_default_branch_flagged_but_complete_map_with_extra_key_is_fine(shops):
    def mapping(lit):
        return {"must_change": [{"type": "columns_added", "target": ["tier"]},
                                {"type": "column_value_relation", "target": "tier", "fn": "map",
                                 "args": [{"col": "grade"}, {"lit": lit}]}], "must_preserve": KEEP_ALL}
    issues = diagnose('tier is Top for A and Other for the rest', ok(mapping({"A": "Top", "Other": "Rest"})), shops)
    assert 'map_key_absent' in codes(issues)
    assert not diagnose('map grades', ok(mapping({"A": "Top", "B": "Mid", "C": "Low", "D": "Bad"})), shops)


def fill(null_rule):
    return {"must_change": [{"type": "column_value_relation", "target": "rating", "where":
                             {"column": "rating", "op": "is_null"}, "fn": "const", "args": [{"lit": 0.0}]}]
            + ([null_rule] if null_rule else []),
            "must_preserve": [{"type": "columns_unchanged", "target": "*", "except": ["rating"]},
                              {"type": "row_count", "op": "unchanged"}]}


def test_null_count_must_agree_with_fill(shops):
    issues = diagnose('Fill missing ratings with 0.', ok(fill({"type": "null_count", "target": "rating",
                                                                "op": "unchanged"})), shops)
    assert 'null_count_contradiction' in codes(issues)
    assert not diagnose('Fill missing ratings with 0.', ok(fill({"type": "null_count", "target": "rating",
                                                                 "op": "eq", "value": 0})), shops)


def test_replacement_cannot_increase_nulls(shops):
    c = {"must_change": [{"type": "column_value_relation", "target": "staff", "where":
                          {"column": "staff", "op": "==", "value": 0}, "fn": "const", "args": [{"lit": 1}]},
                         {"type": "null_count", "target": "staff", "op": "increased"}],
         "must_preserve": [{"type": "column_values_unchanged", "target": "staff",
                            "where": {"column": "staff", "op": "!=", "value": 0}},
                           {"type": "columns_unchanged", "target": "*", "except": ["staff"]},
                           {"type": "row_count", "op": "unchanged"}]}
    assert 'null_count_contradiction' in codes(diagnose('Staff 0 means unknown; set it to 1.', ok(c), shops), 'proof')


def test_type_conversion_vs_value_preservation(shops):
    bad = {"must_change": [{"type": "column_dtype", "target": "opened", "dtype": "datetime"}],
           "must_preserve": [{"type": "column_values_unchanged", "target": ["opened"]},
                             {"type": "columns_unchanged", "target": "*", "except": ["opened"]}]}
    assert 'dtype_vs_value_preservation' in codes(diagnose('Make opened a datetime.', ok(bad), shops), 'proof')
    good = {"must_change": bad['must_change'], "must_preserve": bad['must_preserve'][1:]}
    assert not diagnose('Make opened a datetime.', ok(good), shops)


def test_noop_change_is_evidence(shops):
    c = {"must_change": [{"type": "column_value_relation", "target": "staff", "fn": "mul",
                          "args": [{"col": "staff"}, {"lit": 1}]}],
         "must_preserve": [{"type": "columns_unchanged", "target": "*", "except": ["staff"]}]}
    issues = diagnose('Double staff.', ok(c), shops)
    assert codes(issues) == {'change_is_noop'} and not codes(issues, 'proof')


@pytest.mark.parametrize('split', ['dev', 'test'])
def test_no_gold_contract_has_a_proof_issue(split):
    # The checks must never prove a reviewed gold contract inconsistent.
    frames = {}
    for _, task in load_tasks(split):
        if task['data_file'] not in frames:
            from eval.stage2_combined_3b.runner import ROOT
            frames[task['data_file']] = pd.read_csv(ROOT / task['data_file'])
        issues = diagnose(task['request'], ok(task['gold_contract']), frames[task['data_file']])
        assert not codes(issues, 'proof'), (task['id'], issues)


class Scripted:
    model = 'scripted-parser'

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def prompt_tokens(self, messages):
        return 10

    def complete(self, messages, *args, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return self.replies.pop(0), 100, 40


def first_parse(contract):
    text = contract_to_text(contract)
    return dict(task_id='t', messages=[dict(role='system', content='s'), dict(role='user', content='u')],
                raw_output=text, extracted=ok(contract), tokens=150, llm_calls=1, seconds=1.0)


TASK = dict(id='t', request='Keep one store per owner, the first one.', data_file='x.csv')
BAD = {"must_change": [{"type": "duplicate_count", "target": "*", "op": "eq", "value": 0, "keep": "first"},
                       {"type": "row_count", "op": "decreased"}], "must_preserve": KEEP_ALL}
GOOD = copy.deepcopy(BAD)
GOOD['must_change'][0]['target'] = ['owner']


def test_no_issue_means_no_extra_call(shops):
    backend = Scripted([])
    record, trace = check_and_correct(TASK, shops, first_parse(GOOD), backend, 8192)
    assert trace['decision'] == 'no_issue' and not backend.calls and record['extracted']['contract'] == GOOD


def test_correction_used_when_it_resolves_the_issue(shops):
    backend = Scripted([contract_to_text(GOOD)])
    record, trace = check_and_correct(TASK, shops, first_parse(BAD), backend, 8192)
    assert trace['decision'] == 'use_corrected' and len(backend.calls) == 1
    assert record['extracted']['contract']['must_change'][0]['target'] == ['owner']
    assert record['tokens'] == 150 + 140 and record['llm_calls'] == 2
    assert 'dedup_key_has_no_duplicates' in backend.calls[0][-1]['content']
    assert backend.calls[0][-2]['role'] == 'assistant'


def test_unresolved_proof_is_withheld_after_exactly_one_correction(shops):
    backend = Scripted([contract_to_text(BAD), contract_to_text(GOOD)])
    record, trace = check_and_correct(TASK, shops, first_parse(BAD), backend, 8192)
    assert trace['decision'] == 'withhold_proven_inconsistency' and len(backend.calls) == 1
    assert record['extracted']['status'] == 'contract_check_failed' and record['extracted']['contract'] is None


def test_evidence_only_keeps_first_when_correction_is_worse(shops):
    evidence = filt('grade', '==', 'Z', decrease=False)
    backend = Scripted(['not a contract'])
    record, trace = check_and_correct(dict(TASK, request='Remove grade Z stores.'), shops, first_parse(evidence),
                                      backend, 8192)
    assert trace['decision'] == 'keep_first_evidence_only' and record['extracted']['contract'] == evidence


def test_context_limit_skips_the_call(shops):
    backend = Scripted([])
    record, trace = check_and_correct(TASK, shops, first_parse(BAD), backend, 100)
    assert trace.get('context_limit') and not backend.calls
    assert trace['decision'] == 'withhold_proven_inconsistency'


@pytest.mark.parametrize('arm', ['control', 'checked'])
def test_pipeline_smoke_with_one_bad_first_parse(tmp_path, arm):
    from tests.test_stage2_combined_3b import factory
    from eval.stage2_contract_check.runner import run
    Base, live, calls, lifecycle = factory('selective')
    request = next(t['request'] for _, t in load_tasks() if t['id'] == 'adult_col_001')
    seen = set()

    class Backend(Base):
        def complete(self, messages, *args, **kwargs):
            first_adult_parse = (self.role == 'parser' and len(messages) == 2 and 'adult' not in seen and
                                 ' '.join(request.split()) in ' '.join(messages[1]['content'].split()))
            if first_adult_parse:
                seen.add('adult')
                calls.append((self.role, copy.deepcopy(messages)))
                broken = {"must_change": [{"type": "rows_removed_where", "condition":
                                           {"column": "workclass", "op": "==", "value": "Nobody"}},
                                          {"type": "row_count", "op": "decreased"}], "must_preserve": KEEP_ALL}
                return contract_to_text(broken), 100, 50
            return super().complete(messages, *args, **kwargs)

    out = run('selective', tmp_path / arm, smoke=True, backend_factory=Backend, arm=arm)
    rows = {json.loads(p.read_text())['record']['task_id']: json.loads(p.read_text())['record']
            for p in (out / 'tasks').glob('*.json')}
    assert set(rows) == {'adult_col_001', 'titanic_miss_004'}
    adult = rows['adult_col_001']
    assert 'removal_matches_nothing' in adult['check_first_issue_codes']
    if arm == 'checked':
        assert adult['check_decision'] == 'use_corrected' and adult['final_correct']
        assert adult['parser_calls'] == 2 and adult['first_parser_reference_verdict']['status'] != 'PASS'
    else:
        assert adult['check_decision'] == 'control_not_applied' and not adult['final_correct']
    assert rows['titanic_miss_004']['check_decision'] in ('no_issue', 'control_not_applied')
    assert rows['titanic_miss_004']['final_correct']
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['arm'] == arm and plan['experiment'] == 'stage2-contract-check-3b-fp16-v1'
    assert all('reference_code' not in json.dumps(m) for _, m in calls)


@pytest.mark.parametrize('method', ['selective', 'tools'])
def test_notebook_loader_unpacks_sealed_pack_and_runner_imports_from_it(tmp_path, monkeypatch, method):
    import shutil
    import subprocess
    import sys
    import nbformat
    from eval.stage2_combined_3b.runner import ROOT
    here = ROOT / 'notebooks/kaggle/stage2_contract_check'
    inputs = tmp_path / 'input'
    inputs.mkdir()
    shutil.copy(here / 'stage2_contract_check_source.zip', inputs)
    book = nbformat.read(here / f'stage2_contract_check_{method}_kaggle.ipynb', as_version=4)
    assert "'--arm','checked'" in book.cells[4].source and 'eval.stage2_contract_check.runner' in book.cells[4].source
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: None)
    ns = {}
    for cell in book.cells[1:3]:
        exec(cell.source.replace("INPUT=Path('/kaggle/input')", f'INPUT=Path({str(inputs)!r})')
             .replace("WORK=Path('/kaggle/working')", f'WORK=Path({str(tmp_path / "work")!r})'), ns)
    root = ns['ROOT']
    assert (root / '_stage2_contract_check_manifest.json').exists()
    code = ('import sys;sys.path[:0]=[".","src"];'
            'from eval.stage2_contract_check.artifacts import verify_pack;assert verify_pack();'
            'import eval.stage2_combined_3b.artifacts as a;assert a.verify_pack();'
            'import eval.stage2_contract_check.runner,eval.stage2_combined_3b.preflight;print("ok")')
    result = real_run([sys.executable, '-I', '-c', code], cwd=root, capture_output=True, text=True)
    assert result.stdout.strip() == 'ok', result.stderr[-2000:]
