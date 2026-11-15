"""repair-hints-v1: recipe semantics, lints on synthetic code, prompt injection."""
import pandas as pd
import pytest

from igt.verifier import verify
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from eval.stage2_repair_dev.guard import guarded_verifier
from eval.stage2_repair_hints import lint
from eval.stage2_repair_hints.recipes import assemble, recipes, Unsupported
from eval.stage2_repair_hints.agent import hint_block
from eval.stage2_selective.runner import load_tasks


@pytest.fixture
def parcels():
    # Synthetic data with names unrelated to the benchmark datasets.
    return pd.DataFrame(dict(
        parcel=[10, 11, 12, 13, 14, 15],
        depot=['Ash', 'Bay', 'Ash', 'Cove', 'Bay', None],
        weight=[1.25, 2.35, None, 0.0, 7.55, 3.45],
        route=[3, 1, 3, 2, 1, 2],
        flag=['Y', 'N', 'N', 'Y', 'N', 'Y'],
    ))


KEEP = {"type": "columns_unchanged", "target": "*", "except": []}


def passes(contract, frame):
    code = assemble(contract, list(frame.columns))
    run = safe_run(code, frame, **Limits().sandbox())
    assert run.ran_ok, run.error
    verdict = guarded_verifier(lambda a: verify(contract, frame, a)[0], contract, frame)(run.df_out)
    return verdict.status == 'PASS', run.df_out


def test_removal_keeps_null_rows_and_not_in_excludes_nulls(parcels):
    contract = {"must_change": [{"type": "rows_removed_where",
                                 "condition": {"column": "depot", "op": "not_in", "value": ["Ash"]}}],
                "must_preserve": [KEEP]}
    ok, out = passes(contract, parcels)
    assert ok and list(out['parcel']) == [10, 12, 15]  # the null depot row is not removed


def test_dedup_keeps_input_order(parcels):
    contract = {"must_change": [{"type": "duplicate_count", "target": ["route"], "op": "eq", "value": 0, "keep": "last"}],
                "must_preserve": [KEEP]}
    ok, out = passes(contract, parcels)
    assert ok and list(out['parcel']) == [12, 14, 15]


def test_relations_where_round_map_and_category(parcels):
    contract = {"must_change": [
        {"type": "column_value_relation", "target": "weight", "fn": "round", "args": [{"col": "weight"}, {"lit": 1}]},
        {"type": "column_value_relation", "target": "depot", "fn": "const", "args": [{"lit": "Unknown"}],
         "where": {"column": "depot", "op": "is_null"}},
        {"type": "column_dtype", "target": "flag", "dtype": "bool"},
        {"type": "columns_added", "target": ["heavy"]},
        {"type": "column_value_relation", "target": "heavy", "fn": "mul", "args": [{"col": "weight"}, {"lit": 2}]}],
        "must_preserve": [{"type": "column_value_relation", "target": "flag", "fn": "map",
                           "args": [{"col": "flag"}, {"lit": {"Y": True, "N": False}}]},
                          {"type": "columns_unchanged", "target": "*", "except": ["weight", "depot", "flag"]}]}
    ok, out = passes(contract, parcels)
    assert ok and out['weight'].tolist()[:2] == [1.2, 2.4] and out['depot'].iloc[-1] == 'Unknown'


def test_constraint_only_contract_is_not_compiled(parcels):
    contract = {"must_change": [{"type": "null_count", "target": "weight", "op": "eq", "value": 0}],
                "must_preserve": [KEEP]}
    with pytest.raises(Unsupported):
        assemble(contract, list(parcels.columns))
    assert 'fill using the rule in the request' in recipes(contract, parcels.columns)['c1']['note']


def test_every_dev_test_gold_contract_compiles_to_a_verified_reference_output():
    from eval.equivalence import outputs_equal
    for split in ('dev', 'test'):
        for _, task in load_tasks(split):
            before = pd.read_csv(task['data_file'])
            ok, out = passes(task['gold_contract'], before)
            ref = safe_run(task['reference_code'], before, **Limits().sandbox()).df_out
            assert ok and outputs_equal(ref, out)[0], task['id']


DEDUP = {"must_change": [{"type": "duplicate_count", "target": ["route"], "op": "eq", "value": 0, "keep": "first"}],
         "must_preserve": [KEEP]}
COLS = ['parcel', 'depot', 'weight', 'route', 'flag']


def kinds(code, contract=DEDUP):
    return {h['code'] for h in lint.hints(code, contract, COLS)}


def test_lints_fire_on_pitfalls():
    assert 'sort_before_dedup' in kinds("def transform(df):\n    df = df.sort_values('route')\n"
                                        "    return df.drop_duplicates(subset=['route'], keep='first')\n")
    assert 'unrequested_dropna' in kinds("def transform(df):\n    df = df.dropna()\n"
                                         "    return df.drop_duplicates(subset=['route'])\n")
    cat = {"must_change": [{"type": "column_dtype", "target": "depot", "dtype": "category", "ordered": True,
                            "categories": ["Ash", "Bay", "Cove"]}], "must_preserve": [KEEP]}
    found = kinds("def transform(df):\n    df['depot'] = df['depot'].astype('category')\n"
                  "    df['depot'].cat.add_categories(['Z'], ordered=True)\n    return df\n", cat)
    assert {'discarded_result', 'bad_category_kwargs'} <= found
    rnd = {"must_change": [{"type": "column_value_relation", "target": "weight", "fn": "round",
                            "args": [{"col": "weight"}, {"lit": 1}]}], "must_preserve": [KEEP]}
    assert 'round_style' in kinds("def transform(df):\n    df['weight'] = df['weight'].apply(round)\n    return df\n", rnd)
    filt = {"must_change": [{"type": "rows_removed_where", "condition": {"column": "route", "op": "==", "value": 1}}],
            "must_preserve": [{"type": "columns_unchanged", "target": "*", "except": []}]}
    assert 'drops_required_column' in kinds("def transform(df):\n    return df[df['route'] != 1].drop(columns=['flag'])\n", filt)
    assert 'syntax_error' in kinds("def transform(df):\n    return df[df[(df['a'] == 1)]\n")


def test_lints_silent_on_correct_code():
    assert not kinds("def transform(df):\n    return df.drop_duplicates(subset=['route'], keep='first')\n")
    rnd = {"must_change": [{"type": "column_value_relation", "target": "weight", "fn": "round",
                            "args": [{"col": "weight"}, {"lit": 1}]}], "must_preserve": [KEEP]}
    assert not kinds("def transform(df):\n    df['r2'] = df['route'].map({1: 'a'})\n"
                     "    return df.drop('route', axis=1).rename(columns={'r2': 'route'})\n", rnd)
    assert 'groupby_for_dedup' in kinds("def transform(df):\n    return df.groupby(['route'], as_index=False).first()\n")
    assert not kinds("def transform(df):\n    out = df.copy()\n    out['weight'] = out['weight'].round(1)\n    return out\n", rnd)


def test_hint_block_targets_failed_predicates():
    contract = {"must_change": [{"id": "c1", "type": "duplicate_count", "target": ["route"], "op": "eq", "value": 0,
                                 "keep": "first"}, {"id": "c2", "type": "row_count", "op": "decreased"}],
                "must_preserve": [{"id": "p1", **KEEP}]}
    code = "def transform(df):\n    df = df.sort_values('route')\n    return df.drop_duplicates(subset=['route'])\n"
    block = hint_block(contract, COLS, code, ['c1'])
    assert "df.duplicated(subset=['route'], keep='first')" in block and 'sort' in block and 'p1' not in block
    crashed = hint_block(contract, COLS, "def transform(df):\n    return df['nope']\n", ['execution'])
    assert 'c1' in crashed and 'c2' in crashed


@pytest.mark.parametrize('pipeline', ['gold', 'selective'])
def test_pipeline_smoke_uses_hints_only_in_repair(tmp_path, pipeline):
    import json
    from tests.test_stage2_combined_3b import factory
    from eval.stage2_repair_hints.runner import run
    from eval.stage2_repair_hints.agent import HEADER
    from eval.stage2_combined_3b import execute as proposed_execute
    from eval.stage2_contract_check import runner as check_runner
    original = proposed_execute.run_repair, check_runner.EXPERIMENT
    backend, live, calls, lifecycle = factory('selective')
    out = run(pipeline, tmp_path / pipeline, smoke=True, backend_factory=backend)
    assert (proposed_execute.run_repair, check_runner.EXPERIMENT) == original  # frozen modules restored
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['experiment'] == 'stage2-repair-hints-3b-fp16-v1' and plan['common']['repair_hints'] == 'repair-hints-v1'
    assert plan['repair_prompt'].endswith('+repair-hints-v1')
    agent_msgs = [m for role, m in calls if role == 'agent']
    other_msgs = [m for role, m in calls if role != 'agent']
    assert agent_msgs and any(HEADER in x['content'] for m in agent_msgs for x in m)
    assert not any(HEADER in json.dumps(m) for m in other_msgs)
    assert all('reference_code' not in json.dumps(m) for _, m in calls)
    for p in (out / 'tasks').glob('*.json'):
        task = json.loads(p.read_text())
        assert task['record']['repair_version'].endswith('+repair-hints-v1')
        if task['record']['attempts']:
            assert any(e['event'] == 'repair_hints' for e in task['trace']['events'])


@pytest.mark.parametrize('pipeline', ['gold', 'selective', 'tools'])
def test_notebook_loader_unpacks_sealed_pack_and_runs_from_it(tmp_path, monkeypatch, pipeline):
    import shutil
    import subprocess
    import sys
    import nbformat
    from eval.stage2_combined_3b.runner import ROOT
    here = ROOT / 'notebooks/kaggle/stage2_repair_hints'
    inputs = tmp_path / 'input'
    inputs.mkdir()
    shutil.copy(here / 'stage2_repair_hints_source.zip', inputs)
    book = nbformat.read(here / f'stage2_repair_hints_{pipeline}_kaggle.ipynb', as_version=4)
    assert 'eval.stage2_repair_hints.runner' in book.cells[4].source
    assert ("'--adapter'" in book.cells[4].source) == (pipeline != 'gold')
    assert ('adapters' in book.cells[3].source) == (pipeline != 'gold')
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: None)
    ns = {}
    for cell in book.cells[1:3]:
        exec(cell.source.replace("INPUT=Path('/kaggle/input')", f'INPUT=Path({str(inputs)!r})')
             .replace("WORK=Path('/kaggle/working')", f'WORK=Path({str(tmp_path / "work")!r})'), ns)
    root = ns['ROOT']
    assert ns['PIPELINE'] == pipeline and (root / '_stage2_repair_hints_manifest.json').exists()
    # From the unpacked pack alone: every frozen pack still verifies, and a scripted gold smoke runs end to end.
    code = ('import sys;sys.path[:0]=[".","src"];'
            'from eval.stage2_repair_hints.artifacts import verify_pack;assert verify_pack();'
            'import eval.stage2_contract_check.artifacts as c;assert c.verify_pack();'
            'import eval.stage2_combined_gold.artifacts as g;assert g.verify_pack();'
            'import eval.stage2_combined_3b.artifacts as a;assert a.verify_pack();'
            'import eval.stage2_repair_hints.runner,eval.stage2_combined_3b.preflight;print("ok")')
    result = real_run([sys.executable, '-I', '-c', code], cwd=root, capture_output=True, text=True)
    assert result.stdout.strip() == 'ok', result.stderr[-2000:]
