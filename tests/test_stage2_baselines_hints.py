"""Baselines 2/3 and no_contract_to_generator on the current 3B stack: system semantics, isolation, sealed pack."""
import json
from pathlib import Path

import pytest

from igt.agents.intent_parser import contract_to_text
from tests.test_stage2_combined_3b import factory
from eval.stage2_baselines_hints.config import RUNS, CONVENTIONAL_THRESHOLDS, REPAIR_VERSIONS
from eval.stage2_baselines_hints import nocontract
from eval.stage2_repair_hints.agent import HEADER

ROOT = Path(__file__).resolve().parents[1]
PARSER_PROMPT_START = 'You turn a data-transformation request into an intent contract'


def scripted(name):
    method = RUNS[name]['method']
    Base, live, calls, lifecycle = factory(method)

    class Backend(Base):
        def complete(self, messages, *args, **kwargs):
            if self.role == 'agent' and messages[0]['content'].startswith(PARSER_PROMPT_START):
                # baseline 3: the base 3B model acts as the prompted parser
                calls.append(('agent_as_parser', messages))
                tasks = {t['id']: t for _, t in __import__('eval.stage2_selective.runner', fromlist=['x']).load_tasks()}
                t = next(t for t in tasks.values() if ' '.join(t['request'].split()) in ' '.join(messages[1]['content'].split()))
                return contract_to_text(t['gold_contract']), 100, 50
            if self.role == 'generator' and name == 'baseline2':
                calls.append(('generator', messages))
                return '```python\ndef transform(df):\n    return df.drop(columns=list(df.columns[:2]))\n```', 100, 50
            return super().complete(messages, *args, **kwargs)
    return Backend, live, calls, lifecycle


def test_conventional_thresholds_match_the_reported_six_system_baseline2():
    saved = json.loads((ROOT / 'results/stage1/baseline2/thresholds.json').read_text())['tuned']['thresholds']
    assert saved == CONVENTIONAL_THRESHOLDS


@pytest.mark.parametrize('name', list(RUNS))
def test_scripted_smoke_follows_the_six_system_definitions(tmp_path, name):
    from eval.stage2_baselines_hints.runner import run
    from eval.stage2_contract_check import runner as check_runner
    saved = (check_runner.execute, check_runner.EXPERIMENT, check_runner.SYSTEMS, check_runner.THRESHOLDS)
    backend, live, calls, lifecycle = scripted(name)
    out = run(name, tmp_path / name, smoke=True, backend_factory=backend)
    assert (check_runner.execute, check_runner.EXPERIMENT, check_runner.SYSTEMS, check_runner.THRESHOLDS) == saved
    assert not live
    spec = RUNS[name]
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['experiment'] == 'stage2-baselines-hints-3b-fp16-v1' and plan['preset'] == spec['preset']
    assert plan['common']['baselines_run'] == name and plan['repair_prompt'] == REPAIR_VERSIONS[spec['preset']]
    assert plan['generator_prompt'] == plan['generator_prompt'] and 'b1' in json.dumps(plan['generator_prompt'])
    roles = [r for op, r in lifecycle if op == 'load']
    if name == 'baseline2':
        assert roles == ['generator', 'agent'] and plan['thresholds'] == CONVENTIONAL_THRESHOLDS
    elif name == 'baseline3':
        assert roles == ['agent', 'generator', 'agent'] and any(r == 'agent_as_parser' for r, _ in calls)
    else:
        assert roles == ['parser', 'generator', 'agent']
    for role, messages in calls:
        text = json.dumps(messages)
        assert 'reference_code' not in text
        if role == 'generator':  # request-only generation for all three systems
            assert '"must_change"' not in text and 'must_change' not in text
        if role == 'agent':
            if name == 'baseline2':
                assert 'intent contract)' not in text and HEADER not in text and 'must_change' not in text
                assert nocontract.REPLACEMENT.strip() in messages[0]['content']
            else:
                assert 'must_change' in text  # contract is used for checking/repair
    rows = [json.loads(p.read_text())['record'] for p in (out / 'tasks').glob('*.json')]
    assert len(rows) == 2
    for r in rows:
        assert r['system'] == spec['preset'] and r['generator_contract'] is False
        assert r['checker'] == ('conventional' if name == 'baseline2' else 'intent')
        assert r['attempts'] >= 1  # scripted generators always fail the first check
        if name != 'baseline2':
            assert r['final_correct'] and r['check_decision']
    if name != 'baseline2':
        events = [e['event'] for p in (out / 'tasks').glob('*.json') for e in json.loads(p.read_text())['trace']['events']]
        assert 'repair_hints' in events
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['summary']['tasks'] == 2 and not summary['model_quality_result']


def test_adapter_rules():
    from eval.stage2_baselines_hints.runner import run
    with pytest.raises(ValueError, match='does not use'):
        run('baseline3', '/nonexistent', adapter=Path('x'))
    with pytest.raises(ValueError, match='Unknown run'):
        run('proposed', '/nonexistent')


@pytest.mark.parametrize('name', list(RUNS))
def test_notebook_loader_unpacks_sealed_pack_and_runs_from_it(tmp_path, monkeypatch, name):
    import shutil
    import subprocess
    import sys
    import nbformat
    here = ROOT / 'notebooks/kaggle/stage2_baselines_hints'
    inputs = tmp_path / 'input'
    inputs.mkdir()
    shutil.copy(here / 'stage2_baselines_hints_source.zip', inputs)
    book = nbformat.read(here / f'stage2_baselines_hints_{name}_kaggle.ipynb', as_version=4)
    assert f"'--run',RUN" in book.cells[4].source and 'eval.stage2_baselines_hints.runner' in book.cells[4].source
    assert ("'--adapter'" in book.cells[4].source) == RUNS[name]['adapter']
    assert ('adapters' in book.cells[3].source) == RUNS[name]['adapter']
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: None)
    ns = {}
    for cell in book.cells[1:3]:
        exec(cell.source.replace("INPUT=Path('/kaggle/input')", f'INPUT=Path({str(inputs)!r})')
             .replace("WORK=Path('/kaggle/working')", f'WORK=Path({str(tmp_path / "work")!r})'), ns)
    root = ns['ROOT']
    assert ns['RUN'] == name and (root / '_stage2_baselines_hints_manifest.json').exists()
    code = ('import sys;sys.path[:0]=[".","src"];'
            'from eval.stage2_baselines_hints.artifacts import verify_pack;assert verify_pack();'
            'import eval.stage2_repair_hints.artifacts as h;assert h.verify_pack();'
            'import eval.stage2_contract_check.artifacts as c;assert c.verify_pack();'
            'import eval.stage2_combined_3b.artifacts as a;assert a.verify_pack();'
            'import eval.stage2_baselines_hints.runner,eval.stage2_combined_3b.preflight;print("ok")')
    result = real_run([sys.executable, '-I', '-c', code], cwd=root, capture_output=True, text=True)
    assert result.stdout.strip() == 'ok', result.stderr[-2000:]
