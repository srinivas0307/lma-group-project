import csv
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from igt.contracts import load_contract, expand_columns
from igt.sandbox import safe_run
from igt.verifier import verify

rows = list(csv.DictReader(open('finetune/data/v2/review_changed.csv')))
examples = {r['id']: r for r in map(json.loads, open('finetune/data/v2/checked.jsonl'))}
frames = {}
results = []
for row in rows:
    ex = examples[row['id']]
    assert row['request'] == ex['request']
    assert json.loads(row['contract']) == ex['contract']
    assert row['reference_code'] == ex['reference_code']
    if ex['data_file'] not in frames:
        frames[ex['data_file']] = pd.read_csv(ROOT / ex['data_file'])
    before = frames[ex['data_file']]
    contract = load_contract(ex['contract'])
    run = safe_run(row['reference_code'], before, timeout_s=30, cpu_s=15, mem_gb=2)
    assert run.ran_ok, (row['id'], run.status, run.error)
    out = run.df_out
    def status(after):
        return verify(contract, before, after)[0].status
    result = dict(id=row['id'], before=len(before), after=len(out), reference=status(out), noop=status(before.copy()))
    assert result['reference'] == 'PASS', result
    assert result['noop'] == 'VIOLATION', result
    if row['type'] == 'deduplication':
        pred = contract.must_change[0]
        key = expand_columns(pred.target, pred.except_, before.columns)
        assert pred.keep in ('first', 'last')
        result.update(empty=status(before.iloc[:0].copy()), one_row=status(out.iloc[:1].copy()),
                      missing_group=status(out.iloc[1:].copy()),
                      reversed_survivors=status(before.drop_duplicates(subset=key, keep='last' if pred.keep == 'first' else 'first').copy()))
        for name in ('empty', 'one_row', 'missing_group', 'reversed_survivors'):
            assert result[name] == 'VIOLATION', result
    else:
        col = ex['contract']['must_change'][0]['condition']['column']
        result['nulls_before'] = int(before[col].isna().sum())
        result['nulls_after'] = int(out[col].isna().sum())
        result['drop_nulls'] = status(out.loc[out[col].notna()].copy())
        assert result['nulls_before'] == result['nulls_after'] > 0, result
        assert result['drop_nulls'] == 'VIOLATION', result
    results.append(result)
    print(row['id'], 'PASS; negative controls rejected', flush=True)
Path('tmp/review/v2_checks.json').write_text(json.dumps(results, indent=2))
print('Verified', len(results), 'examples:', dict(Counter(row['type'] for row in rows)))
