"""CPU replay of repair-hints-v1 on saved 3B runs (no model calls).

    .venv/bin/python tools/replay_repair_hints.py

For each saved run (gold 3B, selective/tools contract-check) it:
  1. compiles every emitted contract with the recipes and runs the result through the guarded verifier and
     the evaluator-only reference (upper bound and wrong-accept exposure of contract-following code);
  2. replays the lints on every candidate the 3B repair agent actually submitted, and on initially
     correct programs (false-hint rate);
  3. renders the exact hint block the repair agent would see on the first repair turn and counts tokens.
Writes results/repair_hints_v1/replay.json.
"""
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import pandas as pd  # noqa: E402
from eval.stage2_selective.runner import load_tasks  # noqa: E402
from eval.stage2_repair_hints.recipes import assemble, Unsupported  # noqa: E402
from eval.stage2_repair_hints import lint  # noqa: E402
from eval.stage2_repair_hints.agent import hint_block  # noqa: E402
from eval.stage2_repair_dev.guard import guarded_verifier  # noqa: E402
from eval.equivalence import outputs_equal  # noqa: E402
from igt.verifier import verify  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402
from igt.orchestrator import Limits  # noqa: E402
from igt.agents.code_extract import extract_code  # noqa: E402
from tools.analyze_contract_diagnostics import read_bundle  # noqa: E402

RUNS = {
    'gold': ROOT / 'kaggle_outputs/stage2/combined_gold_v1/stage2_combined_gold_3b.zip',
    'selective_checked': ROOT / 'kaggle_outputs/stage2/contract_check_v1/stage2_contract_check_selective_checked.zip',
    'tools_checked': ROOT / 'kaggle_outputs/stage2/contract_check_v1/stage2_contract_check_tools_checked.zip',
}


def traces(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            parts = name.split('/')
            if len(parts) >= 2 and parts[-2] == 'tasks' and name.endswith('.json'):
                d = json.loads(z.read(name))
                out[d['record']['task_id']] = d.get('trace')
    return out


def submitted(trace):
    """(code, failed predicate ids) for each verified candidate, in order."""
    if not trace:
        return []
    codes, pending = [], trace['initial_code']
    for e in trace['events']:
        if e['event'] == 'llm' and '"propose_fix"' in e.get('reply', ''):
            x = extract_code(e['reply'])
            pending = x['code'] if x['parse_ok'] else None
        elif e['event'] in ('verified', 'execute_failed') and pending is not None:
            codes.append((pending, e.get('failed') or ([] if e['event'] == 'verified' else ['execution'])))
            pending = None
    return codes


def main():
    tasks = {t['id']: t for s in ('dev', 'test') for _, t in load_tasks(s)}
    data, refs = {}, {}
    report = {}
    for run, path in RUNS.items():
        bundle, trace_of = read_bundle(path), traces(path)
        compiled = collections.Counter()
        lint_fail, lint_correct, rows, hint_tokens = collections.Counter(), collections.Counter(), [], []
        for tid in sorted(bundle):
            t, rec = tasks[tid], bundle[tid]['record']
            before = data.setdefault(t['data_file'], pd.read_csv(ROOT / t['data_file']))
            ref = refs.setdefault(tid, safe_run(t['reference_code'], before, **Limits().sandbox()).df_out)
            system = 'correct' if rec['final_correct'] else 'wrong_accept' if rec['outcome'] == 'ACCEPT' else 'withheld'
            if run == 'gold':
                contract = t['gold_contract']
            else:
                ex = (bundle[tid]['contract'] or {}).get('extracted') or {}
                contract = ex.get('contract') if ex.get('status') == 'ok' and rec['stage_ended'] != 'parse' else None
            if contract is None:
                compiled[(system, 'no_contract')] += 1
                continue
            columns = list(before.columns)
            try:
                code = assemble(contract, columns)
                res = safe_run(code, before, **Limits().sandbox())
                if not res.ran_ok:
                    kind = 'crash'
                else:
                    v = guarded_verifier(lambda a: verify(contract, before, a)[0], contract, before)(res.df_out)
                    kind = ('correct' if outputs_equal(ref, res.df_out)[0] else 'wrong') if v.status == 'PASS' else v.status
            except Unsupported:
                kind = 'unsupported'
            compiled[(system, 'compiled_' + kind)] += 1
            attempts = submitted(trace_of.get(tid))
            if rec['initial_correct']:
                for c in lint.hints(attempts[0][0], contract, columns) if attempts else []:
                    lint_correct[c['code']] += 1
            if system != 'correct' and attempts:
                fired = []
                for code_i, failed in attempts:
                    found = lint.hints(code_i, contract, columns)
                    fired.append([h['code'] for h in found])
                    for h in found:
                        lint_fail[h['code']] += 1
                first_code, first_failed = attempts[0]
                block = hint_block(contract, columns, first_code, first_failed)
                hint_tokens.append(len(block) // 4)
                rows.append(dict(task_id=tid, system=system, compiled=kind, attempts=len(attempts),
                                 lint_per_attempt=fired, first_failed=first_failed))
        report[run] = dict(compiled={f'{a}/{b}': n for (a, b), n in sorted(compiled.items())},
                           lint_on_failed_attempts=dict(lint_fail), lint_on_initially_correct=dict(lint_correct),
                           hint_block_tokens_approx=dict(max=max(hint_tokens, default=0),
                                                         mean=round(sum(hint_tokens) / max(len(hint_tokens), 1))),
                           failing_tasks=rows)
        print(f'== {run}')
        print(' compiled:', report[run]['compiled'])
        print(' lint on failed attempts:', dict(lint_fail))
        print(' lint on initially correct code (false hints):', dict(lint_correct))
        print(' hint block tokens ~', report[run]['hint_block_tokens_approx'])
        for r in rows:
            print(f"   {r['task_id']:18s} {r['system']:12s} compiled={r['compiled']:8s} {r['lint_per_attempt']}")
    out = ROOT / 'results/repair_hints_v1/replay.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print('Wrote', out)


if __name__ == '__main__':
    main()
