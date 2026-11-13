"""Offline audit of frozen Stage 2 traces; never calls a model or changes experiment outputs.
Run: .venv/bin/python tools/analyze_stage2_failures.py
"""
import collections
import dataclasses
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import pandas as pd
import yaml
from eval import runner_stage2 as old
from eval.stage2_clean.config import SYSTEMS, PARSES, verify_sources
from eval.stage2_clean.artifacts import load_bundle, validate_bundle
from igt.agents.repair_agent import parse_action
from igt.orchestrator.memory import code_hash


def main():
    verify_sources(ROOT)
    base = ROOT / 'results/stage2_clean_v1'
    out = ROOT / 'results/stage2_failure_analysis'
    out.mkdir(exist_ok=True)
    cache_dir = out / 'replay_cache'
    cache_dir.mkdir(exist_ok=True)
    tasks = {t['id']: t for p in (ROOT/'benchmark/tasks').glob('*/*.yaml')
             if (t := yaml.safe_load(p.read_text()))['split'] == 'test'}
    inputs = {f: pd.read_csv(ROOT/f) for f in {t['data_file'] for t in tasks.values()}}
    payloads = {s: {p['record']['task_id']: p for f in sorted((base/s/'tasks').glob('*.json'))
                   for p in [json.loads(f.read_text())]} for s in SYSTEMS}
    for s in SYSTEMS:
        validate_bundle(load_bundle(base/s))
    parses = {s: {r['task_id']: old.extract_contract(r['raw_output'])
                  for r in old.load_jsonl(ROOT/path)} for s,path in PARSES.items()}
    audit = dict(method='Offline reference/checker probe and sandbox replay of saved candidate code; no LLM calls.',
                 environment=dict(python=platform.python_version(), pandas=pd.__version__),
                 limitations=['CPU replay uses the local Python version, not Kaggle Python 3.13.16.',
                              'Reference rejection is evidence of checker/contract incompatibility, not proof that every equivalent implementation is rejected.',
                              'A reference passing the checker does not establish contract completeness.',
                              'Only submitted propose_fix candidates are replayed; code used solely in tool calls is excluded.',
                              'Accuracy counts only correct ACCEPT outputs; REJECT and ESCALATE count as unsuccessful task completion.'],
                 summaries={}, references={}, comparisons={}, cases={}, replay_label_mismatches=[])
    cache_context = hashlib.sha256(json.dumps(dict(
        python=platform.python_version(), pandas=pd.__version__,
        frozen=hashlib.sha256((ROOT/'eval/stage2_clean/frozen_inputs.json').read_bytes()).hexdigest(),
        thresholds=hashlib.sha256((ROOT/'results/stage1/baseline2/thresholds.json').read_bytes()).hexdigest(),
    ), sort_keys=True).encode()).hexdigest()
    references = {}
    for index,(tid,task) in enumerate(sorted(tasks.items()),1):
        before=inputs[task['data_file']]
        ref=old.safe_run(task['reference_code'],before,**old.Limits().sandbox())
        if not ref.ran_ok:raise RuntimeError((tid,ref.status,ref.error))
        references[tid]=ref.df_out
        checks={}
        for s in ('baseline2_repair','baseline3_repair','proposed','gold_contract'):
            if s in parses:
                parsed=parses[s][tid]
                if parsed['status']!='ok':
                    checks[s]=dict(status='PARSE_ESCALATION',parse_status=parsed['status']);continue
                contract=parsed['contract']
            else:contract=task['gold_contract'] if s=='gold_contract' else None
            v=old.make_checker(old.PRESETS[s]['verifier'],contract)(before,ref.df_out)
            checks[s]=dataclasses.asdict(v)
        audit['references'][tid]=checks
        if index%20==0:print('Reference probes',index,'/100',flush=True)
    def replay(tid,code,system):
        task=tasks[tid];key=hashlib.sha256((cache_context+'\0'+tid+'\0'+code+'\0'+system).encode()).hexdigest()
        path=cache_dir/(key+'.json')
        if path.exists():return json.loads(path.read_text())
        before=inputs[task['data_file']]
        r=old.safe_run(code,before,**old.Limits().sandbox())
        result=dict(code_hash=code_hash(code),code=code,ran_ok=r.ran_ok,execution_status=r.status,error=r.error)
        if r.ran_ok:
            equal,reason=old.outputs_equal(references[tid],r.df_out)
            contract=parses[system][tid]['contract'] if system in parses else task['gold_contract'] if system=='gold_contract' else None
            verdict=old.make_checker(old.PRESETS[system]['verifier'],contract)(before,r.df_out)
            result.update(correct=bool(equal),equivalence_reason=reason,checker=dataclasses.asdict(verdict))
        else:result.update(correct=False)
        path.write_text(json.dumps(result,indent=2,default=str)+'\n')
        return result
    baseline=payloads['baseline1']
    for system in SYSTEMS:
        rows=[p['record'] for p in payloads[system].values()]
        ids=lambda pred:[r['task_id'] for r in rows if pred(r)]
        lost=ids(lambda r:r['initial_correct'] is True and not r['final_correct'])
        fixed=ids(lambda r:r['initial_correct'] is False and r['final_correct'])
        parse=ids(lambda r:r['initial_correct'] is None)
        accepted_wrong=ids(lambda r:r['outcome']=='ACCEPT' and not r['final_correct'])
        summary=dict(initial_correct=sum(r['initial_correct'] is True for r in rows),
                     retained_correct=sum(r['initial_correct'] is True and r['final_correct'] for r in rows),
                     lost_initial_correct=lost,repaired_initial_wrong=fixed,parse_escalations=parse,
                     false_accepts=accepted_wrong,final_correct=sum(r['final_correct'] for r in rows),
                     withheld=sum(r['outcome']!='ACCEPT' for r in rows),
                     repetitions=ids(lambda r:any(e['event']=='repeated_code' for e in (payloads[system][r['task_id']]['trace'] or {}).get('trace',[]))),
                     escalations=dict(collections.Counter(r['escalate_reason'] for r in rows if r['outcome']=='ESCALATE')))
        assert summary['initial_correct']-len(lost)+len(fixed)==summary['final_correct']
        audit['summaries'][system]=summary
        loses=[tid for tid,p in payloads[system].items() if baseline[tid]['record']['final_correct'] and not p['record']['final_correct']]
        gains=[tid for tid,p in payloads[system].items() if not baseline[tid]['record']['final_correct'] and p['record']['final_correct']]
        common=[tid for tid,p in payloads[system].items() if p['trace']]
        same=sum(payloads[system][tid]['trace']['initial_code']==baseline[tid]['trace']['initial_code'] for tid in common)
        audit['comparisons'][system]=dict(loses_to_direct=loses,gains_over_direct=gains,initial_code_identical_to_direct=same,generated_task_count=len(common),
                                        direct_correct_parse_blocked=[tid for tid in parse if baseline[tid]['record']['final_correct']])
        selected=set(lost+accepted_wrong)
        if system=='proposed':selected|={r['task_id'] for r in rows if not r['final_correct']}
        if system=='baseline1':selected=set()
        for tid in sorted(selected):
            p=payloads[system][tid];trace=p['trace'];row=p['record']
            case=dict(request=tasks[tid]['request'],record=row,reference_checks=audit['references'][tid],candidates=[])
            if trace:
                code_by_hash={code_hash(trace['initial_code']):trace['initial_code']}
                for e in trace['trace']:
                    if e['event']=='llm':
                        action,error=parse_action(e['reply'])
                        if action and action['action']=='propose_fix':code_by_hash[code_hash(action['code'])]=action['code']
                submitted=[e for e in trace['trace'] if e['event'] in ('initial_code','propose_fix')]
                for e in submitted:
                    h=e['code_hash'];code=code_by_hash.get(h)
                    if code is None:raise RuntimeError(('Unrecoverable submitted code',system,tid,h))
                    item=dict(step=e['step'],kind=e['event'],**replay(tid,code,system))
                    case['candidates'].append(item)
                if case['candidates'][0]['correct']!=row['initial_correct']:
                    audit['replay_label_mismatches'].append([system,tid,'initial'])
                if row['outcome']=='ACCEPT' and case['candidates'][-1]['correct']!=row['final_correct']:
                    audit['replay_label_mismatches'].append([system,tid,'accepted'])
            audit['cases'][system+'/'+tid]=case
        print('Audited',system,'cases',len(selected),flush=True)
    assert not audit['replay_label_mismatches'], audit['replay_label_mismatches']
    (out/'audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')
    print('Done:',out/'audit.json','label mismatches:',audit['replay_label_mismatches'],flush=True)

if __name__=='__main__':main()
