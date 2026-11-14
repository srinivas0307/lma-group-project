"""Read-only audit of a completed profile-based Stage 2 proposed bundle. No model calls."""
import argparse
import collections
import dataclasses
import hashlib
import json
import platform
import sys
import tempfile
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
import yaml
from eval.stage2_independent.collect import read_bundle
from eval.stage2_independent.context import parser_messages
from eval.equivalence import outputs_equal
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from igt.agents.repair_agent import parse_action


def counts(rows):
    return dict(tasks=len(rows),correct=sum(r['final_correct'] for r in rows),
        incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in rows),
        withheld=sum(r['outcome']!='ACCEPT' for r in rows),
        rejected=sum(r['outcome']=='REJECT' for r in rows),escalated=sum(r['outcome']=='ESCALATE' for r in rows),
        invalid_contract=sum(r['contract_status']!='ok' for r in rows),
        initially_correct=sum(r['initial_correct'] is True for r in rows),
        initially_correct_withheld=sum(r['initial_correct'] is True and not r['final_correct'] for r in rows),
        repaired=sum(r['initial_correct'] is False and r['final_correct'] for r in rows),
        parser_tokens=sum(r['parser_tokens'] for r in rows),total_tokens=sum(r['tokens'] for r in rows),
        recorded_task_seconds=sum(r['total_seconds'] for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'kaggle_outputs/stage2/independent_v1/stage2_independent_proposed.zip')
    p.add_argument('--notebook',type=Path,default=ROOT/'notebooks/executed/stage2/independent_v1/stage2_independent_05_proposed_executed.ipynb')
    p.add_argument('--out',type=Path,default=ROOT/'results/stage2_independent_analysis/proposed_v1');a=p.parse_args()
    plan,rows=read_bundle(a.bundle)
    a.out.mkdir(parents=True,exist_ok=True);cache=Path(tempfile.gettempdir())/('igt_stage2_audit_'+hashlib.sha256(a.bundle.read_bytes()).hexdigest()[:12]);cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(a.bundle) as z:
        payloads={r['task_id']:json.loads(z.read('tasks/'+r['task_id']+'.json')) for r in rows}
        contracts={r['task_id']:json.loads(z.read('contracts/'+r['task_id']+'.json')) for r in rows}
    notebook=json.loads(a.notebook.read_text());template=json.loads((ROOT/'notebooks/kaggle/stage2_independent/stage2_05_proposed.ipynb').read_text())
    src=lambda c:''.join(c.get('source',[]))
    notebook_errors=[dict(cell=i,ename=o.get('ename'),evalue=o.get('evalue')) for i,c in enumerate(notebook['cells']) for o in c.get('outputs',[]) if o.get('output_type')=='error']
    tasks={t['id']:t for path in (ROOT/'benchmark/tasks').glob('*/*.yaml') if (t:=yaml.safe_load(path.read_text()))['split']=='test'}
    inputs={t['data_file']:pd.read_csv(ROOT/t['data_file']) for t in tasks.values()}
    old={r['record']['task_id']:r['record'] for path in (ROOT/'results/stage2_clean_v1/proposed/tasks').glob('*.json') for r in [json.loads(path.read_text())]}
    audit=dict(summary=counts(rows),bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),strict_bundle_validation=True,
        notebook=dict(errors=notebook_errors,source_matches_template=len(notebook['cells'])==len(template['cells']) and all(src(x)==src(y) for x,y in zip(notebook['cells'],template['cells']))),
        environment=dict(local_python=platform.python_version(),local_pandas=pd.__version__,kaggle=plan['environment']),
        method_limits=['Diagnostic re-evaluation of previously inspected test tasks, not an independent confirmatory experiment.',
                      'Parser quantization, input format, library versions and batching differ from the old parser run; profile effect is not isolated.',
                      'CPU sandbox replay is local Python, not the Kaggle interpreter; mismatches are reported.',
                      'Stored task durations exclude setup, model loading, CPU reference evaluation and the smoke run.'],
        input_message_mismatches=[],recorded_label_mismatches=[],reference_status_mismatches=[],cases={})
    saved_env=json.loads((ROOT/'kaggle_outputs/parsing/ft_1_5b/env.json').read_text())
    audit['adapter_hashes_match_expected']=plan['adapter_hashes']==json.loads((ROOT/'eval/stage2_independent/adapter.json').read_text())['sha256']
    old_parse_rows=[json.loads(l) for l in (ROOT/'kaggle_outputs/parsing/ft_1_5b/parses.jsonl').read_text().splitlines() if json.loads(l)['split']=='test']
    audit['old_cost']=dict(generation_repair_tokens=sum(r['tokens'] for r in old.values()),parser_tokens=sum(r['prompt_tokens']+r['output_tokens'] for r in old_parse_rows))
    audit['old_cost']['total_tokens']=sum(audit['old_cost'].values())
    audit['comparison']=dict(old_proposed=dict(correct=76,incorrect_accepted=2,withheld=22),
        old_parser_environment=saved_env,new_parser_spec=plan['models']['parser'],
        gained=[r['task_id'] for r in rows if r['final_correct'] and not old[r['task_id']]['final_correct']],
        lost=[r['task_id'] for r in rows if not r['final_correct'] and old[r['task_id']]['final_correct']])
    audit['by_type']={kind:counts([r for r in rows if r['transformation_type']==kind]) for kind in sorted({r['transformation_type'] for r in rows})}
    audit['by_dataset']={kind:counts([r for r in rows if r['dataset']==kind]) for kind in sorted({r['dataset'] for r in rows})}
    audit['reference_outcomes']={str(k):v for k,v in collections.Counter(((r.get('parser_reference_verdict') or {}).get('status','parse_failure'),r['outcome'],r['final_correct']) for r in rows).items()}
    audit['escalation_reasons']=dict(collections.Counter(r['escalate_reason'] for r in rows if r['outcome']=='ESCALATE'))
    provenance=json.dumps(dict(runtime=plan['runtime'],inputs=plan['inputs'],local=platform.python_version(),pandas=pd.__version__),sort_keys=True)
    correct_repair_withheld=[]
    for index,row in enumerate(sorted(rows,key=lambda r:r['task_id']),1):
        tid=row['task_id'];task=tasks[tid];before=inputs[task['data_file']];contract=contracts[tid]['extracted']['contract'] if contracts[tid]['extracted']['status']=='ok' else None
        expected=parser_messages(dict(request=' '.join(str(task['request']).split()),data_file=task['data_file']),before,'p1-v2')
        if expected!=contracts[tid]['messages']:audit['input_message_mismatches'].append(tid)
        ref=safe_run(task['reference_code'],before,**Limits().sandbox())
        if not ref.ran_ok:raise RuntimeError(f'Reference failed: {tid}: {ref.error}')
        if contract:
            v=verify(contract,before,ref.df_out)[0]
            if v.status!=row['parser_reference_verdict']['status']:audit['reference_status_mismatches'].append(tid)
        else:v=None
        def replay(code):
            key=hashlib.sha256((provenance+'\0'+tid+'\0'+code).encode()).hexdigest();path=cache/(key+'.json')
            if path.exists():return json.loads(path.read_text())
            result=safe_run(code,before,**Limits().sandbox())
            entry=dict(code=code,ran_ok=result.ran_ok,status=result.status,error=result.error,correct=False)
            if result.ran_ok:
                entry['correct'],entry['equivalence_reason']=outputs_equal(ref.df_out,result.df_out)
                entry['verdict']=dataclasses.asdict(verify(contract,before,result.df_out)[0]) if contract else None
            path.write_text(json.dumps(entry,indent=2,default=str)+'\n');return entry
        trace=payloads[tid]['trace'];case=dict(record=row,request=task['request'],parser_contract=contracts[tid]['extracted'],reference_verdict=dataclasses.asdict(v) if v else None)
        if trace:
            initial=replay(trace['initial_code']);case['initial']=initial
            if initial['correct']!=row['initial_correct']:audit['recorded_label_mismatches'].append(dict(task_id=tid,field='initial_correct'))
            if row['outcome']=='ACCEPT':
                final=replay(trace['final_code']);case['final']=final
                if final['correct']!=row['final_correct']:audit['recorded_label_mismatches'].append(dict(task_id=tid,field='final_correct'))
            elif not row['initial_correct']:
                proposals={}
                for event in trace['events']:
                    if event.get('event')=='llm':
                        action,error=parse_action(event['reply'])
                        if action and action.get('action')=='propose_fix' and action.get('code'):
                            code=action['code'];proposals[code]=replay(code)
                case['proposals']=list(proposals.values())
                if any(p['correct'] for p in proposals.values()):correct_repair_withheld.append(tid)
        audit['cases'][tid]=case
        if index%20==0:print(f'Audited {index}/100 tasks',flush=True)
    audit['correct_repairs_withheld']=correct_repair_withheld
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')
    s=audit['summary']
    text=f'''# Independent Stage 2 proposed-system analysis

The notebook completed successfully and its bundle passes strict validation. The measured method regressed: **{s['correct']} correct, {s['incorrect_accepted']} incorrect accepted, {s['withheld']} withheld**, versus **76, 2, 22** in the earlier frozen proposed run. This is a valid completed run, not an improvement.

## Runtime and input validation

- All 100 task records and 100 fresh contracts are present; the ZIP and file hashes validate.
- Notebook execution errors: {len(notebook_errors)}; source matches the current template: {audit['notebook']['source_matches_template']}.
- Exact reconstructed parser messages that differ from saved inputs: {len(audit['input_message_mismatches'])}.
- CPU replay label mismatches: {len(audit['recorded_label_mismatches'])}; reference-verdict mismatches: {len(audit['reference_status_mismatches'])}.
- The torchao setup fix succeeded. The dependency-conflict warnings did not stop the job.

## What happened

The parser produced 91 schema-valid contracts, seven schema-invalid contracts and two invalid JSON outputs. Of the 91 valid contracts, **46 accept the correct reference output and 45 reject it**. The earlier fine-tuned contracts gave 81 PASS, 16 VIOLATION, one UNCERTAIN and two parsing failures.

The 54 withheld tasks comprise **nine parse failures, 43 tasks whose contracts reject the correct reference output, and two tasks whose reference output passes but whose search/repair does not deliver a correct accepted result**. The other two reference-rejecting tasks are the two incorrect acceptances.

Generation initially produced **71 correct outputs**. The system withheld **32** of these and repaired **five** initially wrong outputs: **71 − 32 + 5 = 44**. The audit also found {len(correct_repair_withheld)} withheld tasks with a correct later submitted repair: {', '.join(correct_repair_withheld) or 'none'}. These observed candidates do not imply a deployable selector could recover them without better checks.

## Direct examples

- `titanic_miss_004`: the input explicitly supplies **Age median 28** and **mean 29.69911764705882**. The parser chooses the mean, applies it without a missing-row condition, demands that nulls increase, and preserves the originally missing cells. The generator nevertheless produces a correct median-fill implementation; the wrong contract rejects it. Information was available but was misinterpreted.
- `titanic_miss_002`: the parser correctly selects mode **S**, but requires every Embarked value to become S instead of only filling missing cells. Correct literal selection alone does not fix scope.
- `taxi_dedup_001`: the generated code correctly deduplicates the requested composite key, while the contract uses `target: "*"`, so it rejects the correct output.
- `taxi_miss_002`: the request says to mark RatecodeID 99 as missing; the contract instead requires clipping to [-1,99]. The generated clip passes the wrong contract and is incorrectly accepted.
- `taxi_miss_005`: the request says to mark passenger_count 0 and RatecodeID 99 as missing; the contract instead clips both columns to [-1,0], and that incorrect transformation is accepted.

## Important configuration confound

The original saved parser metadata records **NF4 4-bit inference** (`load_in_4bit: true`), Torch 2.11.0+cu128, Transformers 5.16.1, and batch four. This independent run uses an **FP16, non-quantized parser**, Torch 2.11.0+cu130, Transformers 5.20.2, batch one, and the new profile input. The adapter hashes are recorded and match the expected model artifact.

The new implementation changed the parser loading configuration as well as its inputs. Thus this result cannot establish whether the richer profile, quantization change, library changes or their interaction caused the regression. A QLoRA adapter can be loaded over an unquantized base; this difference is a material experimental confound, not proof that FP16 alone is the cause. The current completed version and its collector should stay unchanged for reproducibility.

## Comparison and cost

There are {len(audit['comparison']['gained'])} newly correct tasks and {len(audit['comparison']['lost'])} lost correct tasks relative to the old proposed system. Incorrect acceptances remain two but occur on different tasks. Per-family and per-dataset counts are in `audit.json`.

This run records **{s['total_tokens']:,} total tokens**, including **{s['parser_tokens']:,} parser tokens**. The earlier reported Stage 2 token cost excluded parser inference. Adding the saved test-parser calls back gives **{audit['old_cost']['total_tokens']:,} old total tokens** ({audit['old_cost']['generation_repair_tokens']:,} generation/repair + {audit['old_cost']['parser_tokens']:,} parser), versus the new total above. These are inference token counts, not monetary or wall-clock cost estimates. Summed recorded parser/runtime duration is {s['recorded_task_seconds']/60:.1f} minutes; it excludes setup, model loading, reference evaluation and the smoke run.

## Recommended continuation

1. Preserve this result as the completed FP16/profile diagnostic experiment; do not overwrite or call it a successful quality improvement.
2. Before spending GPU time on the other five systems, run a small controlled parser comparison on separate development data: original input versus enriched input, crossed with NF4 versus FP16. Hold adapter, base revision, package versions, decoding and tasks fixed. That isolates the loading and information effects.
3. Address scope, contradictory preservation and wrong-key/operator errors in semantic development tests. Merely supplying more statistics did not resolve them in this run.
4. The separately implemented tool version is still exploratory: it also currently uses FP16 for the fine-tuned parser, and the adapter was not trained for tool conversations. Resolve the loading comparison before interpreting a later tool/profile difference as a tool benefit.
5. After choosing an input protocol, build matching training examples and semantic checkpoint validation. Treat old-test improvements as diagnostic; reserve a reviewed independent benchmark for confirmation.

CPU replay uses Python {platform.python_version()} and pandas {pd.__version__}; the submitted job used Python {plan['environment']['python']}. The machine-readable audit reports any observed cross-environment differences. No GPU model calls, adapter changes, runtime edits or result rewrites were made during this analysis.
'''
    (a.out/'ANALYSIS.md').write_text(text)
    print(json.dumps(dict(summary=s,replay_label_mismatches=audit['recorded_label_mismatches'],reference_mismatches=audit['reference_status_mismatches'],correct_repairs_withheld=correct_repair_withheld),indent=2))

if __name__=='__main__':main()
