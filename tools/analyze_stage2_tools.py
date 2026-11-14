"""Read-only audit of a tool-protocol Stage 2 run; replay CPU labels and parser control flow."""
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
from eval.stage2_tools.collect import read_bundle
from eval.stage2_tools.runner import load_tasks
from eval.stage2_tools.parser import parse_fresh,action_object
from eval.stage2_tools.context import parser_messages
from eval.stage2_tools.data_tools import DataTools
from eval.equivalence import outputs_equal
from igt.llm import ScriptedBackend
from igt.orchestrator import Limits
from igt.sandbox import safe_run
from igt.verifier import verify


def summary(rows):
    return dict(tasks=len(rows),correct=sum(r['final_correct'] for r in rows),
        incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in rows),
        withheld=sum(r['outcome']!='ACCEPT' for r in rows),rejected=sum(r['outcome']=='REJECT' for r in rows),escalated=sum(r['outcome']=='ESCALATE' for r in rows),
        initially_correct=sum(r['initial_correct'] is True for r in rows),initially_correct_withheld=sum(r['initial_correct'] is True and not r['final_correct'] for r in rows),
        repaired=sum(r['initial_correct'] is False and r['final_correct'] for r in rows),
        parser_model_calls=sum(r['parser_calls'] for r in rows),parser_tokens=sum(r['parser_tokens'] for r in rows),
        generator_repair_tokens=sum(r['generation_repair_tokens'] for r in rows),total_tokens=sum(r['tokens'] for r in rows),
        parser_tool_calls=sum(r.get('parser_tool_calls',0) for r in rows),parser_tool_failures=sum(r.get('parser_tool_failures',0) for r in rows),
        tasks_using_parser_tools=sum(r.get('parser_tool_calls',0)>0 for r in rows),repair_tool_calls=sum(r.get('tool_calls',0) for r in rows),
        recorded_seconds=sum(r['total_seconds'] for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'kaggle_outputs/stage2/tools_v1/stage2_tools_proposed.zip')
    p.add_argument('--notebook',type=Path,default=ROOT/'notebooks/executed/stage2/tools_v1/stage2_tools_05_proposed_executed.ipynb');p.add_argument('--out',type=Path,default=ROOT/'results/stage2_tools_analysis/proposed_v1');args=p.parse_args()
    plan,rows=read_bundle(args.bundle)
    with zipfile.ZipFile(args.bundle) as z:
        if z.testzip() is not None:raise ValueError('CRC failure')
        contracts={r['task_id']:json.loads(z.read('contracts/'+r['task_id']+'.json')) for r in rows}
        payloads={r['task_id']:json.loads(z.read('tasks/'+r['task_id']+'.json')) for r in rows}
    notebook=json.loads(args.notebook.read_text());template=json.loads((ROOT/'notebooks/kaggle/stage2_tools/stage2_tools_05_proposed.ipynb').read_text())
    src=lambda c:''.join(c.get('source',[]))
    errors=[dict(cell=i,name=o.get('ename'),message=o.get('evalue')) for i,c in enumerate(notebook['cells']) for o in c.get('outputs',[]) if o['output_type']=='error']
    audit=dict(bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),strict_validation=True,summary=summary(rows),
        notebook_errors=errors,notebook_source_matches=len(notebook['cells'])==len(template['cells']) and all(src(a)==src(b) for a,b in zip(notebook['cells'],template['cells'])),
        input_mismatches=[],control_flow_mismatches=[],reference_replay_mismatches=[],correctness_replay_mismatches=[],counter_mismatches=[],cases={},
        parser_status_counts=dict(collections.Counter(c['extracted']['status'] for c in contracts.values())),
        model_call_distribution=dict(collections.Counter(c['llm_calls'] for c in contracts.values())),
        invalid_response_distribution=dict(collections.Counter(c['invalid_responses'] for c in contracts.values())),
        reference_status_counts=dict(collections.Counter((r['parser_reference_verdict'] or {}).get('status','parse_failure') for r in rows)),
        reference_outcomes=[dict(status=status,outcome=outcome,correct=correct,tasks=count) for (status,outcome,correct),count in collections.Counter(((r['parser_reference_verdict'] or {}).get('status','parse_failure'),r['outcome'],r['final_correct']) for r in rows).items()],
        wrong_acceptances=[r['task_id'] for r in rows if r['outcome']=='ACCEPT' and not r['final_correct']],
        by_type={kind:summary([r for r in rows if r['transformation_type']==kind]) for kind in sorted({r['transformation_type'] for r in rows})},
        environment=dict(kaggle=plan['environment'],local_python=platform.python_version(),local_pandas=pd.__version__),
        models=plan['models'],resolved_models=plan['resolved_models'])
    expected_adapter=json.loads((ROOT/'eval/stage2_tools/adapter.json').read_text())['sha256']
    if plan['adapter_hashes']!=expected_adapter:raise ValueError('Unexpected adapter artifact')
    tasks={t['id']:t for _,t in load_tasks()}
    inputs={t['data_file']:pd.read_csv(ROOT/t['data_file']) for t in tasks.values()}
    key=hashlib.sha256(args.bundle.read_bytes()).hexdigest()[:12]
    cache=Path(tempfile.gettempdir())/('igt_tools_audit_'+key);cache.mkdir(exist_ok=True)
    replay_env=json.dumps(dict(runtime=plan['runtime'],inputs=plan['inputs'],python=platform.python_version(),pandas=pd.__version__),sort_keys=True)
    for index,row in enumerate(sorted(rows,key=lambda r:r['task_id']),1):
        tid=row['task_id'];task=tasks[tid];before=inputs[task['data_file']];saved=contracts[tid]
        initial_messages=parser_messages(task,before)
        if initial_messages!=saved['messages'][:len(initial_messages)]:audit['input_mismatches'].append(tid)
        replies=[m['content'] for m in saved['messages'] if m['role']=='assistant']
        # Disable only budget prediction for the scripted control-flow replay; real tokenizer counts
        # and latency cannot be reproduced without the GPU environment. No result here measures quality.
        class Replay(ScriptedBackend):
            def prompt_tokens(self,messages):return 0
        replay=parse_fresh(task,before,Replay(replies),'p1-v2')
        for field in ['messages','raw_output','extracted','llm_calls','tool_calls','tool_attempts','tool_failures','invalid_responses','stop_reason']:
            if replay[field]!=saved[field]:audit['control_flow_mismatches'].append([tid,field])
        for field,expected in [('parser_tool_calls',saved['tool_attempts']),('parser_tool_failures',saved['tool_failures']),('parser_calls',saved['llm_calls'])]:
            if row[field]!=expected:audit['counter_mismatches'].append([tid,field])
        assistant_tool_objects=[obj for text in replies if isinstance(obj:=action_object(text),dict) and set(obj)=={'tool','args'}]
        if len(assistant_tool_objects)!=saved['tool_attempts']:audit['counter_mismatches'].append([tid,'tool_objects'])
        contract=saved['extracted']['contract'] if saved['extracted']['status']=='ok' else None
        ref=safe_run(task['reference_code'],before,**Limits().sandbox())
        if not ref.ran_ok:raise RuntimeError(f'Reference failed: {tid}: {ref.error}')
        verdict=dataclasses.asdict(verify(contract,before,ref.df_out)[0]) if contract else None
        if verdict!=row['parser_reference_verdict']:audit['reference_replay_mismatches'].append(tid)
        case=dict(request=task['request'],record=row,contract=saved['extracted'],assistant_replies=replies,reference_verdict=verdict,reference_rows=len(ref.df_out))
        def candidate(code):
            fingerprint=hashlib.sha256((replay_env+'\0'+tid+'\0'+code).encode()).hexdigest();file=cache/(fingerprint+'.json')
            if file.exists():return json.loads(file.read_text())
            run=safe_run(code,before,**Limits().sandbox())
            result=dict(ran_ok=run.ran_ok,error=run.error,correct=False,rows=None)
            if run.ran_ok:
                result.update(correct=bool(outputs_equal(ref.df_out,run.df_out)[0]),rows=len(run.df_out),verdict=dataclasses.asdict(verify(contract,before,run.df_out)[0]) if contract else None)
            file.write_text(json.dumps(result,indent=2)+'\n');return result
        trace=payloads[tid]['trace']
        if trace:
            initial=candidate(trace['initial_code']);case['initial']=initial
            if initial['correct']!=row['initial_correct']:audit['correctness_replay_mismatches'].append([tid,'initial'])
            if row['outcome']=='ACCEPT':
                final=candidate(trace['final_code']);case['final']=final;case['final_code']=trace['final_code']
                if final['correct']!=row['final_correct']:audit['correctness_replay_mismatches'].append([tid,'final'])
        audit['cases'][tid]=case
        if index%20==0:print(f'Audited {index}/100 tool-run tasks',flush=True)
    audit['audit_only_tool_examples']={c:DataTools(inputs['data/Titanic-Dataset.csv']).call('profile_column',{'column':c}) for c in ['Age','Embarked']}
    audit['comparisons']={}
    old_path=ROOT/'results/stage2_clean_v1/proposed/tasks'
    if old_path.exists():
        old={r['task_id']:r for path in old_path.glob('*.json') for r in [json.loads(path.read_text())['record']]}
        if set(old)==set(tasks):
            audit['comparisons']['original_proposed']=dict(correct=sum(r['final_correct'] for r in old.values()),incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in old.values()),
                withheld=sum(r['outcome']!='ACCEPT' for r in old.values()),gained=[r['task_id'] for r in rows if r['final_correct'] and not old[r['task_id']]['final_correct']],lost=[r['task_id'] for r in rows if not r['final_correct'] and old[r['task_id']]['final_correct']])
    profile=ROOT/'results/stage2_independent_analysis/proposed_v1/audit.json'
    if profile.exists():
        previous=json.loads(profile.read_text());old={tid:c['record'] for tid,c in previous['cases'].items()}
        audit['comparisons']['full_profile_fp16']=dict(correct=previous['summary']['correct'],incorrect_accepted=previous['summary']['incorrect_accepted'],withheld=previous['summary']['withheld'],
            gained=[r['task_id'] for r in rows if r['final_correct'] and not old[r['task_id']]['final_correct']],lost=[r['task_id'] for r in rows if not r['final_correct'] and old[r['task_id']]['final_correct']])
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    s=audit['summary'];lines=['# Tool-enabled Stage 2 proposed run analysis','',
        '**Completed, but the intent parser made zero tool calls.** The measured result is 67 correct, 1 incorrect accepted and 32 withheld. This is not evidence that parser tool use improved results.','',
        '## Validation','',
        f'All 100 task records and 100 fresh parser records pass the strict collector and ZIP CRC check. Notebook exception outputs: {len(errors)}; source matches the tool template: {audit["notebook_source_matches"]}. Expected adapter hashes match.',
        'Exact prompt/control-flow/counter checks and CPU replays were run without model inference. Any mismatches are listed in audit.json.',
        '- Input prompt mismatches: '+str(len(audit['input_mismatches'])),
        '- Parser control-flow mismatches: '+str(len(audit['control_flow_mismatches'])),
        '- Recorded parser-counter mismatches: '+str(len(audit['counter_mismatches'])),
        '- Reference-verdict replay mismatches: '+str(len(audit['reference_replay_mismatches'])),
        '- Initial/accepted-final correctness replay mismatches: '+str(len(audit['correctness_replay_mismatches'])), '',
        '## What the parser actually did','',
        'The parser made 111 model calls: 89 tasks returned a schema-valid contract immediately; 11 returned an invalid response and then chose explicit escalation after the correction message. All 100 saved parser traces contain zero tool attempts and zero observations. No assistant response is a tool-call JSON object. Zero tool failures here means zero attempts, not proven tool reliability.',
        'The repair agent separately made 67 tool calls. Those tools inspect/repair generated code after parsing and are not the parser\'s profile_column, count_rows, inspect_rows or group_profile tools.',
        'The prompt lists the read-only dataset tools and instructs the model to query missing facts. The existing adapter was trained to emit a final contract, not tool conversations. The runtime accepts a schema-valid final contract without enforcing a prior query. The observed behavior is consistent with that training/protocol mismatch; this run does not isolate its causal components.', '',
        '## Outcomes and withholding','',
        'The parser produced 89 valid contracts and 11 explicit escalations. Of the valid contracts, 71 accept the correct reference and 18 reject it. The 32 withheld tasks comprise 11 parser escalations, 17 reference-incompatible contracts, and 4 reference-compatible tasks whose generation/repair did not produce a correct accepted result.',
        'The generator initially produced 71 correct outputs. Twelve were ultimately withheld and eight initially incorrect outputs were repaired: **71 - 12 + 8 = 67**.', '',
        '## Direct examples','',
        '- `titanic_miss_004`: asks for median Age. The first parser reply guesses 39.0 and uses unsupported `<>`; after feedback it escalates. It never queries the tool. An audit-only query returns the true input median 28.0.',
        '- `titanic_miss_002`: asks for modal Embarked. The parser chooses C without querying. The actual mode is S (644 occurrences). Correct generated code is withheld against this wrong literal.',
        '- `taxi_dedup_001`: the parser selects the correct composite key, but adds invalid null_count op=not_null and preserves row count. It escalates rather than correcting the contract; tools would not automatically solve these semantic/format errors.',
        '- `airbnb_filt_002`: the sole incorrect acceptance. The parser checks neighbourhood instead of neighbourhood_group. Generated code satisfies that wrong contract while removing every row; the reference retains 927. The exact code and replay verdicts are saved in audit.json.',
        'The two tool-query outputs in audit.json were calculated locally for explanation. They were not observations made by the model during the submitted run.', '',
        '## Comparisons and limits','',
        'This tool-protocol run gives 67/1/32, versus the earlier frozen proposed system\'s 76/2/22 and the full-profile FP16 run\'s 44/2/54. It gains 23 correct tasks in net over the full-profile run, but no parser tools were used: that gain cannot be credited to tool retrieval. It remains nine correct tasks below the original proposed result.',
        'The tool run uses the original adapter over an FP16 base, the newer package environment, new tool/escalation instructions and correction behavior. The frozen original parser used NF4; these are diagnostic experiments, not a clean tool/no-tool ablation. Model revisions, per-task gains/losses and family results are recorded in audit.json. Previously inspected test tasks are not fresh confirmatory evidence.', '',
        '## Transformation families','',
        'Correct accepted outputs out of 20 per family: column transformations 18, type conversions 15, filtering 15 (plus the single incorrect acceptance), missing-value tasks 11, deduplication 8. Details and parser escalation counts are in audit.json.','',
        '## Cost','',
        f'Parser: {s["parser_tokens"]:,} tokens across {s["parser_model_calls"]} calls. Generation/repair: {s["generator_repair_tokens"]:,} tokens. Total: {s["total_tokens"]:,}. Summed recorded parser/runtime duration is {s["recorded_seconds"]/60:.1f} minutes; it excludes setup, model loading, reference scoring and the smoke job.', '',
        '## Recommended next action','',
        'Keep the current selective-facts training running. That model learns to use automatically supplied facts; it does not train tool selection. This untrained tool-protocol experiment is separate and does not invalidate the current training run.',
        'For genuine model-selected tool behavior, a later adapter needs matching multi-turn examples that supervise assistant tool requests, tool observations and final contracts. Evaluate tool uptake specifically on tasks requiring unavailable facts, observation-to-literal correctness, scope correctness, cost and wrong-output acceptance. A runtime requirement to retrieve missing facts can prevent guessing, but forcing retrieval in code is orchestrated/hybrid retrieval and must be reported as such.',
        'Do not attach the eventual sf1 adapter directly to this old tool notebook: its protocol and strict expected adapter hashes differ. Choose and validate the input protocol before another tool-training experiment.', '',
        'Reproduce: `.venv/bin/python tools/analyze_stage2_tools.py`. No GPU calls, adapter changes or runtime/source modifications were made by this audit.']
    (args.out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print('Audit saved:',args.out,flush=True)
    print('Mismatch counts:',{key:len(audit[key]) for key in ['input_mismatches','control_flow_mismatches','counter_mismatches','reference_replay_mismatches','correctness_replay_mismatches']},flush=True)

if __name__=='__main__':main()
