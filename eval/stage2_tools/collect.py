"""Strict collection of six independently generated Stage 2 bundles."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from .config import EXPERIMENT,SYSTEMS,SETTINGS,MODELS,THRESHOLDS
from .runner import runtime_hashes,load_tasks,write_json

def read_bundle(path,allow_scripted=False):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP members')
        files={n:z.read(n) for n in z.namelist()}
    seal=json.loads(files.pop('bundle.json'))
    if not seal['complete']:raise ValueError('Incomplete job')
    if seal['sha256']!={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}:raise ValueError('Bundle hashes differ')
    plan=json.loads(files['plan.json'])
    if plan['experiment']!=EXPERIMENT or plan['settings']!=SETTINGS or plan['models']!=MODELS or plan['thresholds']!=THRESHOLDS:raise ValueError('Method differs')
    if plan['runtime']!=runtime_hashes():raise ValueError('Runtime differs from this notebook pack')
    if not allow_scripted and (plan['backend']!='hf' or plan['smoke'] or plan['split']!='test'):raise ValueError('Not a complete real-model test run')
    tasks={t['id']:(p,t) for p,t in load_tasks(plan['split'])}
    if not allow_scripted and set(plan['task_ids'])!=set(tasks):raise ValueError('Task inventory differs')
    if len(plan['task_ids'])!=len(set(plan['task_ids'])):raise ValueError('Duplicate declared task IDs')
    expected_inputs={str(tasks[tid][0].relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(tasks[tid][0].read_bytes()).hexdigest() for tid in plan['task_ids']}
    root=Path(__file__).resolve().parents[2]
    expected_inputs.update({tasks[tid][1]['data_file']:hashlib.sha256((root/tasks[tid][1]['data_file']).read_bytes()).hexdigest() for tid in plan['task_ids']})
    if plan['inputs']!=expected_inputs:raise ValueError('Raw inputs differ')
    rows=[json.loads(b)['record'] for n,b in files.items() if n.startswith('tasks/') and n.endswith('.json')]
    if len(rows)!=len(plan['task_ids']) or {r['task_id'] for r in rows}!=set(plan['task_ids']):raise ValueError('Incomplete or duplicate task records')
    if plan['preset'] not in SYSTEMS or any(r['system']!=plan['preset'] for r in rows):raise ValueError('System mismatch')
    if any(r['outcome'] not in ['ACCEPT','REJECT','ESCALATE'] or (r['final_correct'] and r['outcome']!='ACCEPT') for r in rows):raise ValueError('Invalid outcome record')
    required={'proposed','no_contract_to_generator','baseline3_repair'}
    contracts=[json.loads(b) for n,b in files.items() if n.startswith('contracts/') and n.endswith('.json')]
    if plan['preset'] in required and (len(contracts)!=len(rows) or {r['task_id'] for r in contracts}!=set(plan['task_ids'])):raise ValueError('Fresh contracts missing')
    return plan,rows

def collect(paths,out,allow_scripted=False):
    loaded=[read_bundle(p,allow_scripted) for p in paths]
    if len(loaded)!=len(SYSTEMS) or {p['preset'] for p,r in loaded}!=set(SYSTEMS):raise ValueError('Need exactly one bundle for each of the six systems')
    if len({tuple(sorted(p['task_ids'])) for p,r in loaded})!=1:raise ValueError('Systems use different tasks')
    summaries={}
    for plan,rows in loaded:
        summaries[plan['preset']]=dict(tasks=len(rows),correct=sum(r['final_correct'] for r in rows),
            incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in rows),
            withheld=sum(r['outcome']!='ACCEPT' for r in rows),
            parse_escalations=sum(r['stage_ended']=='parse' for r in rows),
            correct_initial_withheld=sum(r['initial_correct'] is True and not r['final_correct'] for r in rows),
            reference_rejected=sum(r.get('parser_reference_verdict') is not None and r['parser_reference_verdict']['status']!='PASS' for r in rows),
            tasks_using_parser_tools=sum(r.get('parser_tool_calls',0)>0 for r in rows),parser_model_calls=sum(r['parser_calls'] for r in rows),parser_tool_calls=sum(r.get('parser_tool_calls',0) for r in rows),parser_tool_failures=sum(r.get('parser_tool_failures',0) for r in rows),parser_tokens=sum(r['parser_tokens'] for r in rows),total_tokens=sum(r['tokens'] for r in rows))
    lines=['# Tool-enabled Stage 2 results','',
           'Fresh inference with model-selected read-only parser tools. The existing adapter was not trained on this tool protocol. Existing test tasks are diagnostic because their earlier failures informed this method.',
           'Conventional checks use hand-set thresholds. The prompted parser is zero-shot p1-v3 with read-only tools; it differs from the earlier baseline.',
           'Costs include live parser inference. No old parses, tuned thresholds, cached replies or previous run outputs were loaded.','']
    for name in SYSTEMS:
        s=summaries[name];lines += [f"- {name}: **{s['correct']} correct, {s['incorrect_accepted']} incorrect accepted, {s['withheld']} withheld**; parser tools {s['parser_tool_calls']} ({s['parser_tool_failures']} failed), parser tokens {s['parser_tokens']}, total tokens {s['total_tokens']}."]
    result=dict(experiment=EXPERIMENT,model_quality_result=not allow_scripted,summary=summaries,plans=[p for p,r in loaded])
    out=Path(out)
    if out.exists():raise ValueError('Use a fresh collector output directory')
    out.mkdir(parents=True);write_json(out/'summary.json',result);(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    from .runner import export
    export(out,True)
    print('\n'.join(lines),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('bundles',type=Path,nargs=6);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    collect(a.bundles,a.out)

if __name__=='__main__':main()
