"""Strict two-method comparison, with optional same-method six-system collection later."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from igt.orchestrator import Limits
from igt.agents.code_generator_v2 import generator_prompt_info
from igt.agents.repair_agent import AGENT_PROMPT_VERSION
from .config import EXPERIMENT,METHODS,SYSTEMS,COMMON,MODELS,PARSER_SETTINGS,PROTOCOLS,THRESHOLDS
from .adapters import SPECS
from .artifacts import ROOT,digest,runtime_hashes,write,export
from .runner import load_tasks,PARSED

def read_bundle(path,allow_scripted=False):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid ZIP')
        contents={n:z.read(n) for n in z.namelist()}
    seal=json.loads(contents.pop('bundle.json'))
    if not seal['complete'] or seal['sha256']!={n:hashlib.sha256(b).hexdigest() for n,b in contents.items()}:raise ValueError('Incomplete/changed bundle')
    if 'failure.json' in contents:raise ValueError('Failure record in result bundle')
    plan=json.loads(contents['plan.json']);method=plan['method'];preset=plan['preset']
    if method not in METHODS or preset not in SYSTEMS:raise ValueError('Unknown method/system')
    if plan['experiment']!=EXPERIMENT or plan['protocol']!=PROTOCOLS[method] or plan['common']!=COMMON or plan['models']!=MODELS or plan['parser_settings']!=PARSER_SETTINGS[method] or plan['limits']!=Limits().to_dict() or plan['thresholds']!=THRESHOLDS:raise ValueError('Different experiment settings')
    if plan['generator_prompt']!=generator_prompt_info(preset in ['proposed','gold_contract']) or plan['repair_prompt']!=AGENT_PROMPT_VERSION:raise ValueError('Generator/repair prompts differ')
    if plan['runtime']!=runtime_hashes():raise ValueError('Runtime differs from this source pack')
    if not allow_scripted and (plan['backend']!='hf' or plan['smoke'] or plan['split']!='test'):raise ValueError('Not a full real-model test run')
    if not allow_scripted and (not (ROOT/'_stage2_dual_manifest.json').exists() or plan.get('source_manifest_sha256')!=digest(ROOT/'_stage2_dual_manifest.json')):raise ValueError('Different source/data notebook pack; use its matching collector')
    inventory={t['id']:(p,t) for p,t in load_tasks(plan['split'])}
    ids=plan['task_ids']
    if len(ids)!=len(set(ids)) or any(i not in inventory for i in ids):raise ValueError('Invalid task inventory')
    if not allow_scripted and set(ids)!=set(inventory):raise ValueError('Full task coverage missing')
    expected={str(inventory[i][0].relative_to(ROOT)):digest(inventory[i][0]) for i in ids}
    expected.update({inventory[i][1]['data_file']:digest(ROOT/inventory[i][1]['data_file']) for i in ids})
    if plan['inputs']!=expected:raise ValueError('Raw input hashes differ')
    completion=json.loads(contents['completion.json'])
    if not completion['complete'] or completion['tasks']!=len(ids) or completion['method']!=method or completion['preset']!=preset or completion['smoke']!=plan['smoke']:raise ValueError('Completion mismatch')
    rows=[json.loads(b)['record'] for n,b in contents.items() if n.startswith('tasks/') and n.endswith('.json')]
    if len(rows)!=len(ids) or {r['task_id'] for r in rows}!=set(ids):raise ValueError('Missing/duplicate task records')
    contract_items=[json.loads(b) for n,b in contents.items() if n.startswith('contracts/') and n.endswith('.json')]
    contracts={r['task_id']:r for r in contract_items}
    if preset in PARSED and (len(contract_items)!=len(ids) or set(contracts)!=set(ids)):raise ValueError('Fresh contracts missing')
    generations=[json.loads(b) for n,b in contents.items() if n.startswith('generation/') and n.endswith('.json')]
    required_generation={r['task_id'] for r in rows if r['stage_ended']!='parse'}
    if len(generations)!=len(required_generation) or {g['task_id'] for g in generations}!=required_generation:raise ValueError('Fresh initial generations missing/duplicated')
    generated={g['task_id']:g for g in generations}
    if not allow_scripted and preset in ['proposed','no_contract_to_generator']:
        if plan['adapter_hashes']!=SPECS[method]['sha256']:raise ValueError('Different selected adapter')
    for r in rows:
        if r['method']!=method or r['system']!=preset or r['protocol']!=PROTOCOLS[method] or r['outcome'] not in ['ACCEPT','REJECT','ESCALATE'] or (r['final_correct'] and r['outcome']!='ACCEPT'):raise ValueError('Invalid outcome identity')
        if r['task_id'] in generated and generated[r['task_id']]['prompt_tokens']+generated[r['task_id']]['output_tokens']>r['generation_repair_tokens']:raise ValueError('Initial generation cost missing')
        if r['tokens']!=r['parser_tokens']+r['generation_repair_tokens']:raise ValueError('Token accounting mismatch')
        if preset in PARSED:
            parsed=contracts[r['task_id']]
            if parsed['tokens']!=r['parser_tokens'] or parsed['llm_calls']!=r['parser_calls'] or parsed['extracted']['status']!=r['contract_status']:raise ValueError('Parser accounting mismatch')
            if len(parsed.get('tool_calls',[]))!=r['parser_tool_calls'] or parsed.get('tool_failures',0)!=r['parser_tool_failures']:raise ValueError('Tool accounting mismatch')
            if sum(not t['observation']['ok'] for t in parsed.get('tool_calls',[]))!=r['parser_tool_failures']:raise ValueError('Failed tool observations do not match counters')
            expected_protocol='p1-v3+norm1' if preset=='baseline3_repair' else PROTOCOLS[method]
            if parsed['parser_protocol']!=expected_protocol:raise ValueError('Parser protocol differs')
    if not allow_scripted:
        for role in (['parser'] if preset in ['proposed','no_contract_to_generator'] else []):
            if plan['resolved_models'].get(role,{}).get('revision')!=MODELS[role]['revision']:raise ValueError('Model revision differs')
        for role in ['generator','agent']:
            if role in plan['resolved_models'] and plan['resolved_models'][role]['revision']!=MODELS[role]['revision']:raise ValueError('Generator/repair revision differs')
    return plan,rows

def summarize(rows):
    return dict(tasks=len(rows),correct=sum(r['final_correct'] for r in rows),incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in rows),withheld=sum(r['outcome']!='ACCEPT' for r in rows),
      parse_escalations=sum(r['stage_ended']=='parse' for r in rows),initial_correct_lost=sum(r['initial_correct'] is True and not r['final_correct'] for r in rows),
      repaired=sum(r['initial_correct'] is False and r['final_correct'] for r in rows),reference_incompatible=sum(r.get('parser_reference_verdict') is not None and r['parser_reference_verdict']['status']!='PASS' for r in rows),
      parser_calls=sum(r['parser_calls'] for r in rows),parser_tokens=sum(r['parser_tokens'] for r in rows),parser_tools=sum(r['parser_tool_calls'] for r in rows),parser_tool_failures=sum(r['parser_tool_failures'] for r in rows),
      generation_repair_tokens=sum(r['generation_repair_tokens'] for r in rows),total_tokens=sum(r['tokens'] for r in rows),normalized_tasks=sum(r['normalization_events']>0 for r in rows),summed_task_seconds=sum(r['total_seconds'] for r in rows))

def collect(paths,out,allow_scripted=False,six=False):
    out=Path(out)
    if out.exists():raise ValueError('Collector output already exists')
    loaded=[read_bundle(p,allow_scripted) for p in paths]
    if six:
        if len(loaded)!=6 or {p['preset'] for p,r in loaded}!=set(SYSTEMS) or len({p['method'] for p,r in loaded})!=1:raise ValueError('Need all six systems of one method')
    elif len(loaded)!=2 or {p['method'] for p,r in loaded}!=set(METHODS) or any(p['preset']!='proposed' for p,r in loaded):raise ValueError('Need one proposed bundle for each method')
    if len({tuple(sorted(p['task_ids'])) for p,r in loaded})!=1:raise ValueError('Different task inventories')
    if not allow_scripted:
        if len({json.dumps(p['environment'].get('versions',{}),sort_keys=True) for p,r in loaded})!=1 or len({p['environment']['python'] for p,r in loaded})!=1:raise ValueError('Different software environments; run both notebooks with matching package/Python versions')
    names=[p['preset'] if six else p['method'] for p,r in loaded]
    summaries={name:summarize(rows) for name,(plan,rows) in zip(names,loaded)}
    paired={}
    if not six:
        left={r['task_id']:r for p,rows in loaded if p['method']=='selective' for r in rows};right={r['task_id']:r for p,rows in loaded if p['method']=='tools' for r in rows}
        paired=dict(only_selective_correct=[i for i in sorted(left) if left[i]['final_correct'] and not right[i]['final_correct']],only_tools_correct=[i for i in sorted(left) if right[i]['final_correct'] and not left[i]['final_correct']])
    report=dict(experiment=EXPERIMENT,model_quality_result=not allow_scripted,summary=summaries,paired=paired,plans=[p for p,r in loaded],
       note='Diagnostic end-to-end comparison of different parser methods/adapters on previously inspected tasks; not independent test evidence. Timings include per-task parser/generation/repair but exclude model loading/setup.')
    lines=['# Fresh Stage 2 parser comparison','',report['note'],'']
    for name,s in summaries.items():lines.append(f"- {name}: {s['correct']} correct, {s['incorrect_accepted']} incorrect accepted, {s['withheld']} withheld; parser calls {s['parser_calls']}, parser tool attempts {s['parser_tools']} ({s['parser_tool_failures']} failed), total tokens {s['total_tokens']}.")
    out.mkdir(parents=True);write(out/'summary.json',report);(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n');export(out,True)
    print('\n'.join(lines),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundles',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);p.add_argument('--six',action='store_true')
    a=p.parse_args();collect(a.bundles,a.out,six=a.six)
