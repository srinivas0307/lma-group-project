"""Strict two-method comparison, with optional same-method six-system collection later."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from igt.orchestrator import Limits
from igt.agents.code_generator_v2 import generator_prompt_info
from .config import REPAIR_VERSION as AGENT_PROMPT_VERSION
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
    if not allow_scripted and (not (ROOT/'_stage2_combined_gold_manifest.json').exists() or plan.get('source_manifest_sha256')!=digest(ROOT/'_stage2_combined_gold_manifest.json')):raise ValueError('Different source/data notebook pack; use its matching collector')
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
        if r.get('repair_version')!=AGENT_PROMPT_VERSION or r.get('repair_arm')!='combined' or r.get('schema_guard')!='declared-output-columns-v1':raise ValueError('Repair/guard record identity mismatch')
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
    task_payloads={json.loads(b)['record']['task_id']:json.loads(b) for n,b in contents.items() if n.startswith('tasks/') and n.endswith('.json')}
    for r in rows:
        trace=task_payloads[r['task_id']]['trace']
        if r['stage_ended']=='parse':
            if trace is not None or r['fresh_recovery_requests'] or r['repair_diagnostic_seconds']:raise ValueError('Parse stop has repair activity')
            continue
        if trace is None:raise ValueError('Repair trace missing')
        events=trace['events']
        if r['fresh_recovery_requests']!=sum(e['event']=='fresh_recovery_requested' for e in events):raise ValueError('Recovery accounting mismatch')
        if r['repair_diagnostic_seconds']!=sum(e.get('seconds',0) for e in events if e['event']=='repair_evidence'):raise ValueError('Diagnostic accounting mismatch')
        g=generated[r['task_id']]
        if r['generation_repair_tokens']!=g['prompt_tokens']+g['output_tokens']+sum(e['tokens'] for e in events if e['event']=='llm'):raise ValueError('Repair token accounting mismatch')
        if not any(e['event']=='repair_evidence' for e in events):raise ValueError('Combined repair evidence missing')
    if not allow_scripted:
        required_roles={'generator','agent'} if required_generation else set()
        if not required_roles.issubset(plan['resolved_models']):raise ValueError('Resolved model metadata missing')
        for role in required_roles:
            if not plan['resolved_models'][role]['model'].startswith(MODELS[role]['id']):raise ValueError('Resolved model identity differs')
        for role in (['parser'] if preset in ['proposed','no_contract_to_generator'] else []):
            if plan['resolved_models'].get(role,{}).get('revision')!=MODELS[role]['revision']:raise ValueError('Model revision differs')
        for role in ['generator','agent']:
            if role in plan['resolved_models'] and plan['resolved_models'][role]['revision']!=MODELS[role]['revision']:raise ValueError('Generator/repair revision differs')
            if role in plan['resolved_models']:
                precision=plan['resolved_models'][role].get('precision') or {}
                if precision.get('quantization')!='none' or precision.get('compute_dtype')!='torch.float16' or precision.get('floating_parameter_dtypes')!=['torch.float16']:raise ValueError('Generator/repair must have verified unquantized FP16 weights')
    if plan.get('contract_source')!='task.gold_contract' or plan['adapter_hashes'] or 'parser' in plan['resolved_models']:raise ValueError('Unexpected parser/adapter in gold run')
    if len(contract_items)!=len(ids) or set(contracts)!=set(ids):raise ValueError('Gold contracts missing')
    for row in rows:
        c=contracts[row['task_id']]
        if c['extracted']['contract']!=inventory[row['task_id']][1]['gold_contract'] or c.get('source')!='task.gold_contract':raise ValueError('Gold contract differs from task')
        if row['parser_calls'] or row['parser_tokens'] or row['parser_tool_calls'] or row['parser_tool_failures']:raise ValueError('Unexpected parser cost')
    return plan,rows

def summarize(rows):
    return dict(tasks=len(rows),correct=sum(r['final_correct'] for r in rows),incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in rows),withheld=sum(r['outcome']!='ACCEPT' for r in rows),
      parse_escalations=sum(r['stage_ended']=='parse' for r in rows),initial_correct_lost=sum(r['initial_correct'] is True and not r['final_correct'] for r in rows),
      repaired=sum(r['initial_correct'] is False and r['final_correct'] for r in rows),reference_incompatible=sum(r.get('parser_reference_verdict') is not None and r['parser_reference_verdict']['status']!='PASS' for r in rows),
      parser_calls=sum(r['parser_calls'] for r in rows),parser_tokens=sum(r['parser_tokens'] for r in rows),parser_tools=sum(r['parser_tool_calls'] for r in rows),parser_tool_failures=sum(r['parser_tool_failures'] for r in rows),
      generation_repair_tokens=sum(r['generation_repair_tokens'] for r in rows),total_tokens=sum(r['tokens'] for r in rows),normalized_tasks=sum(r['normalization_events']>0 for r in rows),summed_task_seconds=sum(r['total_seconds'] for r in rows),fresh_recovery_requests=sum(r['fresh_recovery_requests'] for r in rows),repair_diagnostic_seconds=sum(r['repair_diagnostic_seconds'] for r in rows),combined_reference_incompatible=sum(r.get('combined_reference_verdict') is not None and r['combined_reference_verdict']['status']!='PASS' for r in rows))

def collect(gold,out,predicted=(),allow_scripted=False):
    """Validate complete bundles and report paired oracle-versus-predicted outcomes."""
    out=Path(out)
    if out.exists():raise ValueError('Use a fresh collector directory')
    gp,gr=read_bundle(gold,allow_scripted)
    loaded={'gold':(gp,gr)}
    from eval.stage2_combined_3b.collect import read_bundle as read_predicted
    for path in predicted:
        plan,rows=read_predicted(path,allow_scripted)
        method=plan['method']
        if method in loaded:raise ValueError('Duplicate predicted method')
        for key in ['inputs','task_ids','models','common','generator_prompt','repair_prompt','limits','thresholds','split']:
            if plan[key]!=gp[key]:raise ValueError('Unmatched comparison setting: '+key)
        if not allow_scripted and plan['environment']!=gp['environment']:raise ValueError('Different Python/software/GPU environment; retain separate summaries or rerun matched configurations')
        loaded[method]=(plan,rows)
    summaries={name:summarize(rows) for name,(plan,rows) in loaded.items()}
    gold_rows={r['task_id']:r for r in gr};paired={}
    for name,(plan,rows) in loaded.items():
        if name=='gold':continue
        other={r['task_id']:r for r in rows}
        paired[name]=dict(only_gold_correct=[i for i in sorted(gold_rows) if gold_rows[i]['final_correct'] and not other[i]['final_correct']],
                         only_predicted_correct=[i for i in sorted(gold_rows) if other[i]['final_correct'] and not gold_rows[i]['final_correct']])
    report=dict(experiment=EXPERIMENT,model_quality_result=not allow_scripted,summary=summaries,paired=paired,
                plans={name:plan for name,(plan,rows) in loaded.items()},
                note='Oracle diagnostic on previously inspected tasks. Gold replaces the contract in generation and verification; initial programs can differ. This is the total contract-source effect, not a repair-only ablation or guaranteed upper bound. Setup/model loading excluded from task time; gold has zero parser cost.')
    lines=['# Combined 3B gold-contract diagnostic','',report['note'],'']
    for name,s in summaries.items():lines.append(f"- {name}: {s['correct']} correct, {s['incorrect_accepted']} incorrect accepted, {s['withheld']} withheld; total tokens {s['total_tokens']} (parser {s['parser_tokens']}).")
    for name,items in paired.items():
        lines.append(f"- Gold versus {name}: {len(items['only_gold_correct'])} tasks correct only with gold; {len(items['only_predicted_correct'])} correct only with predicted contracts.")
    out.mkdir(parents=True);write(out/'summary.json',report);(out/'SUMMARY.md').write_text('\n'.join(lines)+'\n');export(out,True)
    print('\n'.join(lines),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gold',required=True,type=Path);p.add_argument('--predicted',nargs='*',type=Path,default=[]);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();collect(a.gold,a.out,a.predicted)
