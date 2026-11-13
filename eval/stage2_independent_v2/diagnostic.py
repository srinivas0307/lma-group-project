"""Controlled parser-only development comparison. No test selection or code generation."""
import argparse
import dataclasses
import importlib.metadata
import json
import time
import zipfile
from pathlib import Path
import pandas as pd
from igt.agents.intent_parser import build_messages, extract_contract
from igt.llm import LLMClient
from igt.sandbox import safe_run
from igt.sandbox.environment import sandbox_environment
from igt.orchestrator import Limits
from igt.verifier import verify
from eval.equivalence import outputs_equal
from .config import MODELS
from .context import parser_messages, schema_entry
from .runner import ROOT, load_tasks, digest, runtime_hashes, write_json, export

ARMS = ('original_nf4', 'profile_nf4', 'original_fp16', 'profile_fp16')

def messages_for(task, before, variant):
    public = dict(request=' '.join(str(task['request']).split()), data_file=task['data_file'])
    if variant == 'profile':
        return parser_messages(public, before, 'p1-v2')
    if variant != 'original':
        raise ValueError(variant)
    return build_messages(public, {public['data_file']: schema_entry(before)}, [], version='p1-v2')

def evaluate(extracted, before, reference):
    # Reference is used only after inference. A weak contract may pass it: check a wrong no-op too.
    contract = extracted.get('contract') if extracted['status'] == 'ok' else None
    noop_is_wrong = not outputs_equal(reference, before)[0]
    return dict(reference_verdict=dataclasses.asdict(verify(contract, before, reference)[0]) if contract else None,
                wrong_noop_accepted=(verify(contract, before, before)[0].status == 'PASS') if contract and noop_is_wrong else None,
                noop_is_wrong=noop_is_wrong)

def run_arm(arm, adapter, out, packages=None, limit=None):
    if arm not in ARMS: raise ValueError(arm)
    out = Path(out)
    if out.exists(): raise ValueError('Use a fresh output directory')
    adapter = Path(adapter)
    spec = json.loads((Path(__file__).parent/'adapter.json').read_text())
    for name, expected in spec['sha256'].items():
        if digest(adapter/name) != expected: raise ValueError(f'Wrong adapter: {name}')
    tasks = load_tasks('dev')
    if limit is not None:
        if limit < 1: raise ValueError('limit must be positive')
        tasks = tasks[:limit]
    variant, precision = arm.split('_')
    # Each arm runs in a fresh subprocess; this setting cannot affect the Stage 2 job.
    MODELS['parser']['four_bit'] = precision == 'nf4'
    plan = dict(experiment='stage2-parser-loading-diagnostic-v1', arm=arm, split='dev',
                smoke=limit is not None, task_ids=[t['id'] for _, t in tasks],
                parser=dict(MODELS['parser']), adapter_hashes=spec['sha256'], runtime=runtime_hashes(),
                inputs={str(p.relative_to(ROOT)):digest(p) for p,_ in tasks},
                decoding=dict(seed=0,temperature=0,max_new_tokens=1024,batch_size=1),
                packages={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','pandas','numpy']})
    plan['inputs'].update({t['data_file']:digest(ROOT/t['data_file']) for _,t in tasks})
    write_json(out/'plan.json',plan)
    backend = None
    try:
        import torch
        if torch.__version__ != '2.11.0+cu130' or torch.cuda.device_count() != 2:
            raise RuntimeError('Requires pinned torch 2.11.0+cu130 and GPU T4 x2')
        if any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):
            raise RuntimeError('Select GPU T4 x2')
        from eval.stage2_clean.attention import configure_attention
        def event(value):
            with (out/'gpu_events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        configure_attention(torch,event)
        from .backend import Backend
        backend = Backend('parser',adapter)
        data = {t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks}
        for _,t in tasks: backend.check_prompt(messages_for(t,data[t['data_file']],variant),1024)
        with sandbox_environment(packages):
            for i,(_,task) in enumerate(tasks,1):
                before = data[task['data_file']]
                messages = messages_for(task,before,variant)
                client = LLMClient(backend)
                start = time.monotonic()
                response = client.generate(messages,max_new_tokens=1024,temperature=0,seed=0)
                elapsed = time.monotonic()-start
                extracted = extract_contract(response.text)
                reference = safe_run(task['reference_code'],before,**Limits().sandbox())
                if not reference.ran_ok: raise RuntimeError(f'Reference failed: {task["id"]}: {reference.error}')
                record = dict(task_id=task['id'],messages=messages,raw_output=response.text,extracted=extracted,
                              tokens=client.tokens_used,seconds=elapsed,**evaluate(extracted,before,reference.df_out))
                write_json(out/'tasks'/f'{task["id"]}.json',record)
                export(out)
                print(f'{arm}: {i}/{len(tasks)} {task["id"]} {extracted["status"]}',flush=True)
        export(out,True)
    except BaseException as exc:
        write_json(out/'failure.json',dict(error=repr(exc)))
        export(out)
        raise
    finally:
        if backend is not None: backend.close()

def summarize(directory):
    directory=Path(directory);plans=[];rows={};summary={}
    for arm in ARMS:
        with zipfile.ZipFile((directory/arm).with_suffix('.zip')) as z:
            if len(z.namelist()) != len(set(z.namelist())): raise ValueError('Duplicate ZIP members')
            files={n:z.read(n) for n in z.namelist()}
        import hashlib
        seal=json.loads(files.pop('bundle.json'))
        if not seal['complete'] or seal['sha256']!={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}:
            raise ValueError('Incomplete or corrupt diagnostic')
        plan=json.loads(files['plan.json']);plans.append(plan)
        if plan['experiment']!='stage2-parser-loading-diagnostic-v1' or plan['arm']!=arm or plan['split']!='dev' or plan['smoke']: raise ValueError('Not a full development diagnostic')
        expected_parser=dict(MODELS['parser'],four_bit=arm.endswith('_nf4'))
        if plan['parser']!=expected_parser: raise ValueError('Parser configuration differs')
        if plan['runtime']!=runtime_hashes(): raise ValueError('Runtime differs')
        tasks=load_tasks('dev')
        if plan['task_ids']!=[t['id'] for _,t in tasks]: raise ValueError('Development inventory differs')
        inputs={str(p.relative_to(ROOT)):digest(p) for p,_ in tasks}
        inputs.update({t['data_file']:digest(ROOT/t['data_file']) for _,t in tasks})
        if plan['inputs']!=inputs: raise ValueError('Development inputs differ')
        records=[json.loads(b) for n,b in files.items() if n.startswith('tasks/') and n.endswith('.json')]
        if len(records)!=len(plan['task_ids']) or {r['task_id'] for r in records}!=set(plan['task_ids']):
            raise ValueError('Task coverage differs')
        rows[arm]={r['task_id']:r for r in records}
        summary[arm]=dict(tasks=len(records),valid_contracts=sum(r['extracted']['status']=='ok' for r in records),
                         reference_pass=sum((r['reference_verdict'] or {}).get('status')=='PASS' for r in records),
                         wrong_noop_accepted=sum(r['wrong_noop_accepted'] is True for r in records),
                         wrong_noop_tasks=sum(r['noop_is_wrong'] for r in records),tokens=sum(r['tokens'] for r in records))
    for key in ('task_ids','inputs','runtime','adapter_hashes','packages','decoding'):
        if any(p[key]!=plans[0][key] for p in plans): raise ValueError(f'Uncontrolled difference: {key}')
    paired={}
    def passes(record): return (record['reference_verdict'] or {}).get('status')=='PASS'
    for a,b in [('original_nf4','profile_nf4'),('original_fp16','profile_fp16'),('profile_fp16','profile_nf4'),('original_fp16','original_nf4')]:
        paired[a+' -> '+b]=dict(gained=[t for t in rows[a] if not passes(rows[a][t]) and passes(rows[b][t])],
                                lost=[t for t in rows[a] if passes(rows[a][t]) and not passes(rows[b][t])])
    write_json(directory/'summary.json',dict(summary=summary,paired_reference_pass=paired,plans=plans))
    lines=['# Parser development diagnostic','', 'Reference pass measures false rejection, not complete contract correctness. The wrong no-op probe covers only one error type. No automatic winner or Stage 2 accuracy is claimed.','']
    lines += [f'- {a}: {s["reference_pass"]}/{s["tasks"]} reference passes; {s["valid_contracts"]} schema-valid; {s["wrong_noop_accepted"]}/{s["wrong_noop_tasks"]} wrong no-ops accepted.' for a,s in summary.items()]
    (directory/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    export(directory,True)
    print('\n'.join(lines),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS);p.add_argument('--adapter',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--packages',type=Path);p.add_argument('--limit',type=int);p.add_argument('--summarize',action='store_true')
    a=p.parse_args()
    if a.summarize: summarize(a.out)
    else:
        if not a.arm or not a.adapter:p.error('--arm and --adapter required')
        run_arm(a.arm,a.adapter,a.out,a.packages,a.limit)

if __name__=='__main__':main()
