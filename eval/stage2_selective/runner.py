"""Independent Stage 2: raw CSV -> fresh contract -> generation -> check/repair -> evaluation."""
import argparse
import dataclasses
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
import yaml
from igt.agents.intent_parser import extract_contract
from igt.agents.code_extract import extract_code
from igt.agents.code_generator_v2 import build_generator_messages
from igt.agents.repair_agent import run_repair,AgentConfig
from igt.llm import LLMClient
from igt.orchestrator import Limits
from igt.sandbox import safe_run
from igt.sandbox.environment import sandbox_environment
from igt.verifier import verify
from igt.types import Verdict
from igt.baselines.conventional_checks import measure,judge
from eval.equivalence import outputs_equal
from .config import EXPERIMENT,SYSTEMS,MODELS,SETTINGS,THRESHOLDS
from .context import parser_messages,schema_entry
from finetune.selective_facts.context import FactUnavailable

PARSED={'proposed','no_contract_to_generator','baseline3_repair'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def runtime_hashes():
    paths=list((ROOT/'src/igt').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.json'))
    paths += list((ROOT/'finetune/selective_facts').glob('*.py'))
    paths += [ROOT/'eval/equivalence.py',ROOT/'eval/contract_quality/context.py',ROOT/'eval/stage2_clean/attention.py',ROOT/'requirements.txt']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)

def export(directory,complete=False):
    files={str(p.relative_to(directory)):p.read_bytes() for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix!='.tmp'}
    path=directory.with_suffix('.zip');temp=Path(str(path)+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
        z.writestr('bundle.json',json.dumps(dict(complete=complete,sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})))
    temp.replace(path)

def load_tasks(split='test'):
    items=[]
    for path in sorted((ROOT/'benchmark/tasks').glob('*/*.yaml')):
        task=yaml.safe_load(path.read_text())
        if task['split']==split:items.append((path,task))
    if not items:raise ValueError('No input tasks')
    return items

def parse_fresh(task,before,backend,prompt):
    try:messages=parser_messages(dict(request=' '.join(str(task['request']).split()),data_file=task['data_file']),before,prompt)
    except FactUnavailable as exc:
        return dict(task_id=task['id'],messages=[],raw_output='',extracted=dict(status='input_facts_unavailable',contract=None,error=str(exc)),tokens=0,llm_calls=0,seconds=0,model=backend.model)
    client=LLMClient(backend)  # no disk cache and no earlier parse file
    start=time.monotonic();response=client.generate(messages,max_new_tokens=1024,temperature=0,seed=0)
    return dict(task_id=task['id'],messages=messages,raw_output=response.text,
                extracted=extract_contract(response.text),tokens=client.tokens_used,llm_calls=client.n_calls,
                seconds=time.monotonic()-start,model=backend.model)

def execute(task,before,preset,parsed,generator,agent):
    """Runtime boundary: does not receive gold contract or reference program except the explicit oracle arm."""
    limits=Limits();start=time.monotonic()
    contract=parsed['extracted']['contract'] if parsed and parsed['extracted']['status']=='ok' else None
    status=parsed['extracted']['status'] if parsed else 'not_used'
    parser_tokens=parsed['tokens'] if parsed and preset in PARSED else 0
    parser_calls=parsed['llm_calls'] if parsed and preset in PARSED else 0
    record=dict(task_id=task['id'],system=preset,split=task['split'],dataset=task['dataset'],
                transformation_type=task['transformation_type'],contract_status=status,parser_tokens=parser_tokens,
                parser_calls=parser_calls,parser_seconds=parsed.get('seconds',0) if parsed and preset in PARSED else 0,generation_repair_tokens=0,tokens=parser_tokens,llm_calls=parser_calls,
                initial_correct=None,final_correct=False)
    if preset in PARSED and contract is None:
        record.update(outcome='ESCALATE',stage_ended='parse',attempts=0,escalate_reason='Invalid live parser contract',runtime_seconds=time.monotonic()-start)
        return dict(record=record,trace=None,initial_output=None,final_output=None)
    checker=(lambda after:Verdict('PASS',[],[])) if preset=='baseline1' else (
        (lambda after:Verdict(**judge(measure(before,after),THRESHOLDS))) if preset=='baseline2_repair' else
        (lambda after:verify(contract,before,after)[0]))
    public=dict(request=task['request'],data_file=task['data_file'])
    schema={task['data_file']:schema_entry(before)}
    gc_client=LLMClient(generator)
    response=gc_client.generate(build_generator_messages(public,schema,contract if preset in {'proposed','gold_contract'} else None),max_new_tokens=512,temperature=0,seed=0)
    extracted=extract_code(response.text);code=extracted['code'] if extracted['parse_ok'] else response.text
    ac=LLMClient(agent,token_budget=limits.token_budget)
    result=run_repair(dict(request=task['request']),code,before,checker,ac,
        AgentConfig(mode='none' if preset=='baseline1' else 'full',limits=limits,temperature=0),
        contract_text=json.dumps(contract) if contract else None,schema_text=messages_schema(schema[task['data_file']]))
    first=safe_run(code,before,**limits.sandbox())
    record.update(outcome=result.outcome,stage_ended='repair' if result.attempts else 'check',attempts=result.attempts,
        generation_repair_tokens=gc_client.tokens_used+ac.tokens_used,tokens=parser_tokens+gc_client.tokens_used+ac.tokens_used,
        llm_calls=parser_calls+gc_client.n_calls+ac.n_calls,tool_calls=result.tool_calls,
        unsafe_attempts=result.unsafe_attempts,invalid_actions=result.invalid_actions,
        escalate_reason=result.escalate_reason,runtime_seconds=time.monotonic()-start)
    return dict(record=record,trace=dict(generator_reply=response.text,initial_code=code,final_code=result.final_code,events=result.trace),
                initial_output=first.df_out if first.ran_ok else None,final_output=result.final_output)

def messages_schema(entry):
    return f"{entry['n_rows']} rows. Columns:\n"+'\n'.join(f"{c['name']}: {c['dtype']}, {c['nulls']} missing" for c in entry['columns'])+'\nFirst 3 rows:\n'+entry['head_csv']

def run(preset,out,adapter=None,split='test',limit=None,backend_factory=None,packages=None):
    if preset not in SYSTEMS:raise ValueError(preset)
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory; this workflow does not reuse earlier runs')
    tasks=load_tasks(split);tasks=tasks[:limit] if limit else tasks
    out.mkdir(parents=True,exist_ok=True)
    inputs={str(p.relative_to(ROOT)):digest(p) for p,t in tasks}
    inputs.update({t['data_file']:digest(ROOT/t['data_file']) for p,t in tasks})
    adapter_hashes={}
    if preset in {'proposed','no_contract_to_generator'} and backend_factory is None:
        if adapter is None:raise ValueError('Attach the trained 1.5B adapter')
        adapter=Path(adapter)
        from finetune.selective_facts.adapter import validate_adapter
        adapter_hashes=validate_adapter(adapter)
    plan=dict(experiment=EXPERIMENT,preset=preset,settings=SETTINGS,models=MODELS,thresholds=THRESHOLDS,
        runtime=runtime_hashes(),inputs=inputs,task_ids=[t['id'] for p,t in tasks],split=split,smoke=limit is not None,
        backend='scripted_pipeline_check' if backend_factory else 'hf',adapter_hashes=adapter_hashes,
        environment=dict(python=platform.python_version(),pandas=pd.__version__),resolved_models={})
    write_json(out/'plan.json',plan)
    loaded=[];active=None
    try:
        if backend_factory is None:
            import importlib.metadata
            import torch
            if torch.__version__!='2.11.0+cu130' or torch.cuda.device_count()!=2:raise RuntimeError('Requires pinned torch 2.11.0+cu130 and GPU T4 x2')
            if any('T4' not in torch.cuda.get_device_name(i) for i in [0,1]):raise RuntimeError('Select GPU T4 x2')
            from eval.stage2_clean.attention import configure_attention
            def event(e):
                with (out/'gpu_events.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
            configure_attention(torch,event)
            from .backend import Backend
            backend_factory=lambda role:Backend(role,adapter if role=='parser' else None)
            plan['environment']['packages']={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate','numpy']}
            try:plan['environment']['torchao']=importlib.metadata.version('torchao')
            except importlib.metadata.PackageNotFoundError:plan['environment']['torchao']=None
        def load(role):
            b=backend_factory(role);loaded.append(b)
            plan['resolved_models'][role]=dict(model=b.model,revision=getattr(b,'revision',None))
            write_json(out/'plan.json',plan)
            return b
        def unload(b):
            if hasattr(b,'close'):b.close()
            loaded.remove(b)
            gc.collect()
        data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for p,t in tasks}
        parses={}
        if preset in PARSED:
            parser=load('agent' if preset=='baseline3_repair' else 'parser')
            prompt='p1-v3' if preset=='baseline3_repair' else 'p1-v2'
            if hasattr(parser,'check_prompt'):
                unavailable=0
                for path,task in tasks:
                    try:parser.check_prompt(parser_messages(task,data[task['data_file']],prompt),1024)
                    except FactUnavailable:unavailable+=1
                print(f'Unsupported statistic requests will explicitly escalate: {unavailable}',flush=True)
                print('Parser prompt preflight passed for every task.',flush=True)
            for index,(path,task) in enumerate(tasks,1):
                active=task['id']
                parsed=parse_fresh(task,data[task['data_file']],parser,prompt);parses[active]=parsed
                write_json(out/'contracts'/f'{active}.json',parsed)
                export(out);print(f'FRESH CONTRACT {index}/{len(tasks)}: {active} {parsed["extracted"]["status"]}',flush=True)
            unload(parser)
            del parser
        generator=load('generator')
        agent=load('agent') if preset!='baseline1' else generator
        with sandbox_environment(packages) as sandbox:
            write_json(out/'sandbox.json',sandbox)
            for index,(path,task) in enumerate(tasks,1):
                active=task['id'];before=data[task['data_file']]
                # Runtime sees public fields only; ground truth is handled after it finishes.
                public={k:task[k] for k in ['id','request','data_file','dataset','split','transformation_type']}
                parsed=parses.get(active)
                if preset=='gold_contract':parsed=dict(extracted=dict(status='ok',contract=task['gold_contract']),tokens=0,llm_calls=0)
                result=execute(public,before,preset,parsed,generator,agent)
                reference=safe_run(task['reference_code'],before,**Limits().sandbox())
                if not reference.ran_ok:raise RuntimeError(f'Reference evaluation failed: {reference.error}')
                rec=result['record']
                rec['initial_correct']=bool(result['initial_output'] is not None and outputs_equal(reference.df_out,result['initial_output'])[0]) if rec['stage_ended']!='parse' else None
                rec['final_correct']=bool(rec['outcome']=='ACCEPT' and outputs_equal(reference.df_out,result['final_output'])[0])
                contract=parsed['extracted']['contract'] if parsed and parsed['extracted']['status']=='ok' else None
                rec['total_seconds']=rec['runtime_seconds']+rec['parser_seconds']
                rec['parser_reference_verdict']=dataclasses.asdict(verify(contract,before,reference.df_out)[0]) if contract else None
                result.pop('initial_output');result.pop('final_output')
                write_json(out/'tasks'/f'{active}.json',result)
                export(out);print(f'TASK DONE {index}/{len(tasks)}: {active} {rec["outcome"]}',flush=True)
        export(out,True)
        print(f'COMPLETE: {preset} {len(tasks)}/{len(tasks)} -> {out.with_suffix(".zip")}',flush=True)
        return out
    except BaseException as e:
        write_json(out/'failure.json',dict(task_id=active,error=repr(e),traceback=traceback.format_exc()))
        export(out)
        raise
    finally:
        for b in list(loaded):
            if hasattr(b,'close'):b.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=SYSTEMS,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--split',choices=['dev','test'],default='test');p.add_argument('--limit',type=int);p.add_argument('--packages',type=Path)
    a=p.parse_args();run(a.preset,a.out,a.adapter,a.split,a.limit,packages=a.packages)

if __name__=='__main__':main()
