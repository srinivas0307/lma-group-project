"""Fresh staged inference; no previous parse files/results or simultaneous model residency."""
import argparse
import dataclasses
import gc
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
from igt.llm import LLMClient
from igt.orchestrator import Limits
from igt.sandbox import safe_run
from igt.sandbox.environment import sandbox_environment
from igt.verifier import verify
from igt.agents.code_generator_v2 import build_generator_messages,generator_prompt_info
from igt.agents.repair_agent import AGENT_PROMPT_VERSION
from eval.equivalence import outputs_equal
from eval.stage2_selective.runner import execute,load_tasks,PARSED
from finetune.selective_facts_scopefix.context import schema_entry
from .artifacts import digest,write,export,runtime_hashes
from .config import METHODS,SYSTEMS,EXPERIMENT,MODELS,COMMON,PARSER_SETTINGS,PROTOCOLS,THRESHOLDS
from .adapters import validate
from .parsers import parse

SMOKE_IDS=('adult_col_001','titanic_miss_004')

class NoCalls:
    model='unused'
    def complete(self,*a,**kw):raise RuntimeError('Unexpected model call')

class CurrentRunGeneration:
    """Exact one-use buffer of a response freshly generated in this same run."""
    def __init__(self,item):self.item=item;self.model=item['model'];self.used=False
    def complete(self,messages,*args,**kw):
        if self.used or messages!=self.item['messages']:raise ValueError('Generation replay is not the fresh matching response')
        self.used=True
        return self.item['raw_output'],self.item['prompt_tokens'],self.item['output_tokens']

def run(method,out,adapter=None,preset='proposed',split='test',smoke=False,packages=None,backend_factory=None):
    if method not in METHODS or preset not in SYSTEMS:raise ValueError('Unknown method/system')
    out=Path(out)
    if out.exists():raise ValueError('Use a fresh output directory; previous results are not reused')
    tasks=load_tasks(split)
    if smoke:
        ids=SMOKE_IDS if split=='test' else tuple(t['id'] for _,t in tasks[:2])
        tasks=[item for tid in ids for item in tasks if item[1]['id']==tid]
        if len(tasks)!=2:raise ValueError('Missing smoke tasks')
    out.mkdir(parents=True);loaded=[];active=None;scripted=backend_factory is not None
    try:
        if preset in ['proposed','no_contract_to_generator'] and not scripted:
            if adapter is None:raise ValueError('Selected trained adapter required')
            adapter_hashes=validate(method,adapter)
        else:adapter_hashes={}
        inputs={str(p.relative_to(ROOT)):digest(p) for p,t in tasks}
        inputs.update({t['data_file']:digest(ROOT/t['data_file']) for p,t in tasks})
        plan=dict(experiment=EXPERIMENT,method=method,protocol=PROTOCOLS[method],preset=preset,common=COMMON,parser_settings=PARSER_SETTINGS[method],models=MODELS,generator_prompt=generator_prompt_info(preset in ['proposed','gold_contract']),repair_prompt=AGENT_PROMPT_VERSION,
            limits=Limits().to_dict(),thresholds=THRESHOLDS,runtime=runtime_hashes(),inputs=inputs,task_ids=[t['id'] for _,t in tasks],split=split,smoke=smoke,
            backend='scripted_pipeline_check' if scripted else 'hf',adapter_hashes=adapter_hashes,source_manifest_sha256=digest(ROOT/'_stage2_dual_manifest.json') if (ROOT/'_stage2_dual_manifest.json').exists() else None,environment=dict(python=platform.python_version(),pandas=pd.__version__),resolved_models={},
            limitations=['Previously inspected test tasks are diagnostic, not a fresh independent benchmark.','Parser input protocols and adapters differ; this is not an isolated tool-access ablation.'])
        write(out/'plan.json',plan)
        if not scripted:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()<1:raise RuntimeError('A CUDA GPU is required')
            if any(torch.cuda.get_device_properties(i).total_memory/2**30<12 for i in range(torch.cuda.device_count())):raise RuntimeError('At least 12 GiB memory per visible GPU required')
            from finetune.selective_facts.attention import configure_attention
            def event(e):
                with (out/'gpu_events.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
            configure_attention(torch,event)
            plan['environment'].update(versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate','pandas','numpy']},
                cuda=torch.version.cuda,gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
            from .backend import Backend
            backend_factory=lambda role:Backend(role,method,adapter if role=='parser' else None)
        def load(role):
            if loaded:raise RuntimeError('More than one model would be resident')
            b=backend_factory(role);loaded.append(b)
            plan['resolved_models'][role]=dict(model=b.model,revision=getattr(b,'revision',None),device=getattr(b,'device',None))
            write(out/'plan.json',plan);return b
        def unload(b):
            if getattr(b,'refusals',None):
                plan.setdefault('model_context_refusals',[]).extend(b.refusals);write(out/'plan.json',plan)
            if hasattr(b,'close'):b.close()
            loaded.remove(b);gc.collect()
        data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks};parses={};generated={}
        if preset in PARSED:
            parser=load('agent' if preset=='baseline3_repair' else 'parser')
            for i,(_,task) in enumerate(tasks,1):
                active=task['id'];parsed=parse(method,task,data[task['data_file']],parser,prompted=preset=='baseline3_repair')
                parses[active]=parsed;write(out/'contracts'/f'{active}.json',parsed)
                if i%10==0 or smoke:export(out)
                print(f'PARSE {i}/{len(tasks)} {active}: {parsed["extracted"]["status"]}',flush=True)
            unload(parser);del parser
        generation_tasks=[t for _,t in tasks if preset not in PARSED or parses[t['id']]['extracted']['status']=='ok']
        if generation_tasks:
            generator=load('generator')
            for i,task in enumerate(generation_tasks,1):
                active=task['id'];before=data[task['data_file']]
                contract=task['gold_contract'] if preset=='gold_contract' else (parses[active]['extracted']['contract'] if preset=='proposed' else None)
                msg=build_generator_messages(dict(request=task['request'],data_file=task['data_file']),{task['data_file']:schema_entry(before)},contract)
                client=LLMClient(generator);start=time.monotonic();reply=client.generate(msg,max_new_tokens=512,temperature=0,seed=0)
                item=dict(task_id=active,messages=msg,raw_output=reply.text,prompt_tokens=reply.prompt_tokens,output_tokens=reply.output_tokens,model=generator.model,seconds=time.monotonic()-start)
                generated[active]=item;write(out/'generation'/f'{active}.json',item)
                if i%10==0 or smoke:export(out)
                print(f'GENERATE {i}/{len(generation_tasks)} {active}',flush=True)
            unload(generator);del generator
        agent=load('agent') if generation_tasks and preset!='baseline1' else NoCalls()
        with sandbox_environment(packages) as environment:
            write(out/'sandbox.json',environment)
            for i,(_,task) in enumerate(tasks,1):
                active=task['id'];before=data[task['data_file']];parsed=parses.get(active)
                public={k:task[k] for k in ['id','request','data_file','dataset','split','transformation_type']}
                if preset=='gold_contract':parsed=dict(extracted=dict(status='ok',contract=task['gold_contract']),tokens=0,llm_calls=0)
                generator_buffer=CurrentRunGeneration(generated[active]) if active in generated else NoCalls()
                result=execute(public,before,preset,parsed,generator_buffer,agent)
                reference=safe_run(task['reference_code'],before,**Limits().sandbox())
                if not reference.ran_ok:raise RuntimeError(f'Reference evaluation failed: {task["id"]}: {reference.error}')
                r=result['record'];r.update(method=method,protocol=PROTOCOLS[method],parser_tool_calls=len(parsed.get('tool_calls',[])) if parsed else 0,
                   parser_tool_failures=parsed.get('tool_failures',0) if parsed else 0,normalization_events=len(parsed.get('normalizations',[])) if parsed else 0,
                   generation_seconds=generated.get(active,{}).get('seconds',0))
                r['initial_correct']=bool(result['initial_output'] is not None and outputs_equal(reference.df_out,result['initial_output'])[0]) if r['stage_ended']!='parse' else None
                r['final_correct']=bool(r['outcome']=='ACCEPT' and outputs_equal(reference.df_out,result['final_output'])[0])
                contract=parsed['extracted']['contract'] if parsed and parsed['extracted']['status']=='ok' else None
                r['parser_reference_verdict']=dataclasses.asdict(verify(contract,before,reference.df_out)[0]) if contract else None
                r['total_seconds']=r['parser_seconds']+r['generation_seconds']+r['runtime_seconds']
                result.pop('initial_output');result.pop('final_output');write(out/'tasks'/f'{active}.json',result)
                export(out);print(f'DONE {i}/{len(tasks)} {active}: {r["outcome"]}',flush=True)
        if loaded:unload(agent)
        write(out/'completion.json',dict(complete=True,method=method,preset=preset,tasks=len(tasks),smoke=smoke,retrained=False));export(out,True)
        return out
    except BaseException as exc:
        write(out/'failure.json',dict(task_id=active,error=repr(exc),traceback=traceback.format_exc()));export(out,False);raise
    finally:
        for b in list(loaded):
            if hasattr(b,'close'):b.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--adapter',type=Path);p.add_argument('--preset',choices=SYSTEMS,default='proposed');p.add_argument('--split',choices=['dev','test'],default='test');p.add_argument('--smoke',action='store_true');p.add_argument('--packages',type=Path)
    a=p.parse_args();run(a.method,a.out,a.adapter,a.preset,a.split,a.smoke,a.packages)
if __name__=='__main__':main()
