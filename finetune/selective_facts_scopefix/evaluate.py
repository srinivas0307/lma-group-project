"""Free-generation semantic scoring on OPEN development tasks, never training or selection."""
import argparse
import collections
import dataclasses
import json
import time
from pathlib import Path
import pandas as pd
import yaml
from igt.agents.intent_parser import extract_contract
from igt.verifier import verify
from igt.sandbox import safe_run
from igt.sandbox.environment import sandbox_environment
from igt.orchestrator import Limits
from eval.equivalence import outputs_equal
from eval.contract_quality.harness import prepare,probe
from .context import messages,selective_facts,FactUnavailable,PROTOCOL
from finetune.selective_facts.provenance import ROOT,write,digest
from .provenance import hashes
from finetune.selective_facts.evaluate import development_tasks,summarize

def evaluate(adapter,out,packages=None,smoke=False):
    out=Path(out)
    if out.exists():raise ValueError('Evaluation output already exists')
    out.mkdir(parents=True)
    tasks=development_tasks(smoke)
    from .adapter import validate_existing
    adapter_hashes=validate_existing(adapter)
    import torch
    from finetune.selective_facts.backend import Backend
    from finetune.selective_facts.attention import configure_attention
    configure_attention(torch,lambda event:None)
    backend=Backend('parser',adapter)
    rows=[]
    try:
        with sandbox_environment(packages):
            for index,(task,cohort,root) in enumerate(tasks,1):
                before=pd.read_csv(root/task['data_file'])
                record=dict(task_id=task['id'],cohort=cohort,request=task['request'],prompt_tokens=0,output_tokens=0,
                            all_correct_accepted=False,any_wrong_accepted=False,parse_status='fact_unavailable')
                try:
                    facts=selective_facts(task['request'],before)
                    record['facts']=facts
                    msg=messages(task,before)
                except FactUnavailable as exc:
                    record['error']=str(exc);record['facts']=[]
                else:
                    start=time.monotonic();text,nt,no=backend.complete(msg,1024,0,1,0)
                    extracted=extract_contract(text)
                    record.update(messages=msg,raw_output=text,prompt_tokens=nt,output_tokens=no,parse_status=extracted['status'],
                                  predicted_contract=extracted.get('contract'),seconds=time.monotonic()-start)
                    if extracted['status']=='ok':
                        if cohort=='synthetic':record.update(probe(extracted['contract'],prepare(task,root)))
                        else:
                            reference=safe_run(task['reference_code'],before,**Limits().sandbox())
                            if not reference.ran_ok:raise RuntimeError(f'Reference failed: {task["id"]}: {reference.error}')
                            if outputs_equal(reference.df_out,before)[0]:raise ValueError('No-op is not a negative')
                            record.update(all_correct_accepted=verify(extracted['contract'],before,reference.df_out)[0].status=='PASS',
                                          any_wrong_accepted=verify(extracted['contract'],before,before)[0].status=='PASS')
                rows.append(record);write(out/'tasks'/f'{cohort}_{task["id"]}.json',record)
                print(f'DEV {index}/{len(tasks)} {task["id"]}: {record["parse_status"]}',flush=True)
        summary=summarize(rows)
        inputs={str((root/task['data_file']).relative_to(ROOT)):digest(root/task['data_file']) for task,_,root in tasks}
        provenance=dict(runtime=hashes(),inputs=inputs,synthetic_manifest=digest(ROOT/'benchmark/contract_quality_v1/manifest.json'),
            original_task_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'benchmark/tasks').glob('*/*.yaml') if yaml.safe_load(p.read_text())['split']=='dev'},
            adapter_hashes={p.name:digest(p) for p in Path(adapter).iterdir() if p.is_file() and p.suffix in ['.json','.safetensors','.txt']},
            protocol=PROTOCOL,adapter_trained_protocol='p1-v2+sf1',adapter_validation=adapter_hashes)
        result=dict(complete=True,smoke=smoke,summary=summary,provenance=provenance,limitations=['Open development data; synthetic draft not independently reviewed.','Original tasks have only a wrong-no-op probe; synthetic tasks have targeted wrong outputs.'])
        from .comparison import compare
        comparison=compare(rows,smoke)
        write(out/'comparison.json',comparison)
        write(out/'summary.json',result);return result
    finally:backend.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--packages',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.out.exists():p.error('Output already exists; use a fresh directory')
    from finetune.selective_facts.train import export
    try:
        evaluate(a.adapter,a.out,a.packages,a.smoke)
        write(a.out/'completion.json',dict(complete=True,smoke=a.smoke,protocol=PROTOCOL,retrained=False))
    except BaseException as exc:
        import traceback
        if a.out.exists():write(a.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if a.out.exists():export(a.out,(a.out/'completion.json').exists())
if __name__=='__main__':main()
