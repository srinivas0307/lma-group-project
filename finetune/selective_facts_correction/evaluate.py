"""Three paired arms share fresh first answers; gold/reference access occurs only after parsing."""
import argparse
import collections
import importlib.metadata
import json
import platform
from pathlib import Path
import pandas as pd
from finetune.selective_facts.evaluate import development_tasks,summarize
from finetune.selective_facts_scopefix.adapter import validate_existing
from eval.contract_quality.harness import prepare,probe
from eval.equivalence import outputs_equal
from igt.sandbox import safe_run
from igt.sandbox.environment import sandbox_environment
from igt.orchestrator import Limits
from igt.verifier import verify
from .parser import attempt_correction,ARMS,PROTOCOL
from .artifacts import ROOT,digest,write,hashes,export

def task_set(smoke=False):
    tasks=development_tasks()
    if smoke:
        # Smoke exercises normalization, correction and a no-change control; no success assumed.
        return [next(item for item in tasks if item[0].get('pattern')==pattern) for pattern in ['filter_0','edit_1','dedup_0']]
    return tasks

def summarize_arms(records):
    summaries={}
    for arm in ARMS:
        rows=[]
        for r in records:
            v=r['variants'][arm];replies=r['trace']['replies'] if arm=='corrected' else r['trace']['replies'][:1]
            rows.append(dict(cohort=r['cohort'],parse_status=v['status'],all_correct_accepted=v['score']['all_correct_accepted'],any_wrong_accepted=v['score']['any_wrong_accepted'],
                prompt_tokens=sum(x['prompt_tokens'] for x in replies),output_tokens=sum(x['output_tokens'] for x in replies)))
        summaries[arm]=summarize(rows)
    return summaries

def comparison(records):
    summaries=summarize_arms(records);paired={}
    for left,right in [('control','normalized'),('normalized','corrected'),('control','corrected')]:
        def success(r,a):s=r['variants'][a]['score'];return s['all_correct_accepted'] and not s['any_wrong_accepted']
        paired[left+'_vs_'+right]=dict(gained=[r['task_id'] for r in records if success(r,right) and not success(r,left)],lost=[r['task_id'] for r in records if success(r,left) and not success(r,right)])
    historical=json.loads((Path(__file__).parent/'historical.json').read_text())
    return dict(current=summaries,paired=paired,historical=historical,
        normalization_tasks=[r['task_id'] for r in records if r['trace'].get('initial_normalization')],
        correction_tasks=[r['task_id'] for r in records if r['trace']['attempted']],
        unresolved_tasks=[r['task_id'] for r in records if r['variants']['corrected']['status']!='ok'],
        note='Three fresh paired arms; historical 85/100 is context only. Corrected contracts are not chosen using reference scores.')

def run(adapter,out,packages=None,smoke=False):
    out=Path(out)
    if out.exists():raise ValueError('Existing output directory; choose a fresh run')
    out.mkdir(parents=True)
    backend=None
    try:
        checked=validate_existing(adapter);tasks=task_set(smoke)
        import torch
        from finetune.selective_facts.backend import Backend
        from finetune.selective_facts.attention import configure_attention
        events=[];configure_attention(torch,events.append)
        env=dict(python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),visible_gpus=torch.cuda.device_count(),
            versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate','pandas','numpy']})
        provenance=dict(protocol=PROTOCOL,adapter_trained_protocol='p1-v2+sf1',adapter_validation=checked,runtime=hashes(),environment=env,attention_events=events,
            input_hashes={str((root/task['data_file']).relative_to(ROOT)):digest(root/task['data_file']) for task,_,root in tasks},
            task_sha256=__import__('hashlib').sha256(json.dumps([t for t,_,_ in tasks],sort_keys=True).encode()).hexdigest(),
            settings=dict(nf4=True,compute_dtype='float16',temperature=0,max_new_tokens=1024,max_model_replies=2,context_limit=8192))
        write(out/'run_info.json',provenance)
        backend=Backend('parser',adapter);records=[]
        with sandbox_environment(packages):
            for index,(task,cohort,root) in enumerate(tasks,1):
                before=pd.read_csv(root/task['data_file'])
                # Only request/data filename and input DataFrame cross the inference boundary.
                result=attempt_correction(dict(request=task['request'],data_file=task['data_file']),before,backend)
                record=dict(task_id=task['id'],cohort=cohort,request=task['request'],pattern=task.get('pattern',task.get('transformation_type')),**result)
                prepared=None;reference=None
                # Inference and correction are finished before any gold/reference is executed.
                for arm in ARMS:
                    candidate=record['variants'][arm];score=dict(all_correct_accepted=False,any_wrong_accepted=False)
                    if candidate['status']=='ok':
                        if cohort=='synthetic':
                            if prepared is None:prepared=prepare(task,root)
                            score=probe(candidate['contract'],prepared)
                        else:
                            if reference is None:
                                ref=safe_run(task['reference_code'],before,**Limits().sandbox())
                                if not ref.ran_ok:raise RuntimeError(f'Reference failed: {task["id"]}: {ref.error}')
                                reference=ref.df_out
                                if outputs_equal(reference,before)[0]:raise ValueError('No-op is not a wrong probe')
                            score=dict(all_correct_accepted=verify(candidate['contract'],before,reference)[0].status=='PASS',any_wrong_accepted=verify(candidate['contract'],before,before)[0].status=='PASS')
                    # Independent dictionaries avoid sharing score mutations across arms.
                    record['variants'][arm]=dict(candidate,score=score)
                records.append(record);write(out/'tasks'/f'{cohort}_{task["id"]}.json',record)
                print(f'{index}/{len(tasks)} {task["id"]}: '+json.dumps({a:record['variants'][a]['score']['all_correct_accepted'] for a in ARMS})+f" correction={record['trace']['attempted']}",flush=True)
                if index%10==0:export(out,False)
        comp=comparison(records);write(out/'comparison.json',comp)
        write(out/'summary.json',dict(complete=True,smoke=smoke,arms=comp['current'],actual_model_calls=sum(r['trace']['llm_calls'] for r in records),
            actual_tokens=sum(r['trace']['prompt_tokens']+r['trace']['output_tokens'] for r in records),
            task_count=len(records),protocol=PROTOCOL,limitations=['Open development tasks used to design this method; not independent test performance.','Original tasks have only a wrong-no-op probe.','Input diagnostics are incomplete; passing them does not prove semantic correctness.']))
        write(out/'completion.json',dict(complete=True,smoke=smoke,retrained=False,protocol=PROTOCOL));export(out,True)
        print(json.dumps(comp,indent=2),flush=True)
        return comp
    except BaseException as exc:
        import traceback
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));export(out,False);raise
    finally:
        if backend is not None:backend.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--packages',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.adapter,a.out,a.packages,a.smoke)
if __name__=='__main__':main()
