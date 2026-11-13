"""One adapter per Kaggle job; original and grounded inputs paired on OPEN development only."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from .harness import load_tasks
from .context import messages
from .method import EXPERIMENT, DECODING, runtime_hashes

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['1_5b','3b'],required=True)
    p.add_argument('--adapter',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    import pandas as pd
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from eval.stage2_clean.attention import configure_attention
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Select Kaggle GPU T4 x2')
    if any('T4' not in torch.cuda.get_device_name(i) for i in [0,1]):raise RuntimeError('This pilot expects GPU T4 x2')
    spec=json.loads((ROOT/'benchmark/contract_quality_v1/adapters.json').read_text())[a.size]
    for name,expected in spec['sha256'].items():
        if digest(a.adapter/name)!=expected:raise ValueError(f'Wrong/incomplete adapter: {name}')
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Output directory is not empty; use a fresh job or distinct directory.')
    a.out.mkdir(parents=True,exist_ok=True)
    records=a.out/'predictions.jsonl';events=a.out/'events.jsonl'
    def event(row):
        with events.open('a') as f:f.write(json.dumps(row,default=str)+'\n')
    configure_attention(torch,event)
    # Historical adapters did not record a base revision. Resolve and pin the current commit
    # for this complete job, and report it; do not claim historical revision reproduction.
    revision=model_info(spec['model']).sha
    tok=AutoTokenizer.from_pretrained(spec['model'],revision=revision)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tasks=load_tasks()
    if a.smoke:tasks=[tasks[0],next(t for t in tasks if t['pattern']=='fill_0')]
    # Check every prompt before allocating model weights or spending inference time.
    prepared=[]
    with (a.out/'input_contexts.jsonl').open('w') as context_file:
        for task in tasks:
            df=pd.read_csv(ROOT/'benchmark/contract_quality_v1'/task['data_file'])
            for variant in ['original','grounded']:
                msg=messages(task,df,variant)
                prompt=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
                count=len(tok(prompt,add_special_tokens=False)['input_ids'])
                if count+1024>8192:raise ValueError(f'Context budget exceeded for {task["id"]}: {count}')
                prepared.append((task['id'],variant,msg,prompt))
                context_file.write(json.dumps(dict(task_id=task['id'],variant=variant,messages=msg,prompt_tokens=count))+'\n')
    print(f'Preflight passed for {len(prepared)} complete prompts; no truncation.',flush=True)
    model=AutoModelForCausalLM.from_pretrained(spec['model'],revision=revision,dtype=torch.float16,
            device_map={'':0},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(model,str(a.adapter)).eval()
    plan=dict(experiment=EXPERIMENT,runtime=runtime_hashes(),size=a.size,base=spec['model'],base_revision=revision,
        adapter_hashes=spec['sha256'],suite_manifest=digest(ROOT/'benchmark/contract_quality_v1/manifest.json'),
        smoke=a.smoke,split='dev',variants=['original','grounded'],dtype='float16',quantized=False,
        decoding=DECODING,
        packages={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','pandas','numpy','accelerate']},
        limitations=['Synthetic draft; no independent annotation.', 'Historical base revision was not recorded.'])
    (a.out/'plan.json').write_text(json.dumps(plan,indent=2))
    def export():
        temporary=Path(str(a.out)+'.zip.tmp')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(a.out.iterdir()):
                if path.is_file():z.write(path,path.name)
        temporary.replace(Path(str(a.out)+'.zip'))
    completed=0
    try:
        for tid,variant,msg,prompt in prepared:
            inputs=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda:0')
            count=int(inputs['input_ids'].shape[1])
            torch.manual_seed(0);torch.cuda.manual_seed_all(0);start=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=1024,do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=tok.pad_token_id)[0,count:]
            row=dict(task_id=tid,variant=variant,raw_output=tok.decode(generated,skip_special_tokens=True),
                     prompt_tokens=count,output_tokens=len(generated),seconds=time.monotonic()-start,
                     message_sha256=hashlib.sha256(json.dumps(msg,sort_keys=True).encode()).hexdigest())
            with records.open('a') as f:f.write(json.dumps(row)+'\n')
            completed+=1;export();print(f'{completed}/{len(prepared)} {tid} {variant}',flush=True)
        (a.out/'completion.json').write_text(json.dumps(dict(complete=True,records=completed,smoke=a.smoke)))
    except Exception as error:
        event(dict(event='failure',error=repr(error)));raise
    finally:export()

if __name__=='__main__':main()
