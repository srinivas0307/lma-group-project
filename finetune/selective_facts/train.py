"""Single-GPU QLoRA; matching input protocol, full validation loss, semantic epoch scoring."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile
from .config import MODEL,REVISION,TRAIN
from .context import PROTOCOL
from .provenance import ROOT,hashes,digest,write,verify_data
from .evaluate import eligible


def tokenize(rows,tok,max_length):
    result=[]
    for row in rows:
        messages=row['messages']
        if [m['role'] for m in messages]!=['system','user','assistant']:raise ValueError('sf1 expects one assistant answer')
        prompt=tok.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
        full=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        if not full.startswith(prompt):raise ValueError('Chat template prefix changed')
        p=tok(prompt,add_special_tokens=False)['input_ids'];f=tok(full,add_special_tokens=False)['input_ids']
        if f[:len(p)]!=p:raise ValueError('Token boundary changed; do not silently mask the wrong tokens')
        if len(f)>max_length:raise ValueError(f'{row["id"]}: {len(f)} tokens exceeds {max_length}; no examples are truncated or dropped')
        if len(f)<=len(p):raise ValueError('Empty supervised answer')
        result.append(dict(input_ids=f,attention_mask=[1]*len(f),labels=[-100]*len(p)+f[len(p):]))
    return result

class Collator:
    def __init__(self,pad):self.pad=pad
    def __call__(self,rows):
        import torch
        n=max(len(r['input_ids']) for r in rows)
        return {key:torch.tensor([r[key]+[value]*(n-len(r[key])) for r in rows]) for key,value in [('input_ids',self.pad),('attention_mask',0),('labels',-100)]}

def export(out,complete=False):
    out=Path(out);target=out.with_suffix('.zip');tmp=Path(str(target)+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        seals={}
        for p in sorted(out.rglob('*')):
            if not p.is_file() or p.suffix=='.tmp' or 'checkpoints' in p.relative_to(out).parts:continue
            name=str(p.relative_to(out));z.write(p,name);seals[name]=digest(p)
        z.writestr('bundle.json',json.dumps(dict(complete=complete,sha256=seals)))
    tmp.replace(target)

def train(data,out,baseline_adapter,packages=None,smoke=False):
    out=Path(out)
    if out.exists():raise ValueError('Use a fresh training output directory')
    out.mkdir(parents=True)
    model=None
    try:
        manifest=verify_data(data)
        config=dict(TRAIN)
        if smoke:config.update(epochs=1,max_steps=2)
        import torch
        import transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainerCallback
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        if torch.__version__!='2.11.0+cu130' or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Launch training with CUDA_VISIBLE_DEVICES=0 on Kaggle T4 x2; training must see exactly one GPU')
        from .training_api import make_arguments
        args=make_arguments(config,out/'trainer')
        random.seed(config['seed']);torch.manual_seed(config['seed'])
        def semantic(adapter,destination,original=False):
            command=[sys.executable,'-m','finetune.selective_facts.evaluate','--adapter',str(adapter),'--out',str(destination)]
            if packages:command+=['--packages',str(packages)]
            if original:command+=['--original']
            if smoke:command+=['--smoke']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='1')
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            return json.loads((destination/'summary.json').read_text())
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        tok.padding_side='right'
        raw={split:[json.loads(l) for l in (Path(data)/f'sft_{split}.jsonl').read_text().splitlines()] for split in ['train','val']}
        if smoke:raw={s:rows[:8] for s,rows in raw.items()}
        datasets={s:tokenize(rows,tok,config['max_seq_len']) for s,rows in raw.items()}
        if not all(datasets.values()):raise ValueError('Empty training or validation dataset')
        lengths={s:dict(examples=len(v),min=min(len(r['input_ids']) for r in v),max=max(len(r['input_ids']) for r in v)) for s,v in datasets.items()}
        baseline=semantic(baseline_adapter,out/'baseline',original=True)
        import platform
        plan=dict(experiment='selective-facts-sft-v1',protocol=PROTOCOL,base=MODEL,revision=REVISION,config=config,smoke=smoke,
            source_hashes=hashes(),data_manifest_sha256=digest(Path(data)/'manifest.json'),data_manifest=manifest,
            token_lengths=lengths,training_arguments=args.to_dict(),python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','accelerate','pandas','numpy']},
            device_count=torch.cuda.device_count(),world_size=1,effective_batch=config['batch_size']*config['gradient_accumulation'],
            gpu=torch.cuda.get_device_name(0),validation_examples='all designated validation examples, except explicit smoke job',
            inference_nf4_double_quant=False,training_nf4_double_quant=False)
        write(out/'run_info.json',plan)
        from .attention import configure_attention
        def event(value):
            with (out/'gpu_events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        configure_attention(torch,event)
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16,device_map={'':0},attn_implementation='sdpa',
             quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=False))
        model.config.use_cache=False
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
        model=get_peft_model(model,LoraConfig(r=config['lora_r'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
        candidates=[]
        class Monitor(TrainerCallback):
            def on_log(self,args,state,control,logs=None,**kwargs):
                if logs:
                    with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(dict(step=state.global_step,**logs))+'\n')
            def on_epoch_end(self,args,state,control,**kwargs):
                name=f'epoch_{state.global_step:06d}'
                folder=out/'checkpoints'/name
                model.save_pretrained(folder);tok.save_pretrained(folder)
                (folder/'igt_prompt_version.txt').write_text(PROTOCOL)
                write(folder/'input_protocol.json',dict(protocol=PROTOCOL,base=MODEL,revision=REVISION,context_sha256=digest(Path(__file__).parent/'context.py')))
                result=semantic(folder,out/'semantic'/name)
                row=dict(name=name,step=state.global_step,epoch=state.epoch,summary=result['summary'],adapter_sha256=result['provenance']['adapter_hashes'],eligible=False if smoke else eligible(result['summary'],baseline['summary']))
                candidates.append(row)
                # Keep an inspectable candidate even when no checkpoint meets screening.
                pool=[r for r in candidates if r['eligible']] or candidates
                best=max(pool,key=lambda r:(sum(s['joint_success'] for s in r['summary'].values()),-sum(s['wrong_accepted'] for s in r['summary'].values()),-r['step']))
                destination=out/'adapter'
                if destination.exists():
                    # This folder is created exclusively by this run; replace only its known files.
                    for p in destination.iterdir():
                        if p.is_file():p.unlink()
                        else:raise RuntimeError('Unexpected directory in generated adapter')
                destination.mkdir(exist_ok=True)
                for p in (out/'checkpoints'/best['name']).iterdir():
                    if p.is_file():shutil.copy2(p,destination/p.name)
                selection=dict(eligible=best['eligible'],smoke=smoke,selected=best,all_candidates=candidates,
                    baseline=baseline['summary'],rule='No cohort loses correct acceptance or increases wrong acceptance; total joint successes strictly improve.',
                    limitations='Finite open-development screening on a partly synthetic draft, not a safety guarantee or independent test.')
                write(destination/'selection.json',selection);write(out/'selection.json',selection)
                export(out,False)
        trainer=Trainer(model=model,args=args,train_dataset=datasets['train'],eval_dataset=datasets['val'],data_collator=Collator(tok.pad_token_id),callbacks=[Monitor()])
        start=time.monotonic();result=trainer.train()
        plan.update(training_seconds=time.monotonic()-start,steps=result.global_step,train_loss=result.training_loss,
                    final_eval=trainer.evaluate(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        write(out/'run_info.json',plan)
        if not candidates:raise RuntimeError('No semantically evaluated checkpoint')
        write(out/'completion.json',dict(complete=True,smoke=smoke,eligible=json.loads((out/'selection.json').read_text())['eligible']))
        export(out,True)
        print('COMPLETE:',out.with_suffix('.zip'),flush=True)
        print((out/'selection.json').read_text(),flush=True)
    except BaseException as exc:
        import traceback
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));export(out,False);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline-adapter',type=Path,required=True);p.add_argument('--packages',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();train(a.data,a.out,a.baseline_adapter,a.packages,a.smoke)
if __name__=='__main__':main()
