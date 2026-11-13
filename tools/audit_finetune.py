"""Read-only CPU audit of saved SFT data, adapter tensors, and training provenance.
Run with .venv/bin/python tools/audit_finetune.py; writes only the audit report.
"""
import collections
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
import yaml
from finetune.data_gen.build_sft import to_messages,schema_entry
from finetune.train_qlora import load_rows
from finetune.data_gen.check_examples import ngrams
from igt.contracts import load_contract


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    base=ROOT/'finetune/data/v2'
    train=load_rows(base/'sft_train.jsonl',None)
    val=load_rows(base/'sft_val.jsonl',None)
    selected=load_rows(base/'sft_val.jsonl',200,1234)
    checked={r['id']:r for line in (base/'checked.jsonl').read_text().splitlines() if (r:=json.loads(line))}
    schemas={f:schema_entry(pd.read_csv(ROOT/f)) for f in {r['data_file'] for r in checked.values()}}
    tasks=[yaml.safe_load(p.read_text()) for p in (ROOT/'benchmark/tasks').glob('*/*.yaml')]
    eval_requests={' '.join(t['request'].split()).lower() for t in tasks}
    eval_ngrams=set().union(*(ngrams(t['request']) for t in tasks))
    mismatches=[];exact_leaks=[];ngram_leaks=[]
    for r in train+val:
        raw=checked[r['id']]
        load_contract(raw['contract'])
        if r['messages']!=to_messages(raw,schemas,r['prompt_version']):mismatches.append(r['id'])
        if ' '.join(raw['request'].split()).lower() in eval_requests:exact_leaks.append(r['id'])
        if ngrams(raw['request'])&eval_ngrams:ngram_leaks.append(r['id'])
    by_pattern=lambda rows:dict(sorted(collections.Counter(checked[r['id']]['pattern'] for r in rows).items()))
    report=dict(train=len(train),full_validation=len(val),monitored_validation=len(selected),
                train_datasets=sorted({r['source_dataset'] for r in train}),validation_datasets=sorted({r['source_dataset'] for r in val}),
                train_validation_id_overlap=sorted({r['id'] for r in train}&{r['id'] for r in val}),
                sft_reconstruction_mismatches=mismatches,exact_evaluation_request_overlap=exact_leaks,
                evaluation_five_gram_overlap=ngram_leaks,train_patterns=by_pattern(train),
                validation_patterns=by_pattern(val),monitored_validation_patterns=by_pattern(selected),
                unique_train_dataset_program_contracts=len({(checked[r['id']]['data_file'],checked[r['id']]['reference_code'],json.dumps(checked[r['id']]['contract'],sort_keys=True)) for r in train}),
                runs={},limitations=['Adapter tensor integrity does not prove correctness of all training gradients.',
                                     'Dataset/request separation does not prove independence of semantic templates.',
                                     'No GPU retraining or generation was performed.',
                                     'This checks label serialization and schema, not a fresh human semantic review of all 4,410 examples.'])
    for size in ['1_5b','3b']:
        folder=ROOT/f'finetune/runs/qwen25_{size}'
        info=json.loads((folder/'run_info.json').read_text())
        config=yaml.safe_load((ROOT/f'finetune/configs/qwen25_{size}.yaml').read_text())
        logs=[json.loads(l) for l in (folder/'train_log.jsonl').read_text().splitlines()]
        adapter=folder/'adapter/adapter_model.safetensors'
        with adapter.open('rb') as f:
            header_size=struct.unpack('<Q',f.read(8))[0]
            header=json.loads(f.read(header_size))
        arrays={k:v for k,v in header.items() if k!='__metadata__'}
        nonfinite=[];zero=[];params=0
        for name,spec in arrays.items():
            assert spec['dtype']=='F32',spec['dtype']
            weights=np.memmap(adapter,dtype='<f4',mode='r',offset=8+header_size+spec['data_offsets'][0],shape=tuple(spec['shape']))
            params+=int(np.prod(spec['shape']))
            if not np.isfinite(weights).all():nonfinite.append(name)
            if not np.any(weights):zero.append(name)
        assert params==info['trainable_params']
        evals=[r for r in logs if 'eval_loss' in r]
        effective_samples_per_update=info['n_train']*info['final_eval']['epoch']/info['steps']
        report['runs'][size]=dict(recorded_config=info['config'],adapter_sha256=digest(adapter),tensor_count=len(arrays),
            adapter_parameters=params,nonfinite_tensors=nonfinite,all_zero_tensors=zero,
            corrected_data_hashes_match={k:digest(base/('sft_train.jsonl' if k=='train_file' else 'sft_val.jsonl'))==h for k,h in info['data_sha256'].items()},
            default_data_hashes_match={k:digest(ROOT/info['config'][k])==h for k,h in info['data_sha256'].items()},
            current_config_differences={k:dict(saved_run=v,current_yaml=config[k]) for k,v in info['config'].items() if k in config and config[k]!=v},
            final_eval=info['final_eval'],eval_loss_history=[{k:r[k] for k in ('step','epoch','eval_loss')} for r in evals],
            steps=info['steps'],approx_samples_per_optimizer_step=round(effective_samples_per_update,3),
            training_gpu_topology_recorded=any(k in info for k in ('n_gpu','world_size','effective_batch_size')),
            dropped_too_long=info['dropped_too_long'],max_tokens=info['max_tokens'],prompt_version=(folder/'adapter/igt_prompt_version.txt').read_text().strip())
        assert not nonfinite and not zero
        assert all(report['runs'][size]['corrected_data_hashes_match'].values())
    assert not mismatches and not exact_leaks and not ngram_leaks
    out=ROOT/'results/finetune_audit'
    out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified 4,410 SFT records, both data-hash pairs, 896 adapter tensors, and saved run metadata.')
    print('Monitoring sample:',{p:report['monitored_validation_patterns'].get(p,0) for p in ['fill_median','fill_most_common','to_ordered_category','text_to_datetime']})
    print(out/'audit.json')

if __name__=='__main__':main()
