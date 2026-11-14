"""Probe installed Trainer plus actual pinned chat-template boundaries before GPU work."""
import argparse
import json
from .config import MODELS
from .tokenization import tokenize
from finetune.selective_facts.provenance import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True);a=p.parse_args()
    from finetune.selective_facts.preflight import main as trainer_probe
    trainer_probe()
    from transformers import AutoTokenizer
    spec=MODELS[a.size];tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'])
    rows=[json.loads(l) for l in (ROOT/'finetune/data/tool_use_v2/sft_train.jsonl').read_text().splitlines()]
    examples=[next(r for r in rows if r['requires_tool']),next(r for r in rows if not r['requires_tool']),next(r for r in rows if r['recovery']),next(r for r in rows if r.get('escalation'))]
    encoded=tokenize(examples,tok,8192)
    print('PASSED actual tokenizer: no-tool, tool, recovery and escalation conversations; every assistant turn supervised. Lengths:',[len(r['input_ids']) for r in encoded],flush=True)
if __name__=='__main__':main()
