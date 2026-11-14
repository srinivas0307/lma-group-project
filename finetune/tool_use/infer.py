"""Run either arm on a CSV and request; no evaluation/reference files needed."""
import argparse
from pathlib import Path
import pandas as pd
from .config import MODELS
from .backend import Backend
from .parser import parse_fresh
from finetune.selective_facts.provenance import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--request',required=True)
    p.add_argument('--size',choices=MODELS,default='1_5b');p.add_argument('--adapter',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Output already exists')
    import torch
    from finetune.selective_facts.attention import configure_attention
    configure_attention(torch,lambda event:None)
    backend=Backend(a.size,a.adapter)
    try:write(a.out,parse_fresh(dict(id='input',data_file=a.csv.name,request=a.request),pd.read_csv(a.csv),backend,'p1-v2'))
    finally:backend.close()
if __name__=='__main__':main()
