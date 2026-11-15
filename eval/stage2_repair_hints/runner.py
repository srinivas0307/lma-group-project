"""repair-hints-v1 entry point. Runs the FROZEN runners unchanged and swaps only the repair function:

  --pipeline gold       eval.stage2_combined_gold.runner (gold contracts; oracle diagnostic vs 84/0/16)
  --pipeline selective  eval.stage2_contract_check.runner --arm checked (vs 76/1/23)
  --pipeline tools      eval.stage2_contract_check.runner --arm checked (vs 74/1/25)

Parser, correction turn, generator, models, prompts, budgets, schema guard and verifier are identical.
"""
import argparse
import contextlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from eval.stage2_combined_3b import execute as proposed_execute
from eval.stage2_combined_gold import execute as gold_execute
from eval.stage2_contract_check import runner as check_runner
from eval.stage2_combined_gold import runner as gold_runner
from . import agent
from .artifacts import verify_pack
from .config import EXPERIMENT,HINTS,REPAIR_VERSION,PIPELINES


@contextlib.contextmanager
def installed():
    """Swap the repair function and run labels in the frozen modules; restored on exit."""
    sha=verify_pack()
    patches=[(m,dict(run_repair=agent.run_repair,REPAIR_VERSION=REPAIR_VERSION)) for m in (proposed_execute,gold_execute)]
    patches+=[(m,dict(EXPERIMENT=EXPERIMENT,AGENT_PROMPT_VERSION=REPAIR_VERSION,verify_pack=lambda:sha,
                      COMMON=dict(m.COMMON,**HINTS,repair_hints_source_manifest_sha256=sha))) for m in (check_runner,gold_runner)]
    saved=[(m,{k:getattr(m,k) for k in values}) for m,values in patches]
    try:
        for m,values in patches:
            for k,v in values.items():setattr(m,k,v)
        yield sha
    finally:
        for m,values in saved:
            for k,v in values.items():setattr(m,k,v)


def run(pipeline,out,adapter=None,split='test',smoke=False,packages=None,backend_factory=None):
    if pipeline not in PIPELINES:raise ValueError('Unknown pipeline')
    with installed():
        if pipeline=='gold':
            return gold_runner.run('gold',out,None,'gold_contract',split,smoke,packages,backend_factory)
        return check_runner.run(pipeline,out,adapter,'proposed',split,smoke,packages,backend_factory,arm='checked')


def main():
    p=argparse.ArgumentParser();p.add_argument('--pipeline',choices=PIPELINES,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--split',choices=['dev','test'],default='test');p.add_argument('--smoke',action='store_true');p.add_argument('--packages',type=Path)
    a=p.parse_args();run(a.pipeline,a.out,a.adapter,a.split,a.smoke,a.packages)
if __name__=='__main__':main()
