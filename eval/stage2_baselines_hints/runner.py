"""Entry point: runs the FROZEN contract-check runner (the one behind the repair-hints proposed runs) with this
experiment's execute step and labels; everything is restored on exit.

  --run baseline2              conventional checks, no parser/contract
  --run baseline3              prompted base 3B parser (no adapter) + contract check
  --run no_contract_selective  fine-tuned selective-facts parser (adapter) + contract check
  --run no_contract_tools      fine-tuned tool-use parser (adapter) + contract check

Parser loading, correction turn, generation, models, budgets, sandbox and scoring are the frozen code.
"""
import argparse
import contextlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from eval.stage2_contract_check import runner as check_runner
from .artifacts import verify_pack
from .config import EXPERIMENT,RUNS,SYSTEMS,REPAIR_VERSIONS,CONVENTIONAL_THRESHOLDS,BASELINES
from .execute import execute


@contextlib.contextmanager
def installed(name):
    spec=RUNS[name];sha=verify_pack()
    values=dict(EXPERIMENT=EXPERIMENT,SYSTEMS=SYSTEMS,execute=execute,AGENT_PROMPT_VERSION=REPAIR_VERSIONS[spec['preset']],
                verify_pack=lambda:sha,
                COMMON=dict(check_runner.COMMON,baselines_run=name,baselines=BASELINES,baselines_source_manifest_sha256=sha))
    if spec['preset']=='baseline2_repair':values['THRESHOLDS']=CONVENTIONAL_THRESHOLDS
    saved={k:getattr(check_runner,k) for k in values}
    try:
        for k,v in values.items():setattr(check_runner,k,v)
        yield sha
    finally:
        for k,v in saved.items():setattr(check_runner,k,v)


def run(name,out,adapter=None,split='test',smoke=False,packages=None,backend_factory=None):
    if name not in RUNS:raise ValueError('Unknown run: '+name)
    spec=RUNS[name]
    if not spec['adapter'] and adapter is not None:raise ValueError(name+' does not use a parser adapter')
    with installed(name):
        return check_runner.run(spec['method'],out,adapter,spec['preset'],split,smoke,packages,backend_factory,arm=spec['arm'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',choices=list(RUNS),required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--split',choices=['dev','test'],default='test');p.add_argument('--smoke',action='store_true');p.add_argument('--packages',type=Path)
    a=p.parse_args();run(a.run,a.out,a.adapter,a.split,a.smoke,a.packages)
if __name__=='__main__':main()
