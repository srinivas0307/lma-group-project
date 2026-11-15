"""Fresh, paired repair-only development experiment; no old outputs or adapters required."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from igt.llm import LLMClient, ScriptedBackend
from igt.orchestrator import Limits
from igt.agents.repair_agent import AgentConfig
from igt.sandbox import safe_run
from igt.sandbox.environment import sandbox_environment
from igt.verifier import verify
from eval.equivalence import outputs_equal
from eval.stage2_dual_3b.artifacts import write, export, digest
from eval.stage2_dual_3b.config import MODELS
from .agent import run_repair, ARMS, VERSION
from .artifacts import runtime_hashes, verify_pack
from .guard import guarded_verifier
from .suite import build_suite, frame, VERSION as SUITE_VERSION

EXPERIMENT='stage2-3b-repair-dev-v1'


def check_case(case):
    before=frame(case)
    ref=safe_run(case['reference_code'],before,**Limits().sandbox())
    initial=safe_run(case['initial_code'],before,**Limits().sandbox())
    if not ref.ran_ok:
        raise ValueError(case['id']+': invalid reference: '+str(ref.error))
    vf=lambda after:verify(case['contract'],before,after)[0]
    if guarded_verifier(vf,case['contract'],before)(ref.df_out).status!='PASS':
        raise ValueError(case['id']+': reference fails contract/guard')
    correct=initial.ran_ok and outputs_equal(ref.df_out,initial.df_out)[0]
    if bool(correct)!=case['initial_expected_correct']:
        raise ValueError(case['id']+': unexpected initial correctness')
    return before,ref.df_out,initial,bool(correct)


def summarize(records,arms):
    summaries={}
    for arm in arms:
        rows=[r for r in records if r['arm']==arm]
        summaries[arm]=dict(tasks=len(rows),correct=sum(r['final_correct'] for r in rows),
            incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in rows),
            withheld=sum(r['outcome']!='ACCEPT' for r in rows),
            initial_correct_lost=sum(r['initial_correct'] and not r['final_correct'] for r in rows),
            repairs=sum(not r['initial_correct'] and r['final_correct'] for r in rows),
            tokens=sum(r['tokens'] for r in rows),llm_calls=sum(r['llm_calls'] for r in rows),
            tools=sum(r['tool_calls'] for r in rows),fresh_recovery_requests=sum(r['fresh_recovery_requests'] for r in rows),
            diagnostic_seconds=sum(r['diagnostic_seconds'] for r in rows),
            repair_seconds=sum(r['repair_seconds'] for r in rows))
    return summaries


def run(out,arms=tuple(ARMS),smoke=False,cpu_check=False,packages=None,backend_factory=None):
    if not arms or len(set(arms))!=len(arms) or any(a not in ARMS for a in arms):
        raise ValueError('Provide distinct known arms')
    manifest_sha=verify_pack()
    out=Path(out)
    if out.exists():
        raise ValueError('Use a fresh output folder; completed and partial results are not mixed')
    out.mkdir(parents=True)
    cases=build_suite()
    if smoke:
        cases=[c for c in cases if c['id'] in ['category_assignment_1','filter_extra_helper']]
    records=[]; backend=None
    plan=dict(experiment=EXPERIMENT,repair_version=VERSION,suite=SUITE_VERSION,
              suite_sha256=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
              task_ids=[c['id'] for c in cases],arms=list(arms),smoke=smoke,
              model_quality_result=not cpu_check and backend_factory is None,
              protocol='paired synthetic repair-only; fixed initial programs and fixture contracts; not end-to-end Stage 2',
              model=MODELS['agent'],limits=Limits().to_dict(),
              source_manifest_sha256=manifest_sha,runtime=runtime_hashes(),
              environment=dict(python=platform.python_version()),
              limitations=['Small synthetic development suite informed by prior failures; not an independent test.',
                           'No live intent parsing or initial code generation; does not measure contract improvements.',
                           'Evidence arm includes operator semantics, compact history and one bounded recovery; not isolated contributions.'])
    write(out/'plan.json',plan)
    try:
        with sandbox_environment(packages) as env:
            plan['sandbox']=env
            # Fail before downloading a model if a reference or fixture is invalid.
            validated={c['id']:check_case(c) for c in cases}
            if not cpu_check and backend_factory is None:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA GPU required; CPU check is infrastructure-only')
                if torch.cuda.get_device_properties(0).total_memory/2**30<12:
                    raise RuntimeError('Use a GPU with at least 12 GiB VRAM')
                from finetune.selective_facts.attention import configure_attention
                events=[]
                configure_attention(torch,events.append)
                write(out/'attention.json',events)
                from eval.stage2_dual_3b.backend import Backend
                backend=Backend('agent','selective')
                plan['resolved_model']=dict(model=backend.model,revision=backend.revision,device=backend.device,precision=backend.precision)
                plan['environment'].update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                    versions={n:importlib.metadata.version(n) for n in ['transformers','pandas','numpy','accelerate']})
            write(out/'plan.json',plan)
            for index,case in enumerate(cases):
                before,reference,initial,initial_correct=validated[case['id']]
                # Rotate arm order to reduce a fixed order/timing bias, with fresh clients/no response cache.
                ordered=list(arms[index%len(arms):])+list(arms[:index%len(arms)])
                for arm in ordered:
                    if cpu_check:
                        # Fixture replies exercise plumbing only. Reference code never enters live prompts.
                        local_backend=ScriptedBackend(['{"action":"propose_fix"}\n```python\n'+case['reference_code']+'```']*3)
                    else:
                        local_backend=backend_factory(case,arm) if backend_factory else backend
                    client=LLMClient(local_backend)
                    vf=lambda after:verify(case['contract'],before,after)[0]
                    start=time.monotonic()
                    result=run_repair(dict(request=case['request']),case['initial_code'],before,vf,client,
                                      AgentConfig(limits=Limits()),contract_text=json.dumps(case['contract']),
                                      schema_text=json.dumps(dict(columns={c:str(before[c].dtype) for c in before},rows=len(before))),arm=arm)
                    correct,reason=outputs_equal(reference,result.final_output) if result.outcome=='ACCEPT' else (False,'withheld')
                    row=dict(task_id=case['id'],arm=arm,outcome=result.outcome,initial_correct=initial_correct,
                             final_correct=bool(correct),equivalence=reason,attempts=result.attempts,
                             tokens=result.tokens,llm_calls=result.llm_calls,tool_calls=result.tool_calls,
                             fresh_recovery_requests=sum(e['event']=='fresh_recovery_requested' for e in result.trace),
                             diagnostic_seconds=sum(e.get('seconds',0) for e in result.trace if e['event']=='repair_evidence'),
                             repair_seconds=time.monotonic()-start,escalate_reason=result.escalate_reason)
                    records.append(row)
                    write(out/'tasks'/f'{case["id"]}__{arm}.json',dict(record=row,trace=result.trace,initial_code=case['initial_code'],final_code=result.final_code,
                        request=case['request'],contract=case['contract']))
                    write(out/'summary.json',dict(model_quality_result=plan['model_quality_result'],complete=False,arms=summarize(records,arms)))
                    export(out)
                    print(f'{len(records)}/{len(cases)*len(arms)} {case["id"]} {arm}: {result.outcome}; correct={correct}',flush=True)
            summary=dict(model_quality_result=plan['model_quality_result'],complete=True,arms=summarize(records,arms))
            write(out/'summary.json',summary)
            lines=['# 3B repair development comparison', '', 'Repair-only synthetic development; not full Stage 2 or an independent benchmark.',
                   'Model quality run: '+str(plan['model_quality_result']), '']
            for arm,metrics in summary['arms'].items():
                lines.append(f"- {arm}: {metrics['correct']} correct, {metrics['incorrect_accepted']} incorrect accepted, {metrics['withheld']} withheld; {metrics['tokens']} tokens.")
            (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
            write(out/'completion.json',dict(complete=True,records=len(records),expected=len(cases)*len(arms),model_quality_result=plan['model_quality_result']))
            export(out,complete=True)
            return summary
    except BaseException:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),records=len(records)))
        export(out,complete=False)
        raise
    finally:
        if backend is not None:
            backend.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True);p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    p.add_argument('--smoke',action='store_true');p.add_argument('--cpu-check',action='store_true');p.add_argument('--packages')
    a=p.parse_args();print(json.dumps(run(a.out,tuple(a.arms),a.smoke,a.cpu_check,a.packages),indent=2))
if __name__=='__main__':main()
