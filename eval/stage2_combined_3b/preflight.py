"""Original parser preflight plus a scripted real combined repair (CPU only)."""
import json
from eval.stage2_dual_3b.preflight import check as parser_check
from eval.stage2_repair_dev.suite import build_suite,frame
from eval.stage2_repair_dev.agent import run_repair
from igt.llm import LLMClient,ScriptedBackend
from igt.verifier import verify
from .artifacts import verify_pack

def check():
    verify_pack();result=parser_check()
    c=next(c for c in build_suite() if c['id']=='filter_extra_helper');before=frame(c)
    client=LLMClient(ScriptedBackend([json.dumps(dict(action='propose_fix',code=c['reference_code']))]))
    r=run_repair(dict(request=c['request']),c['initial_code'],before,lambda after:verify(c['contract'],before,after)[0],client,contract_text=json.dumps(c['contract']),arm='combined')
    if r.outcome!='ACCEPT' or r.attempts!=1 or 'temporary_stamp' in r.final_output:raise ValueError('Combined repair preflight failed')
    return dict(result,combined_repair='passed',model_quality_result=False)
if __name__=='__main__':print(json.dumps(check(),indent=2))
