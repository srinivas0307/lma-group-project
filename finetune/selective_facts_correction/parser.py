"""At most two model replies: original generation and one diagnostic-guided correction."""
import json
import time
from finetune.selective_facts_scopefix.context import messages,selective_facts,FactUnavailable
from igt.agents.intent_parser import extract_contract
from .checks import normalize,diagnose
PROTOCOL='p1-v2+sf1-scopefix1-norm1-correct1'
ARMS=('control','normalized','corrected')

def attempt_correction(task,before,backend):
    # Restrict the public input at the boundary; evaluation fields never reach generation/checks.
    public=dict(request=task['request'],data_file=task['data_file'])
    start=time.monotonic();trace=dict(protocol=PROTOCOL,attempted=False,llm_calls=0,prompt_tokens=0,output_tokens=0,replies=[])
    try:initial=messages(public,before);facts=selective_facts(public['request'],before)
    except FactUnavailable as exc:
        stopped=dict(status='fact_unavailable',contract=None,error=str(exc))
        return dict(variants={a:dict(stopped) for a in ARMS},facts=[],trace=dict(trace,error=str(exc),seconds=time.monotonic()-start))
    def generate(msg):
        text,ni,no=backend.complete(msg,1024,0,1,0)
        trace['llm_calls']+=1;trace['prompt_tokens']+=ni;trace['output_tokens']+=no
        trace['replies'].append(dict(messages=msg,raw_output=text,prompt_tokens=ni,output_tokens=no))
        return text
    text=generate(initial);control=extract_contract(text);norm=normalize(text)
    issues=diagnose(norm['extracted'],before)
    trace.update(initial_normalization=norm['changes'],initial_issues=issues)
    variants=dict(control=control,normalized=norm['extracted'],corrected=norm['extracted'])
    if issues:
        feedback='''Review your previous contract once. Return a complete corrected contract in one json code fence, following the same vocabulary and original request.
The checks below use only the input schema/data and your own contract; they are not reference answers.
Fix the reported issues while preserving every requirement in the request. Do not loosen constraints merely to silence feedback. Keep exact input column names and full-precision supplied facts. A correction is not guaranteed to be accepted.
Input-consistency feedback:
'''+json.dumps(issues[:8],ensure_ascii=False)+'\nThe original request remains: '+str(public['request'])
        conversation=initial+[dict(role='assistant',content=norm['text']),dict(role='user',content=feedback)]
        can_fit=True
        if hasattr(backend,'check_prompt'):
            try:backend.check_prompt(conversation,1024)
            except ValueError as exc:
                can_fit=False;trace['correction_stop']=dict(reason='context_limit',detail=str(exc))
        if can_fit:
            trace['attempted']=True;revised=generate(conversation);revised_norm=normalize(revised)
            remaining=diagnose(revised_norm['extracted'],before)
            trace.update(correction_normalization=revised_norm['changes'],remaining_issues=remaining)
            if remaining:
                # No score-based fallback to a previous response, and no third model call.
                variants['corrected']=dict(status='correction_failed',contract=None,error=json.dumps(remaining,ensure_ascii=False))
                trace['rejected_candidate']=revised_norm['extracted']
            else:variants['corrected']=revised_norm['extracted']
        else:variants['corrected']=dict(status='correction_context_limit',contract=None,error='One correction could not fit the declared context limit')
    trace['seconds']=time.monotonic()-start
    return dict(variants=variants,facts=facts,trace=trace)
