import time
from igt.llm import LLMClient
from igt.agents.intent_parser import build_messages
from finetune.selective_facts_scopefix.context import messages,selective_facts,schema_entry,FactUnavailable
from finetune.selective_facts_correction.checks import normalize
from .config import PROTOCOLS
from .backend import ContextLimitError
from .tool_parser import parse_tools

def parse(method,task,before,backend,prompted=False):
    public={k:task[k] for k in ['id','request','data_file']}
    if method=='tools' and not prompted:return parse_tools(public,before,backend,'p1-v2')
    start=time.monotonic();facts=[]
    try:
        if prompted:
            msg=build_messages(dict(request=task['request'],data_file=task['data_file']),{task['data_file']:schema_entry(before)},[],version='p1-v3')
        else:msg=messages(public,before);facts=selective_facts(task['request'],before)
    except FactUnavailable as exc:
        return dict(task_id=task['id'],messages=[],raw_output='',extracted=dict(status='input_facts_unavailable',contract=None,error=str(exc)),facts=[],normalizations=[],tokens=0,llm_calls=0,seconds=time.monotonic()-start,model=backend.model,parser_protocol=PROTOCOLS[method])
    client=LLMClient(backend)
    try:
        response=client.generate(msg,max_new_tokens=1024,temperature=0,seed=0)
    except ContextLimitError as exc:
        return dict(task_id=task['id'],messages=msg,raw_output='',extracted=dict(status='parser_context_limit',contract=None,error=str(exc)),facts=facts,normalizations=[],tokens=client.tokens_used,llm_calls=client.n_calls,seconds=time.monotonic()-start,model=backend.model,parser_protocol=PROTOCOLS[method])
    normalized=normalize(response.text)
    return dict(task_id=task['id'],messages=msg,raw_output=response.text,normalized_output=normalized['text'],extracted=normalized['extracted'],facts=facts,
         normalizations=[dict(reply=1,changes=normalized['changes'])] if normalized['changes'] else [],tokens=client.tokens_used,llm_calls=client.n_calls,
         seconds=time.monotonic()-start,model=backend.model,parser_protocol='p1-v3+norm1' if prompted else PROTOCOLS[method])
