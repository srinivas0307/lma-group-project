"""Bounded model -> read-only tool -> observation -> model loop, separate from profile v1."""
import json
import re
import time
from igt.llm import LLMClient
from igt.agents.intent_parser import extract_contract
from .context import parser_messages
from eval.stage2_tools.data_tools import DataTools


def action_object(text):
    text=text.strip()
    if text.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n?```',text,re.S)
        if not match:return None
        text=match[1]
    try:return json.loads(text)
    except json.JSONDecodeError:return None


def parse_fresh(task,before,backend,prompt,max_tools=6,max_calls=10,max_invalid=2,token_budget=24000):
    from .backend import ContextLimitError
    messages=parser_messages(task,before,prompt);tools=DataTools(before);client=LLMClient(backend)
    trace=[];invalid=0;start=time.monotonic();raw=''
    def finish(status,reason=None,contract=None):
        return dict(task_id=task['id'],messages=messages,raw_output=raw,extracted=dict(status=status,contract=contract,error=reason),
            tokens=client.tokens_used,llm_calls=client.n_calls,seconds=time.monotonic()-start,model=backend.model,
            tool_calls=trace,tool_attempts=len(trace),tool_failures=sum(not t['observation']['ok'] for t in trace),
            invalid_responses=invalid,stop_reason=reason,parser_protocol='parser-tools-v2')
    for _ in range(max_calls):
        try:
            # Real backends expose exact tokenizer counts. Scripted CPU tests use a conservative estimate.
            if hasattr(backend,'prompt_tokens'):count=backend.prompt_tokens(messages)
            else:count=sum(len(m['content']) for m in messages)//4
            if client.tokens_used+count+1024>token_budget:return finish('parser_token_limit','Parser token budget cannot accommodate another reply')
            response=client.generate(messages,max_new_tokens=1024,temperature=0,seed=0)
        except ContextLimitError as error:return finish('parser_context_limit',str(error))
        raw=response.text;messages.append(dict(role='assistant',content=raw))
        obj=action_object(raw)
        if isinstance(obj,dict) and set(obj)=={'action','reason'} and obj.get('action')=='escalate' and isinstance(obj['reason'],str) and obj['reason'].strip():
            return finish('parser_escalation',obj['reason'][:500])
        if isinstance(obj,dict) and set(obj)=={'tool','args'} and isinstance(obj['tool'],str):
            if len(trace)>=max_tools:return finish('parser_tool_limit','Tool limit reached; parser requested another tool instead of finishing')
            observation=tools.call(obj['tool'],obj['args'])
            trace.append(dict(tool=obj['tool'],args=obj['args'],observation=observation))
            instruction='No tool calls remain. Return the final contract or escalate.' if len(trace)==max_tools else 'Choose another tool, return the final contract, or escalate.'
            messages.append(dict(role='user',content='Read-only input-data tool result:\n'+json.dumps(observation,ensure_ascii=False,allow_nan=False)+'\n'+instruction))
            continue
        parsed=extract_contract(raw)
        if parsed['status']=='ok':return finish('ok',contract=parsed['contract'])
        invalid+=1
        if invalid>max_invalid:return finish('parser_invalid_response',parsed['error'])
        messages.append(dict(role='user',content='Invalid response. Send one tool-call JSON object, a schema-valid final contract in a json fence, or {"action":"escalate","reason":"..."}. Parser error: '+str(parsed['error'])))
    return finish('parser_call_limit','Maximum model replies reached without a valid final contract')
