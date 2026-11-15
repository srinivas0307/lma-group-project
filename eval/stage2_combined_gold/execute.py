"""Public runtime boundary: fresh generation and frozen combined repair, no reference access."""
import json,time
from functools import partial
from igt.llm import LLMClient
from igt.orchestrator import Limits
from igt.sandbox import safe_run
from igt.verifier import verify
from igt.types import Verdict
from igt.agents.code_generator_v2 import build_generator_messages
from igt.agents.code_extract import extract_code
from igt.agents.repair_agent import AgentConfig
from eval.stage2_repair_dev.agent import run_repair as combined_repair
from eval.stage2_selective.runner import PARSED
from finetune.selective_facts_scopefix.context import schema_entry
from .config import REPAIR_VERSION
run_repair=partial(combined_repair,arm='combined')
def execute(task,before,preset,parsed,generator,agent):
    """Runtime boundary receives only the request, data and provided gold contract."""
    if preset!='gold_contract':raise ValueError('Combined workflow currently supports gold_contract only')
    limits=Limits();start=time.monotonic()
    contract=parsed['extracted']['contract'] if parsed and parsed['extracted']['status']=='ok' else None
    status=parsed['extracted']['status'] if parsed else 'not_used'
    parser_tokens=parsed['tokens'] if parsed and preset in PARSED else 0
    parser_calls=parsed['llm_calls'] if parsed and preset in PARSED else 0
    record=dict(task_id=task['id'],system=preset,split=task['split'],dataset=task['dataset'],
                transformation_type=task['transformation_type'],contract_status=status,parser_tokens=parser_tokens,
                parser_calls=parser_calls,parser_seconds=parsed.get('seconds',0) if parsed and preset in PARSED else 0,generation_repair_tokens=0,tokens=parser_tokens,llm_calls=parser_calls,
                initial_correct=None,final_correct=False,repair_version=REPAIR_VERSION,repair_arm='combined',schema_guard='declared-output-columns-v1',fresh_recovery_requests=0,repair_diagnostic_seconds=0)
    if preset in PARSED and contract is None:
        record.update(outcome='ESCALATE',stage_ended='parse',attempts=0,escalate_reason='Invalid live parser contract',runtime_seconds=time.monotonic()-start)
        return dict(record=record,trace=None,initial_output=None,final_output=None)
    checker=lambda after:verify(contract,before,after)[0]
    public=dict(request=task['request'],data_file=task['data_file'])
    schema={task['data_file']:schema_entry(before)}
    gc_client=LLMClient(generator)
    response=gc_client.generate(build_generator_messages(public,schema,contract),max_new_tokens=512,temperature=0,seed=0)
    extracted=extract_code(response.text);code=extracted['code'] if extracted['parse_ok'] else response.text
    ac=LLMClient(agent,token_budget=limits.token_budget)
    result=run_repair(dict(request=task['request']),code,before,checker,ac,
        AgentConfig(mode='full',limits=limits,temperature=0),
        contract_text=json.dumps(contract) if contract else None,schema_text=messages_schema(schema[task['data_file']]))
    first=safe_run(code,before,**limits.sandbox())
    record.update(fresh_recovery_requests=sum(e['event']=='fresh_recovery_requested' for e in result.trace),repair_diagnostic_seconds=sum(e.get('seconds',0) for e in result.trace if e['event']=='repair_evidence'))
    record.update(outcome=result.outcome,stage_ended='repair' if result.attempts else 'check',attempts=result.attempts,
        generation_repair_tokens=gc_client.tokens_used+ac.tokens_used,tokens=parser_tokens+gc_client.tokens_used+ac.tokens_used,
        llm_calls=parser_calls+gc_client.n_calls+ac.n_calls,tool_calls=result.tool_calls,
        unsafe_attempts=result.unsafe_attempts,invalid_actions=result.invalid_actions,
        escalate_reason=result.escalate_reason,runtime_seconds=time.monotonic()-start)
    return dict(record=record,trace=dict(generator_reply=response.text,initial_code=code,final_code=result.final_code,events=result.trace),
                initial_output=first.df_out if first.ran_ok else None,final_output=result.final_output)

def messages_schema(entry):
    return f"{entry['n_rows']} rows. Columns:\n"+'\n'.join(f"{c['name']}: {c['dtype']}, {c['nulls']} missing" for c in entry['columns'])+'\nFirst 3 rows:\n'+entry['head_csv']
