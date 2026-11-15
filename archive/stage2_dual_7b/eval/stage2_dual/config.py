EXPERIMENT='stage2-dual-parser-v1'
METHODS=('selective','tools')
SYSTEMS=('baseline1','baseline2_repair','baseline3_repair','gold_contract','proposed','no_contract_to_generator')
MODELS={
 'parser':dict(id='Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',four_bit=True),
 'generator':dict(id='Qwen/Qwen2.5-3B-Instruct',revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',four_bit=False),
 'agent':dict(id='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',four_bit=True),
}
PROTOCOLS={'selective':'p1-v2+sf1-scopefix1-norm1','tools':'parser-tools-v2+norm1'}
COMMON=dict(seed=0,generator_max_new_tokens=512,agent_max_new_tokens=768,max_attempts=3,max_tool_calls_per_attempt=8,
 token_budget=60000,context_tokens=32768,loading='sequential-one-model-at-a-time-v1',normalization='recognized-operator-whitespace-only-v1',
 prompted_baseline='p1-v3-zero-shot-schema-and-three-rows',conventional='hand-set-untuned-v1')
PARSER_SETTINGS={
 'selective':dict(max_new_tokens=1024,max_model_calls=1,context_tokens=8192,model_correction=False),
 'tools':dict(max_new_tokens=1024,max_model_calls=10,context_tokens=12288,max_tools=6,max_invalid_corrections=2,token_budget=24000,max_result_chars=6000),
}

from eval.stage2_selective.config import THRESHOLDS
