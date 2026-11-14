import json
from pathlib import Path
import pandas as pd
import pytest
from igt.llm import ScriptedBackend
from igt.agents.intent_parser import contract_to_text
from finetune.tool_use.parser import parse_fresh
from finetune.tool_use.context import parser_messages
from finetune.tool_use.tokenization import tokenize
from finetune.tool_use.build_data import convert
from finetune.tool_use.adapter import validate
from finetune.tool_use.config import MODELS,PROTOCOL
from finetune.selective_facts.provenance import digest

TASK=dict(id='x',request='Fill missing score with the full-column median.',data_file='x.csv')
GOLD=dict(contract_version='1.0',must_change=[],must_preserve=[])
DF=pd.DataFrame(dict(score=[1.123456789,None,5.987654321],region=['a','a','b']))

class Tokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=''.join('<'+m['role']+'>'+m['content']+'</end>' for m in messages)
        return text+('<assistant>' if add_generation_prompt else '')
    def __call__(self,text,add_special_tokens=False):return {'input_ids':list(map(ord,text))}

def test_supervision_includes_tool_action_and_final_but_not_observation():
    messages=[dict(role='system',content='RULES'),dict(role='user',content='REQUEST'),dict(role='assistant',content='TOOLCALL'),dict(role='user',content='OBSERVATION'),dict(role='assistant',content='FINAL')]
    row=tokenize([dict(id='x',messages=messages)],Tokenizer(),1000)[0]
    supervised=''.join(chr(x) for x in row['labels'] if x!=-100)
    assert supervised=='TOOLCALL</end>FINAL</end>'
    assert 'OBSERVATION' not in supervised and 'RULES' not in supervised
    with pytest.raises(ValueError,match='no truncation'):tokenize([dict(id='x',messages=messages)],Tokenizer(),2)

def test_reject_chat_template_boundary_drift():
    class Broken(Tokenizer):
        def apply_chat_template(self,messages,**kw):
            text=super().apply_chat_template(messages,**kw)
            return text+('broken' if kw.get('add_generation_prompt') else '')
    with pytest.raises(ValueError,match='boundary'):tokenize([dict(id='x',messages=[dict(role='user',content='a'),dict(role='assistant',content='b')])],Broken(),1000)

def test_model_selected_tool_and_exact_observation():
    def final(messages):
        result=json.loads(messages[-1]['content'].splitlines()[1])
        assert result['result']['statistics']['median']==DF.score.median()
        return contract_to_text(GOLD)
    record=parse_fresh(TASK,DF,ScriptedBackend(['{"tool":"profile_column","args":{"column":"score"}}',final]),'p1-v2')
    assert record['tool_attempts']==1 and record['llm_calls']==2 and record['extracted']['status']=='ok'
    assert record['parser_protocol']==PROTOCOL

def test_no_forced_tools_or_gold_in_prompt():
    assert parser_messages(TASK,DF)==parser_messages(dict(TASK,gold_contract='SECRET',reference_code='SECRET'),DF)
    record=parse_fresh(TASK,DF,ScriptedBackend([contract_to_text(GOLD)]),'p1-v2')
    assert record['tool_attempts']==0
    # Full-data profile appears only after an actual query.
    assert 'statistics' not in record['messages'][1]['content']

def test_recovery_and_context_failure():
    b=ScriptedBackend(['{"tool":"profile_column","args":{"column":"missing"}}','{"tool":"profile_column","args":{"column":"score"}}',contract_to_text(GOLD)])
    r=parse_fresh(TASK,DF,b,'p1-v2');assert r['tool_failures']==1 and r['tool_attempts']==2
    from finetune.tool_use.backend import ContextLimitError
    def too_long(messages):raise ContextLimitError('too long')
    b=ScriptedBackend([]);b.prompt_tokens=too_long
    assert parse_fresh(TASK,DF,b,'p1-v2')['extracted']['status']=='parser_context_limit'

def test_conversion_actual_conditional_observations_and_no_fact_leak():
    row=dict(id='x',data_file='x.csv',source_dataset='test',facts=[dict(column='score',statistic='median',value=1.123456789,scope=dict(kind='conditional',where=dict(column='region',op='==',value='a')))],messages=[{},dict(content='Request: '+TASK['request']),dict(role='assistant',content=contract_to_text(GOLD))])
    result=convert(row,DF,recovery=True)
    assert [m['role'] for m in result['messages']]==['system','user','assistant','user','assistant','user','assistant']
    observation=json.loads(result['messages'][-2]['content'].splitlines()[1])
    assert observation['result']['statistics']['median']==1.123456789
    assert 'Required input facts' not in result['messages'][1]['content']

@pytest.mark.parametrize('size',['1_5b','3b'])
def test_adapter_size_protocol_and_weight_seals(tmp_path,size):
    for name in ['adapter_config.json','adapter_model.safetensors']:(tmp_path/name).write_text('dummy')
    meta=dict(protocol=PROTOCOL,size=size,base=MODELS[size],smoke=False,sha256={p.name:digest(p) for p in tmp_path.iterdir()})
    (tmp_path/'tool_protocol.json').write_text(json.dumps(meta))
    assert validate(tmp_path,size)['size']==size
    with pytest.raises(ValueError,match='size'):validate(tmp_path,'3b' if size=='1_5b' else '1_5b')
    (tmp_path/'adapter_model.safetensors').write_text('changed')
    with pytest.raises(ValueError,match='hash'):validate(tmp_path,size)

def test_training_split_and_repeat_provenance():
    root=Path(__file__).resolve().parents[1]/'finetune/data/tool_use_v2'
    manifest=json.loads((root/'manifest.json').read_text());groups={}
    for split in ['train','val']:
        path=root/f'sft_{split}.jsonl';assert digest(path)==manifest['sha256'][path.name]
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        assert len(rows)==manifest['counts'][split]['examples']
        assert len({r['id'] for r in rows})==len(rows)
        groups[split]={r['source_dataset'] for r in rows}
    assert not groups['train']&groups['val']
