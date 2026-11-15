"""CPU-only routing/normalization checks; actual model quality is measured in the run."""
import json
import pandas as pd
from igt.agents.intent_parser import contract_to_text
from .parsers import parse
from .backend import ContextLimitError
from finetune.selective_facts_scopefix.preflight import check as check_scope

class Scripted:
    model='scripted-preflight'
    def __init__(self,replies):self.replies=list(replies);self.calls=[]
    def complete(self,messages,*args,**kwargs):
        self.calls.append(messages)
        if not self.replies:raise AssertionError('Unexpected extra model reply')
        return self.replies.pop(0),100,50

def check():
    frame=pd.DataFrame({'x':[1.,3.,None]})
    task=dict(id='probe',request='Fill missing x with its full-column median. Ignore missing values when computing statistics.',data_file='input.csv')
    contract=dict(contract_version='1.0',must_change=[dict(type='column_value_relation',target='x',fn='const',args=[{'lit':2}],where=dict(column='x',op=' is_null '))],must_preserve=[dict(type='row_count',op='unchanged')])
    text=contract_to_text(contract)
    for method in ['selective','tools']:
        backend=Scripted([text] if method=='selective' else ['{"tool":"profile_column","args":{"column":"x"}}',text])
        r=parse(method,task,frame,backend)
        if r['extracted']['status']!='ok' or not r['normalizations']:raise ValueError('Parser normalization preflight failed')
        if method=='tools' and r['tool_calls'][0]['observation']['result']['statistics']['median']!=2:raise ValueError('Wrong tool observation')
    scope=check_scope()
    return dict(parsers='passed',normalization='passed',scope_repaired=len(scope['repaired']),scope_unchanged=len(scope['unchanged']))
if __name__=='__main__':print(json.dumps(check(),indent=2))
