"""CPU protocol checks before GPU inference; no training or saved model outputs."""
import json
import pandas as pd
from igt.agents.intent_parser import contract_to_text
from .checks import normalize,diagnose
from .parser import attempt_correction
from finetune.selective_facts_scopefix.preflight import check as scope_check

def check():
    df=pd.DataFrame({'amount':[1.,2.,None]})
    valid=dict(contract_version='1.0',must_change=[dict(type='column_value_relation',target='amount',fn='add',args=[{'col':'amount'},{'lit':2}])],must_preserve=[dict(type='row_count',op='unchanged')])
    invalid=json.loads(json.dumps(valid));invalid['must_change'].append(dict(type='columns_added',target=['amount']))
    if not diagnose(normalize(contract_to_text(invalid))['extracted'],df):raise ValueError('Missing input consistency diagnostic')
    class Scripted:
        def __init__(self):self.calls=0
        def complete(self,*args):
            result=invalid if self.calls==0 else valid;self.calls+=1;return contract_to_text(result),10,10
    backend=Scripted();r=attempt_correction(dict(request='Add 2 to amount.',data_file='input.csv'),df,backend)
    if backend.calls!=2 or r['variants']['corrected']['status']!='ok':raise ValueError('One-correction pipeline failed')
    literal=dict(contract_version='1.0',must_change=[dict(type='row_count',op=' eq ',value=3)],must_preserve=[])
    normalized=normalize(contract_to_text(literal))
    if len(normalized['changes'])!=1 or normalized['extracted']['status']!='ok':raise ValueError('Operator normalization failed')
    scope=scope_check()
    return dict(normalization='passed',one_correction='passed',scope_repaired=len(scope['repaired']),scope_unchanged=len(scope['unchanged']))
if __name__=='__main__':print(json.dumps(check(),indent=2))
