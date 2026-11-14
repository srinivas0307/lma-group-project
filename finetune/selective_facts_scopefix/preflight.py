"""CPU-only check of the corrected facts before model downloads."""
import json
from pathlib import Path
import pandas as pd
from .context import messages,selective_facts,FactUnavailable
from finetune.selective_facts.context import messages as old_messages,FactUnavailable as OldUnavailable
from finetune.selective_facts.evaluate import development_tasks
from finetune.selective_facts.provenance import write

def check():
    changed=[];unchanged=[]
    for task,_,root in development_tasks():
        before=pd.read_csv(root/task['data_file']);new=messages(task,before)
        try:old=old_messages(task,before)
        except OldUnavailable:
            facts=selective_facts(task['request'],before)
            if len(facts)!=1:raise ValueError('Blocked task still lacks its fact: '+task['id'])
            changed.append(task['id'])
        else:
            if new!=old:raise ValueError('Unexpected change to previously working prompt: '+task['id'])
            unchanged.append(task['id'])
    if len(changed)!=12 or len(unchanged)!=88:raise ValueError('Unexpected scope-fix coverage')
    return dict(repaired=changed,unchanged=unchanged,retrained=False)

if __name__=='__main__':
    result=check();print(json.dumps(result,indent=2))
