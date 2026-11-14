"""Convert checked training examples to real tool conversations, never model outputs."""
import argparse
import collections
import json
from pathlib import Path
import pandas as pd
from eval.stage2_tools.data_tools import DataTools
from .context import parser_messages,observation_message
from .config import PROTOCOL
from finetune.selective_facts.provenance import ROOT,digest,write,verify_data

def convert(row,df,recovery=False):
    request=row['messages'][1]['content'].rsplit('Request: ',1)[1]
    task=dict(id=row['id'],request=request,data_file=row['data_file'])
    messages=parser_messages(task,df)
    tools=DataTools(df)
    for fact in row['facts']:
        args=dict(column=fact['column'])
        if fact['scope']['kind']=='conditional':args['where']=fact['scope']['where']
        if recovery:
            bad=dict(column='__unknown_column__')
            assert bad['column'] not in df.columns
            messages += [dict(role='assistant',content=json.dumps(dict(tool='profile_column',args=bad))),observation_message(tools.call('profile_column',bad))]
        observation=tools.call('profile_column',args)
        if not observation['ok']:raise ValueError(observation)
        # The original target fact must actually be present in the returned tool data.
        payload=observation['result'];stat=fact['statistic']
        values=payload['modes'] if stat=='mode' else [payload['statistics'][stat]]
        if fact['value'] not in values:raise ValueError('Target literal is absent from tool observation')
        messages += [dict(role='assistant',content=json.dumps(dict(tool='profile_column',args=args))),observation_message(observation)]
    messages.append(row['messages'][-1])
    return dict(id=row['id']+('_recovery' if recovery else ''),source_dataset=row['source_dataset'],messages=messages,requires_tool=bool(row['facts']),recovery=recovery)

def build(source,out,repeat=4):
    source=Path(source);out=Path(out)
    if repeat<1:raise ValueError('tool-repeat must be positive')
    if out.exists():raise ValueError('Use a fresh data directory')
    verify_data(source);out.mkdir(parents=True)
    cache={};counts={}
    for split in ['train','val']:
        rows=[]
        for line in (source/f'sft_{split}.jsonl').read_text().splitlines():
            original=json.loads(line);name=original['data_file']
            if name not in cache:cache[name]=pd.read_csv(ROOT/name)
            row=convert(original,cache[name]);rows.append(row)
            if row['requires_tool']:
                # Repeat only training tool conversations; validation distribution stays unchanged.
                if split=='train':
                    rows.extend(dict(row,id=row['id']+f'_repeat{i}') for i in range(1,repeat))
                    rows.append(convert(original,cache[name],True))
        # Explicit tool-observation -> escalation examples on disjoint source schemas.
        domains={r['source_dataset'] for r in rows if r['source_dataset'].startswith('sf_')}
        for domain in sorted(domains):
            original_df=next(df for name,df in cache.items() if Path(name).stem==domain)
            for kind in ['tied_mode','all_missing']:
                df=original_df.copy();column=domain+('_status' if kind=='tied_mode' else '_amount')
                if kind=='tied_mode':
                    df[column]=['left','right',None,None]*(len(df)//4)
                    request=f'Fill missing {column} with its unique most frequent nonmissing value.'
                    reason='The mode is tied and the request gives no tie-breaking rule.'
                else:
                    df[column]=float('nan')
                    request=f'Fill missing {column} with its full-column median.'
                    reason='There are no nonmissing input values from which to compute the median.'
                task=dict(id=domain+'_'+kind,data_file=domain+'.csv',request=request)
                args=dict(column=column);observation=DataTools(df).call('profile_column',args)
                if not observation['ok']:raise ValueError(observation)
                messages=parser_messages(task,df)+[dict(role='assistant',content=json.dumps(dict(tool='profile_column',args=args))),observation_message(observation),dict(role='assistant',content=json.dumps(dict(action='escalate',reason=reason)))]
                rows.append(dict(id=task['id'],source_dataset=domain,messages=messages,requires_tool=True,recovery=False,escalation=True))
        (out/f'sft_{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
        counts[split]=dict(examples=len(rows),tool_conversations=sum(r['requires_tool'] for r in rows),recovery=sum(r['recovery'] for r in rows),escalation=sum(r.get('escalation',False) for r in rows))
    write(out/'manifest.json',dict(protocol=PROTOCOL,source_manifest=digest(source/'manifest.json'),tool_repeat=repeat,counts=counts,
          sha256={f'sft_{s}.jsonl':digest(out/f'sft_{s}.jsonl') for s in ['train','val']},
          limitations=['Automatically constructed conversations; not independently reviewed.','Tool supervision covers profile_column and single equality scopes; other exposed tools are prompt-only.','Training and validation retain the source dataset split.']))
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'finetune/data/selective_facts_v1');p.add_argument('--out',type=Path,required=True);p.add_argument('--tool-repeat',type=int,default=4)
    a=p.parse_args();build(a.source,a.out,a.tool_repeat)
