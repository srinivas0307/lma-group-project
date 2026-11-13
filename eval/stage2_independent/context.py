"""Build parser input exclusively from the current request and raw input data."""
import json
import re
from eval.contract_quality.context import profile
from igt.agents.intent_parser import build_messages

def schema_entry(df):
    return dict(n_rows=len(df),columns=[dict(name=c,dtype=str(df[c].dtype),nulls=int(df[c].isna().sum())) for c in df],head_csv=df.head(3).to_csv(index=False))

def parser_messages(task,df,prompt='p1-v2'):
    # Explicitly limit the runtime input to user-visible fields.
    public=dict(request=task['request'],data_file=task['data_file'])
    messages=build_messages(public,{task['data_file']:schema_entry(df)},[],version=prompt)
    data=json.loads(profile(df,max_chars=2_000_000))
    # Keep all full-column statistics. Group profiles are bounded by explicit column mentions.
    groups=data.pop('groups')
    requested={c:values for c,values in groups.items() if re.search(r'(?<!\w)'+re.escape(c)+r'(?!\w)',task['request'],re.I)}
    data['groups']=requested
    omitted=sorted(set(groups)-set(requested))
    if omitted:data['unavailable'].append('Group profiles not included for columns not explicitly named in request: '+', '.join(omitted))
    if len(json.dumps(data))>16000:
        data['groups']={}
        data['unavailable'].append('All group profiles omitted to fit the context budget; full-column statistics remain available.')
    text=json.dumps(data,ensure_ascii=False,allow_nan=False)
    if len(text)>16000:raise ValueError('Input profile exceeds budget; cannot silently truncate required information')
    messages[-1]['content']+='\n\nInput dataset profile (input-profile-s2-v1; computed from the input CSV):\n'+text
    return messages
