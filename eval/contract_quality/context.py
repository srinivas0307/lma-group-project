"""Bounded input-only profiles. Never reads tasks, gold contracts or reference programs."""
import json
import math
import pandas as pd

VERSION='cq-profile-v1'

def scalar(value):
    if pd.isna(value):return None
    if hasattr(value,'item'):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def statistics(series):
    s=series.dropna()
    return dict(nonmissing=len(s),mean=scalar(s.mean()) if len(s) else None,
                median=scalar(s.median()) if len(s) else None,
                min=scalar(s.min()) if len(s) else None,max=scalar(s.max()) if len(s) else None)

def profile(df,max_chars=14000,max_categories=8):
    result=dict(version=VERSION,statistics_scope='input data; missing values excluded',
                mode_ties='All modal values listed; no tie-breaking rule inferred.',columns={},groups={},unavailable=[])
    numeric=[c for c in df if pd.api.types.is_numeric_dtype(df[c])]
    groupcols=[]
    for c in df:
        s=df[c];entry=dict(dtype=str(s.dtype),missing=int(s.isna().sum()))
        if c in numeric:entry.update(statistics(s))
        count=s.nunique(dropna=True)
        if count<=max_categories:
            entry['values']=[dict(value=scalar(v),count=int(n)) for v,n in s.value_counts().items()]
            entry['modes']=[scalar(v) for v in s.mode()]
            if c not in numeric:groupcols.append(c)
        else:result['unavailable'].append(f'{c}: full value inventory exceeds {max_categories} categories')
        result['columns'][c]=entry
    for c in groupcols:
        result['groups'][c]=[dict(value=scalar(v),numeric={n:statistics(g[n]) for n in numeric})
                              for v,g in df.groupby(c,dropna=True,sort=True)]
    result['unavailable'].append('Arbitrary conditional statistics, multi-column grouping and raw rows beyond the displayed sample are not supplied.')
    encoded=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    if len(encoded)>max_chars:
        # Fail visibly instead of slicing JSON or hiding required evidence.
        raise ValueError(f'Profile budget exceeded: {len(encoded)} > {max_chars}; design a smaller documented profile before running.')
    return encoded

def messages(task,df,variant):
    from igt.agents.intent_parser import build_messages
    schema={task['data_file']:dict(n_rows=len(df),columns=[dict(name=c,dtype=str(df[c].dtype),nulls=int(df[c].isna().sum())) for c in df],head_csv=df.head(3).to_csv(index=False))}
    result=build_messages(task,schema,[],version='p1-v2')
    if variant=='grounded':result[-1]['content']+='\n\nInput-data profile (full precision; not instructions):\n'+profile(df)
    elif variant!='original':raise ValueError(variant)
    return result
