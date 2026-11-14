"""Conservative, request-only statistic routing. No contracts or reference code enter here."""
import json
import math
import re
import pandas as pd
from igt.agents.intent_parser import build_messages

PROTOCOL='p1-v2+sf1'
RULES='''\n\nSelective input facts (sf1): facts are measurements of the INPUT, not transformation instructions.
Use a fact only for the column, statistic and input-row scope requested. Copy its full-precision value.
For missing-value fills, constrain the replacement to originally missing cells (and any requested group), and preserve the other cells.
Use exact requested column names, deduplication keys and keep rules. A rename uses columns_added, columns_removed and identity; there is no rename function.
Do not preserve an original column that the request removes or renames. Missing-to-value fills do not increase null counts.
'''

class FactUnavailable(ValueError):
    """A needed statistic cannot be routed or computed reliably; do not invent it."""

def names_in(request, columns):
    spans=[]
    for col in sorted(columns,key=lambda c:-len(str(c))):
        for m in re.finditer(r'(?<!\w)'+re.escape(str(col))+r'(?!\w)',request,re.I):
            if not any(m.start()<b and m.end()>a for a,b,_ in spans):spans.append((m.start(),m.end(),col))
    return list(dict.fromkeys(c for _,_,c in sorted(spans)))

def scalar(value):
    if hasattr(value,'item'):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):raise FactUnavailable('Statistic is not finite')
    return value

def selective_facts(request,df):
    text=' '.join(str(request).split())
    # Deliberately bounded first version: data-dependent fill requests only.
    if not re.search(r'\b(missing|blank|empty|impute|fill)\b',text,re.I):return []
    statistic_text=text
    for col in sorted(df.columns,key=lambda c:-len(str(c))):
        statistic_text=re.sub(r'(?<!\w)'+re.escape(str(col))+r'(?!\w)',' ',statistic_text,flags=re.I)
    statistic_text=re.sub(r'''"[^"]*"|'[^']*' '''.rstrip(),' ',statistic_text)
    kinds=[]
    for kind,pattern in [('median',r'\bmedian\b'),('mean',r'\b(mean|average)\b'),('mode',r'\b(mode|most\s+(?:frequent|common))\b')]:
        if re.search(pattern,statistic_text,re.I):kinds.append(kind)
    if not kinds:return []
    if len(kinds)!=1:raise FactUnavailable('Multiple statistics in one request are not supported by sf1')
    kind=kinds[0]
    if len({str(c).casefold() for c in df.columns})!=len(df.columns):raise FactUnavailable('Case-ambiguous input column names')
    mentioned=names_in(text,df.columns)
    # Single equality-group grammar. All matching conditions must agree.
    matches=[]
    for col in mentioned:
        pattern=r'(?<!\w)'+re.escape(str(col))+r'''(?!\w)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s,;]+))'''
        for match in re.finditer(pattern,text,re.I):
            raw=next(v for v in match.groups() if v is not None)
            if match.group(3) is not None:raw=raw.rstrip('.')
            series=df[col]
            if pd.api.types.is_numeric_dtype(series):
                try:value=float(raw);value=int(value) if value.is_integer() else value
                except ValueError:raise FactUnavailable('Group value is not numeric')
            else:value=raw
            matches.append((col,value))
    groups=list(dict.fromkeys(matches))
    if len(groups)>1:raise FactUnavailable('Multiple group conditions are not supported by sf1')
    grouped=bool(re.search(r'\b(among|within|where|by|per|group|in|for|if|when|only)\b',statistic_text,re.I))
    if re.search(r'\b(positive|negative|above|below|greater|less|subset)\b|>=|<=|!=|(?<![=])>(?![=])|(?<![=])<(?![=])',statistic_text,re.I):
        raise FactUnavailable('Statistic uses an unsupported conditional scope')
    if grouped and not groups:raise FactUnavailable('Conditional statistic needs an explicit column = value scope')
    # Group equality without explicit scope language can be a distinct output condition.
    if groups and not grouped:raise FactUnavailable('Ambiguous statistic scope')
    targets=[c for c in mentioned if not groups or c!=groups[0][0]]
    if len(targets)!=1:raise FactUnavailable('Statistic target must name exactly one input column')
    target=targets[0];scope={'kind':'full_column'};values=df[target]
    full_scope=bool(re.search(r'\b(full[- ]column|entire column|whole column)\b',statistic_text,re.I))
    conditional_scope=bool(re.search(r'\b(among|within)\b',statistic_text,re.I))
    if groups and full_scope and conditional_scope:raise FactUnavailable('Conflicting full-column and conditional statistic scopes')
    if groups and not full_scope and not conditional_scope:raise FactUnavailable('Specify full-column or among/within scope explicitly')
    if groups and not full_scope:
        col,value=groups[0];mask=df[col].eq(value).fillna(False)
        if not mask.any():raise FactUnavailable('Requested statistic group is empty')
        values=values.loc[mask];scope={'kind':'conditional','where':{'column':col,'op':'==','value':value}}
    values=values.dropna()
    if values.empty:raise FactUnavailable('No nonmissing values for the requested statistic')
    if kind in ('median','mean'):
        if not pd.api.types.is_numeric_dtype(values):raise FactUnavailable('Numeric statistic requested on nonnumeric input')
        value=scalar(getattr(values,kind)())
    else:
        modes=values.mode()
        if len(modes)>1:
            if re.search(r'\b(smallest|lowest|alphabetically\s+first)\b',text,re.I):value=scalar(modes.iloc[0])
            elif re.search(r'\b(largest|highest|alphabetically\s+last)\b',text,re.I):value=scalar(modes.iloc[-1])
            else:raise FactUnavailable('Mode is tied and the request does not specify a supported tie rule')
        else:value=scalar(modes.iloc[0])
    return [dict(column=target,statistic=kind,value=value,scope=scope,nonmissing_count=len(values),missing_values='excluded')]

def schema_entry(df):
    return dict(n_rows=len(df),columns=[dict(name=str(c),dtype=str(df[c].dtype),nulls=int(df[c].isna().sum())) for c in df],head_csv=df.head(3).to_csv(index=False))

def messages(task,df,original=False):
    public=dict(request=' '.join(str(task['request']).split()),data_file=task['data_file'])
    result=build_messages(public,{public['data_file']:schema_entry(df)},[],version='p1-v2')
    if original:return result
    facts=selective_facts(public['request'],df)
    result[0]['content']+=RULES
    # The original prompt template already places Request last. Insert facts before it.
    if facts:
        suffix='Request: '+public['request']
        if not result[-1]['content'].endswith(suffix):raise ValueError('Unexpected prompt template')
        prefix=result[-1]['content'][:-len(suffix)]
        result[-1]['content']=prefix+'Required input facts (sf1):\n'+json.dumps(facts,ensure_ascii=False,allow_nan=False)+'\n\n'+suffix
    return result
