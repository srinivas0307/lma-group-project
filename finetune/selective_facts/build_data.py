"""Versioned replay data plus new scoped examples; no evaluation tasks enter SFT."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from igt.agents.intent_parser import contract_to_text,extract_contract
from igt.verifier import verify
from .context import messages,selective_facts,FactUnavailable,PROTOCOL
ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'finetune/data/selective_facts_v1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_rows(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
def pred(kind,**kwargs):return dict(type=kind,**kwargs)
def condition(c,op,value=None):return dict(column=c,op=op,**({'value':value} if value is not None else {}))
def program(line):return 'def transform(df):\n    out = df.copy()\n    '+line+'\n    return out\n'

def synthetic(out,train_domains=24,val_domains=6):
    rows=[];validation=[]
    for split,count in [('train',train_domains),('val',val_domains)]:
        for i in range(count):
            rng=np.random.default_rng(93000+i+(1000 if split=='val' else 0))
            name=f'sf_{split}_{i:02d}';x,y,group,key,unit,cat=[name+'_'+s for s in ['amount','amount_backup','zone','batch','unit','status']]
            values=rng.uniform(2,90,64);values[::5]=np.nan
            df=pd.DataFrame({x:values,y:rng.uniform(3,70,64),group:['east','west']*32,key:np.arange(64)//4,unit:np.arange(64)%2,cat:['low','low','high',None]*16})
            filename=out/'data'/f'{name}.csv';filename.parent.mkdir(parents=True,exist_ok=True);df.to_csv(filename,index=False);df=pd.read_csv(filename)
            def preserve(except_cols=(),same_rows=True):return ([pred('row_count',op='unchanged')] if same_rows else [])+[pred('columns_unchanged',target='*',**{'except':list(except_cols)})]
            def add(tag,request,changes,keeps,after,bad,line):
                contract=dict(contract_version='1.0',must_change=changes,must_preserve=keeps)
                parsed=extract_contract(contract_to_text(contract))
                if parsed['status']!='ok':raise ValueError(f'{name}/{tag}: {parsed}')
                if verify(contract,df,after)[0].status!='PASS':raise ValueError(f'Gold rejects correct synthetic output: {name}/{tag}')
                if verify(contract,df,bad)[0].status=='PASS':raise ValueError(f'Gold accepts wrong synthetic output: {name}/{tag}')
                task=dict(id=name+'_'+tag,source_dataset=name,data_file=str(filename.relative_to(ROOT)),transformation_type=tag,
                          request=request+' Preserve all other values and original row indices.',contract=contract,reference_code=program(line),split=split)
                rows.append(task);validation.append(dict(id=task['id'],correct_accepted=True,wrong_rejected=True))
            for kind in ['median','mean','mode','group_median','group_mean']:
                target=cat if kind=='mode' else x
                grouped=kind.startswith('group_');stat=kind.removeprefix('group_')
                mask=df[target].isna() & (df[group].eq('east') if grouped else True)
                scope=df.loc[df[group].eq('east'),target] if grouped else df[target]
                value=scope.mode().iloc[0] if stat=='mode' else float(getattr(scope,stat)())
                where=condition(target,'is_null')
                if grouped:where={'all':[where,condition(group,'==','east')]}
                request=f'Fill missing {target}'+(f' only in {group} = east' if grouped else '')+f' with the '+(f'{stat} of {target} among rows with {group} = east' if grouped else ('unique most frequent nonmissing value' if stat=='mode' else f'full-column {stat}'))+'.'
                after=df.copy();after.loc[mask,target]=value;bad=df.copy();bad.loc[mask,target]='WRONG' if stat=='mode' else value+1
                add('fill_'+kind,request,[pred('column_value_relation',target=target,fn='const',args=[{'lit':value}],where=where)],
                    preserve([target])+[pred('column_values_unchanged',target=target,where={'not':where})],after,bad,
                    f'out.loc[out[{target!r}].isna()'+(f' & out[{group!r}].eq("east")' if grouped else '')+f', {target!r}] = {value!r}')
            # Distinguish the rows to change from the rows used to compute the statistic.
            mask=df[x].isna() & df[group].eq('east');value=float(df[x].median())
            where={'all':[condition(x,'is_null'),condition(group,'==','east')]}
            after=df.copy();after.loc[mask,x]=value;bad=df.copy();wrong=float(df.loc[df[group].eq('east'),x].median());bad.loc[mask,x]=wrong if abs(wrong-value)>1e-6 else value+1
            add('scoped_global_median',f'Fill missing {x} only in {group} = east with the full-column median of {x}.',
                [pred('column_value_relation',target=x,fn='const',args=[{'lit':value}],where=where)],
                preserve([x])+[pred('column_values_unchanged',target=x,where={'not':where})],after,bad,
                f'out.loc[out[{x!r}].isna() & out[{group!r}].eq("east"), {x!r}] = {value!r}')
            after=df.copy();after.loc[after[unit].eq(0),unit]=float('nan')
            where=condition(unit,'==',0)
            add('sentinel_missing',f'A {unit} value of 0 means unrecorded. Mark those cells as missing.',
                [pred('column_value_relation',target=unit,fn='null',args=[],where=where)],
                preserve([unit])+[pred('column_values_unchanged',target=unit,where={'not':where})],after,df.copy(),
                f'out.loc[out[{unit!r}].eq(0), {unit!r}] = float("nan")')
            # Literal fill does not need any dataset statistic.
            where=condition(x,'is_null');after=df.fillna({x:0});bad=df.fillna({x:1})
            add('literal_fill',f'Fill missing {x} with 0.',[pred('column_value_relation',target=x,fn='const',args=[{'lit':0}],where=where)],preserve([x])+[pred('column_values_unchanged',target=x,where={'not':where})],after,bad,f'out[{x!r}] = out[{x!r}].fillna(0)')
            renamed=x+'_renamed';after=df.rename(columns={x:renamed})
            add('rename',f'Rename {x} to {renamed}.',[pred('columns_added',target=[renamed]),pred('columns_removed',target=[x]),pred('column_value_relation',target=renamed,fn='identity',args=[{'col':x}])],preserve([x]),after,df.copy(),f'out = out.rename(columns={{{x!r}:{renamed!r}}})')
            for keys,tag in [([key],'single'),([key,unit],'composite')]:
                for keep in ['first','last']:
                    add('dedup_'+tag+'_'+keep,f'Remove duplicates by {", ".join(keys)}, keeping the {keep} row in each group.',
                        [pred('duplicate_count',target=keys,op='eq',value=0,keep=keep)],preserve(same_rows=False),df.drop_duplicates(keys,keep=keep),df.drop_duplicates(keys,keep='last' if keep=='first' else 'first'),f'out = out.drop_duplicates({keys!r}, keep={keep!r})')
            mask=df[group].eq('west');where=condition(group,'==','west');after=df.copy();after.loc[mask,y]*=2;bad=df.copy();bad[y]*=2
            add('scoped_edit',f'Double {y} only for rows with {group} = west.',[pred('column_value_relation',target=y,fn='mul',args=[{'col':y},{'lit':2}],where=where)],preserve([y])+[pred('column_values_unchanged',target=y,where={'not':where})],after,bad,f'out.loc[out[{group!r}].eq("west"), {y!r}] *= 2')
            where=condition(x,'>=',30);after=df.loc[df[x]>=30];bad=df.loc[df[y]>=30]
            add('filter',f'Keep only rows with {x} >= 30; missing values do not qualify.',[pred('rows_removed_where',condition={'not':where})],[pred('rows_retained_where',condition=where)]+preserve(same_rows=False),after,bad,f'out = out.loc[out[{x!r}] >= 30].copy()')
            after=df.copy();after[key]=after[key].astype(str);bad=df.copy()
            add('cast_string',f'Convert {key} to string.',[pred('column_dtype',target=key,dtype='string'),pred('column_value_relation',target=key,fn='cast',args=[{'col':key},{'lit':'string'}])],preserve([key]),after,bad,f'out[{key!r}] = out[{key!r}].astype(str)')
            after=df.copy();after[x]=after[x].clip(20,60)
            add('clip',f'Clip {x} to the range [20, 60].',[pred('column_value_relation',target=x,fn='clip',args=[{'col':x},{'lit':20},{'lit':60}])],preserve([x]),after,df.copy(),f'out[{x!r}] = out[{x!r}].clip(20,60)')
            after=df.copy();after.loc[after[x].isna(),x]=df.loc[after[x].isna(),y]
            add('copy_fill',f'Where {x} is missing, use the corresponding {y} value.',[pred('column_value_relation',target=x,fn='identity',args=[{'col':y}],where=condition(x,'is_null'))],preserve([x])+[pred('column_values_unchanged',target=x,where=condition(x,'not_null'))],after,df.copy(),f'out[{x!r}] = out[{x!r}].fillna(out[{y!r}])')
    return rows,validation

def build(out=DEFAULT,train_domains=24,val_domains=6):
    out=Path(out)
    if out.exists():raise ValueError('Use a new dataset directory; never overwrite a versioned build')
    out.mkdir(parents=True)
    source=ROOT/'finetune/data/v2/checked.jsonl';base=[json.loads(l) for l in source.read_text().splitlines()]
    split_ids={split:{json.loads(l)['id'] for l in (ROOT/f'finetune/data/v2/sft_{split}.jsonl').read_text().splitlines()} for split in ['train','val']}
    if split_ids['train']&split_ids['val']:raise ValueError('Overlapping original splits')
    new,checked=synthetic(out,train_domains,val_domains)
    data={e['data_file']:pd.read_csv(ROOT/e['data_file']) for e in base+new}
    output={'train':[],'val':[]};skipped=[];precision_updates=[];coverage=collections.Counter()
    for e in base+new:
        split=e.get('split') or ('train' if e['id'] in split_ids['train'] else 'val' if e['id'] in split_ids['val'] else None)
        if split is None:raise ValueError(f'Unassigned example {e["id"]}')
        df=data[e['data_file']]
        try:
            msg=messages(e,df);facts=selective_facts(e['request'],df)
        except FactUnavailable as exc:
            if e in new:raise
            skipped.append(dict(id=e['id'],reason=str(exc)));continue
        parsed=extract_contract(contract_to_text(e['contract']))
        if parsed['status']!='ok':raise ValueError(f'Invalid target: {e["id"]}')
        # Input facts must support the annotated replacement literal. Reject inconsistent labels.
        if facts:
            fact=facts[0];literals=[p['args'][0]['lit'] for p in e['contract']['must_change'] if p.get('type')=='column_value_relation' and p.get('target')==fact['column'] and p.get('fn')=='const']
            if not literals or any((abs(float(v)-float(fact['value']))>1e-6 if isinstance(v,(float,int)) and isinstance(fact['value'],(float,int)) else v!=fact['value']) for v in literals):
                raise ValueError(f'Fact/target literal mismatch: {e["id"]}')
            # Correct inherited rounded labels to the exact input-derived fact in this new build only.
            for predicate in e['contract']['must_change']:
                if predicate.get('type')=='column_value_relation' and predicate.get('target')==fact['column'] and predicate.get('fn')=='const':
                    old=predicate['args'][0]['lit']
                    if old!=fact['value']:
                        precision_updates.append(dict(id=e['id'],old=old,new=fact['value']))
                        predicate['args'][0]['lit']=fact['value']
        rec=dict(id=e['id'],source_dataset=e['source_dataset'],transformation_type=e['transformation_type'],prompt_version=PROTOCOL,
                 data_file=e['data_file'],facts=facts,messages=msg+[dict(role='assistant',content=contract_to_text(e['contract']))])
        output[split].append(rec);coverage[split+('/with_facts' if facts else '/no_facts')]+=1
    train_sources={r['source_dataset'] for r in output['train']};val_sources={r['source_dataset'] for r in output['val']}
    if train_sources&val_sources:raise ValueError('Dataset leakage')
    for split,rows in output.items():random.Random(1234).shuffle(rows);write_rows(out/f'sft_{split}.jsonl',rows)
    write_rows(out/'synthetic_examples.jsonl',new)
    manifest=dict(protocol=PROTOCOL,counts={k:len(v) for k,v in output.items()},coverage=dict(coverage),skipped=skipped,precision_updates=precision_updates,
        context_sha256=digest(Path(__file__).parent/'context.py'),source_sha256={str(source.relative_to(ROOT)):digest(source),**{f'finetune/data/v2/sft_{s}.jsonl':digest(ROOT/f'finetune/data/v2/sft_{s}.jsonl') for s in ['train','val']}},
        raw_data_sha256={name:digest(ROOT/name) for name in sorted(data)},
        train_sources=sorted(train_sources),validation_sources=sorted(val_sources),synthetic_semantic_checks=checked,
        review_status='synthetic additions automatically validated; no independent human review',
        limitations=['Conservative sf1 routing skips unresolved statistic requests; see skipped list.',
                     'Synthetic templates can overlap conceptually with development templates; not independent test evidence.',
                     'Original replay examples inherit existing v2 checked provenance; not all rerun here.'])
    manifest['sha256']={p.name:digest(p) for p in out.glob('*.jsonl')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['counts','coverage','review_status']},indent=2),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();build(a.out)
if __name__=='__main__':main()
