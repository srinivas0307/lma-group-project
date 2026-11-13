"""Deterministic synthetic DRAFT suite. Run with python -m eval.contract_quality.build_suite."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'benchmark/contract_quality_v1'
DOMAINS = ['warehouse', 'solar', 'courier', 'nursery', 'reservoir', 'fabrication']

def predicate(kind, **kw):
    return dict(type=kind, **kw)

def cond(column, op, value=None):
    return dict(column=column, op=op, **({} if value is None else {'value': value}))

def program(lines):
    return 'def transform(df):\n    out = df.copy()\n' + ''.join('    '+s+'\n' for s in lines) + '    return out\n'

def build(destination=SUITE):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tasks, unsupported = [], []
    for index, domain in enumerate(DOMAINS):
        split = 'dev' if index < 4 else 'selection'
        rng = np.random.default_rng(719 + index)
        x, y, group, label, key, sub, date = [domain+'_'+s for s in ['reading', 'reading_backup', 'zone', 'priority', 'batch', 'unit', 'date']]
        a = rng.uniform(1, 90, 48); a[::7] = np.nan; a[1] = 30+index
        df = pd.DataFrame({x:a, y:rng.uniform(5,100,48), group:['east','west','north']*16,
                           label:['low','high','low','mid',None,'low']*8,
                           key:np.arange(48)//4, sub:np.arange(48)%2,
                           date:[f'{i%27+1:02d}/03/2025' for i in range(48)]})
        filename = f'data/{domain}.csv'
        (destination/'data').mkdir(exist_ok=True)
        df.to_csv(destination/filename, index=False)
        df = pd.read_csv(destination/filename)  # labels are computed from exactly the persisted input
        def add(family, tag, request, changes, preservation, good, bad, alternate=None):
            tid = f'{domain}_{tag}'
            tasks.append(dict(id=tid, dataset=domain, split=split, family=family, pattern=tag,
                data_file=filename, request=request+' Keep everything else unchanged; retain original row indices.',
                review_status='draft', assumptions=['All statistics use the input, excluding missing values.'],
                gold_contract=dict(contract_version='1.0', must_change=changes, must_preserve=preservation),
                reference_code=program(good), positive_codes=[program(alternate or good)],
                negative_codes=[dict(family=tag, code=program(bad)),
                                dict(family='no_op', code=program([]))]))
        def preserved(exceptions=(), rows=True):
            return ([predicate('row_count', op='unchanged')] if rows else []) + [predicate('columns_unchanged', target='*', **{'except':list(exceptions)})]
        def relation(col, fn, args, where=None):
            return predicate('column_value_relation', target=col, fn=fn, args=args, **({'where':where} if where else {}))
        # Filters test strict/inclusive boundaries, the similar-name column, and compound scopes.
        for j, (op, symbol) in enumerate([('>', '>'), ('>=','>='), ('<','<'), ('<=','<=')]):
            threshold=30+index
            c=cond(x,op,threshold)
            expr=f"out[{x!r}] {symbol} {threshold}"
            wrong=f"out[{y!r}] {symbol} {threshold}"
            add('filter',f'filter_{j}',f'Keep only rows where {x} {symbol} {threshold}; missing readings do not qualify.',
                [predicate('rows_removed_where',condition={'not':c})],
                [predicate('rows_retained_where',condition=c)]+preserved(rows=False),
                [f'out = out.loc[{expr}].copy()'],[f'out = out.loc[{wrong}].copy()'],
                [f'out = out.drop(out.index[~({expr})])'])
        for j, (keys,keep) in enumerate([([key],'first'),([key],'last'),([key,sub],'first'),([key,sub],'last')]):
            add('deduplicate',f'dedup_{j}',f'Remove duplicates by {keys!r}, retaining the {keep} row of each group in input order.',
                [predicate('duplicate_count',target=keys,op='eq',value=0,keep=keep)],preserved(rows=False),
                [f'out = out.drop_duplicates(subset={keys!r}, keep={keep!r})'],
                [f'out = out.drop_duplicates(subset={keys!r}, keep={"last" if keep=="first" else "first"!r})'],
                [f'out = out.loc[~out.duplicated(subset={keys!r}, keep={keep!r})].copy()'])
        for j in range(4):
            target = label if j==2 else x
            where=cond(target,'is_null')
            if j==3: where={'all':[where,cond(group,'==','east')]}
            val = (float(df[x].median()) if j==0 else float(df[x].mean()) if j==1 else
                   df[label].mode().iloc[0] if j==2 else float(df.loc[df[group]=='east',x].median()))
            desc=['the full-column median','the full-column mean','the unique most frequent nonmissing value',f'the median of {x} among rows with {group} = east'][j]
            mask=f'out[{target!r}].isna()'+(f' & (out[{group!r}] == "east")' if j==3 else '')
            add('missing',f'fill_{j}',f'Fill missing {target}'+(f' only in {group} = east' if j==3 else '')+f' with {desc}. Ignore missing values when computing statistics.',
                [relation(target,'const',[{'lit':val}],where)],preserved([target])+[predicate('column_values_unchanged',target=target,where={'not':where})],
                [f'out.loc[{mask}, {target!r}] = {val!r}'],
                [f'out.loc[{mask}, {target!r}] = {"WRONG" if j==2 else val+1!r}'])
        # Explicit datetime format is expressible in schema v1 cast's optional third argument.
        conversions=[(label,'category',f'pd.Categorical(out[{label!r}], categories=["low","mid","high"], ordered=True)',
                      f'pd.Categorical(out[{label!r}], categories=["high","mid","low"], ordered=True)',
                      'an ordered category with order low < mid < high',dict(ordered=True,categories=['low','mid','high'])),
                     (date,'datetime',f'pd.to_datetime(out[{date!r}], format="%d/%m/%Y")',f'pd.to_datetime(out[{date!r}], format="%m/%d/%Y", errors="coerce")','datetime using format %d/%m/%Y',{}),
                     (key,'float',f'out[{key!r}].astype(float)',f'out[{key!r}].astype(str)','float',{}),
                     (sub,'string',f'out[{sub!r}].astype(str)',f'out[{sub!r}].astype(float)','string',{})]
        for j,(target,dtype,expr,bad,desc,extra) in enumerate(conversions):
            args=[{'col':target},{'lit':dtype}]+([{'lit':'%d/%m/%Y'}] if dtype=='datetime' else [])
            changes=[predicate('column_dtype',target=target,dtype=dtype,**extra)]
            changes += [predicate('column_values_unchanged',target=target)] if dtype=='category' else [relation(target,'cast',args)]
            add('convert',f'convert_{j}',f'Convert {target} to {desc}.',changes,preserved([target]),
                [f'out[{target!r}] = {expr}'],[f'out[{target!r}] = {bad}'])
        edits=[('clip',[{'col':x},{'lit':20},{'lit':60}],f'out[{x!r}].clip(20,60)',f'out[{x!r}]+20',f'Clip {x} to [20,60].',x),
               ('map',[{'col':label},{'lit':{'low':1,'mid':2,'high':3}}],f'out[{label!r}].map({{"low":1,"mid":2,"high":3}})',f'out[{label!r}].map({{"low":3,"mid":2,"high":1}})',f'Map {label}: low to 1, mid to 2, high to 3; preserve missing cells.',label),
               ('add',[{'col':y},{'lit':index+2}],f'out[{y!r}]+{index+2}',f'out[{x!r}]+{index+2}',f'Add {index+2} to {y}.',y),
               ('mul',[{'col':y},{'lit':2}],f'out[{y!r}]*2',f'out[{y!r}]+2',f'Double {y} only for rows with {group} = west.',y)]
        for j,(fn,args,expr,bad,req,target) in enumerate(edits):
            where=cond(group,'==','west') if j==3 else None
            extra=[predicate('column_values_unchanged',target=target,where={'not':where})] if where else []
            good=[f'out[{target!r}] = {expr}']; wrong=[f'out[{target!r}] = {bad}']
            if where:
                good=[f'mask = out[{group!r}] == "west"',f'out.loc[mask,{target!r}] = ({expr}).loc[mask]']
                wrong=[f'out[{target!r}] = {expr}']
            add('edit',f'edit_{j}',req,[relation(target,fn,args,where)],preserved([target])+extra,good,wrong)
        unsupported.append(dict(id=domain+'_ambiguous',dataset=domain,split=split,data_file=filename,
            request='Clean up the readings and remove bad rows.',expected_outcome='ESCALATE',review_status='draft',
            reason='No definition of bad rows or intended changes. Not scored as a supported contract task.'))
    for split in ['dev','selection']:
        (destination/f'{split}.jsonl').write_text(''.join(json.dumps(t,allow_nan=False)+'\n' for t in tasks if t['split']==split))
    (destination/'unsupported.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in unsupported))
    review=destination/'review.csv'
    # Never overwrite real review annotations when regenerating a deterministic draft.
    if not review.exists():
        with review.open('w') as f:
            w=csv.writer(f);w.writerow(['task_id','status','reviewer','notes'])
            w.writerows((t['id'],'draft','','') for t in tasks+unsupported)
    files=[p for p in destination.rglob('*') if p.is_file() and p.name not in ['manifest.json','review.csv']]
    manifest=dict(version='cq-draft-v1',seed=719,status='synthetic_draft',counts={'dev':80,'selection':40,'unsupported':6},
        limitations=['Shared transformation templates across dataset-disjoint splits; not an independent benchmark.',
                     'Alternate positive may repeat reference code; independent implementations are not available for every pattern.',
                     'Multi-edit, keep-none, tied modes and numeric-text conversion remain coverage gaps.'],
        sha256={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return tasks

if __name__=='__main__':
    build()
