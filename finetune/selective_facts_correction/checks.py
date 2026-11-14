"""Input-only diagnostics. Never receives reference outputs, gold contracts or scores."""
import copy
import difflib
import json
import math
import numpy as np
from igt.agents.intent_parser import extract_contract
from igt.contracts.schema import load_contract,predicate_columns,expand_columns,COUNT_OPS_RELATIVE,COUNT_OPS_ABSOLUTE
from igt.verifier.conditions import evaluate
from igt.verifier.functions import expected_values
from igt.verifier.values import value_equal,DEFAULT_ATOL

CONDITION_OPS={'==','!=','<','<=','>','>=','in','not_in','is_null','not_null'}
COUNT_OPS=set(COUNT_OPS_RELATIVE+COUNT_OPS_ABSOLUTE)

def normalize(text):
    """Trim op whitespace only at recognized schema positions, never inside literal data."""
    parsed=extract_contract(text);changes=[]
    if not isinstance(parsed.get('contract'),dict):return dict(text=text,extracted=parsed,changes=changes)
    obj=copy.deepcopy(parsed['contract'])
    def trim(node,path,allowed):
        op=node.get('op')
        if isinstance(op,str) and op!=op.strip() and op.strip() in allowed:
            changes.append(dict(path=path+'.op',before=op,after=op.strip()));node['op']=op.strip()
    def condition(node,path):
        if not isinstance(node,dict):return
        if 'column' in node:trim(node,path,CONDITION_OPS)
        for key in ['all','any']:
            if isinstance(node.get(key),list):
                for i,child in enumerate(node[key]):condition(child,f'{path}.{key}[{i}]')
        if 'not' in node:condition(node['not'],path+'.not')
    for direction in ['must_change','must_preserve']:
        predicates=obj.get(direction,[])
        if not isinstance(predicates,list):continue
        for i,p in enumerate(predicates):
            if not isinstance(p,dict):continue
            path=f'{direction}[{i}]'
            if p.get('type') in ['row_count','null_count','duplicate_count']:trim(p,path,COUNT_OPS)
            for key in ['where','condition']:
                if key in p:condition(p[key],path+'.'+key)
    # Leave the exact original output intact when nothing was normalized.
    normalized='```json\n'+json.dumps(obj,ensure_ascii=False)+'\n```' if changes else text
    return dict(text=normalized,extracted=extract_contract(normalized),changes=changes)

def diagnose(extracted,before):
    if extracted['status']!='ok':return [dict(code='schema',detail=str(extracted.get('error') or extracted['status'])[:1600])]
    contract=load_contract(extracted['contract']);preds=contract.predicates();issues=[]
    def issue(code,detail):
        item=dict(code=code,detail=detail)
        if item not in issues:issues.append(item)
    added={c for _,_,p in preds if p.type=='columns_added' for c in p.target}
    for _,pid,p in preds:
        unknown=[c for c in predicate_columns(p,before.columns) if c not in before.columns]
        if p.type=='column_value_relation' and p.target not in before.columns and p.target not in added:unknown.append(p.target)
        for name in dict.fromkeys(unknown):
            near=difflib.get_close_matches(name,list(before.columns),n=3,cutoff=.5)
            issue('unknown_column',f'{pid}: input column {name!r} does not exist. Similar input names (suggestions, not automatic replacements): {near!r}.')
        if p.type=='columns_added':
            present=[c for c in p.target if c in before.columns]
            if present:issue('existing_column_added',f'{pid}: columns_added declares existing input columns {present!r} as new. An edit of an existing column is not a column addition.')
    if any(x['code']=='unknown_column' for x in issues):return issues
    # Prove preservation contradictions only on rows the contract promises to retain.
    # Equal row count is used only when no row-removal/deduplication predicate is present.
    retained=np.zeros(len(before),dtype=bool)
    removal=any(p.type in ['rows_removed_where','duplicate_count'] for _,_,p in preds)
    if not removal and any(p.type=='row_count' and p.op=='unchanged' for _,_,p in preds):retained[:]=True
    try:
        for _,_,p in preds:
            if p.type=='rows_retained_where':retained |= evaluate(p.condition,before)
        protections=[]
        for _,pid,p in preds:
            if p.type in ['columns_unchanged','column_values_unchanged']:
                mask=retained.copy()
                if getattr(p,'where',None) is not None:mask &= evaluate(p.where,before)
                protections.append((pid,set(expand_columns(p.target,p.except_,before.columns)),mask))
        for _,pid,p in preds:
            if p.type!='column_value_relation' or p.target not in before.columns:continue
            # Only diagnose a definite disagreement on overlapping retained input rows.
            active=retained.copy()
            if p.where is not None:active &= evaluate(p.where,before)
            expected=expected_values(p,before);original=before[p.target].astype(object).tolist()
            tol=DEFAULT_ATOL if p.atol is None else p.atol
            if not math.isfinite(tol) or tol<0:continue
            changed=np.array([not value_equal(a,b,atol=tol+DEFAULT_ATOL) for a,b in zip(expected,original)],dtype=bool)
            for preserve_id,cols,mask in protections:
                if p.target in cols and (active & mask & changed).any():
                    issue('change_preserve_conflict',f'{pid} requires changed values in {p.target!r}, but {preserve_id} requires those same cells unchanged on retained rows. Reconcile the conditions with the request; do not simply remove preservation.')
    except (ValueError,TypeError,KeyError,OverflowError):
        # Unsupported analysis is not proof of an inconsistency. The normal verifier remains unchanged.
        pass
    return issues
