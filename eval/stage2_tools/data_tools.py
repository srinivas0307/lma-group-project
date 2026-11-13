"""Read-only, full-precision queries over the input CSV; no code execution or output access."""
import json
import math
import pandas as pd
from igt.contracts.schema import Condition
from igt.verifier.conditions import evaluate

MAX_RESULT_CHARS=6000

def scalar(value):
    if pd.isna(value):return None
    if hasattr(value,'item'):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,pd.Timestamp):return value.isoformat()
    return value

class DataTools:
    def __init__(self,df):self._input=df.copy(deep=True)

    def rows(self,where=None):
        if where is None:return self._input
        def size(x,depth=0):
            if depth>8:raise ValueError('Condition exceeds nesting limit')
            return 1+sum(size(v,depth+1) for v in (x.values() if isinstance(x,dict) else x if isinstance(x,list) else []))
        if size(where)>128:raise ValueError('Condition exceeds size limit')
        condition=Condition.model_validate(where)
        for c in condition.columns():self.column(c)
        return self._input.loc[evaluate(condition,self._input)]

    def column(self,name):
        if not isinstance(name,str) or name not in self._input.columns:raise ValueError(f'Unknown input column: {name!r}')
        return name

    def _profile(self,series):
        nonmissing=series.dropna();counts=nonmissing.value_counts();modes=series.mode(dropna=True)
        out=dict(rows=len(series),missing=int(series.isna().sum()),dtype=str(series.dtype),distinct=int(series.nunique(dropna=True)),
            top_values=[dict(value=scalar(v),count=int(n)) for v,n in counts.head(10).items()],
            values_omitted=max(0,len(counts)-10),modes=[scalar(v) for v in modes.iloc[:10]],
            mode_tie=len(modes)>1,mode_values_total=len(modes),modes_truncated=len(modes)>10,
            null_policy='Statistics and frequencies exclude missing cells; no tie-break is inferred.')
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            out['statistics']={key:scalar(getattr(nonmissing,key)()) if len(nonmissing) else None for key in ['min','max','mean','median']}
            out['nonfinite_nonmissing']=sum(not math.isfinite(float(v)) for v in nonmissing)
        return out

    def profile_column(self,column,where=None):
        self.column(column)
        return dict(column=column,where=where,**self._profile(self.rows(where)[column]))

    def count_rows(self,where):
        return dict(matching_rows=len(self.rows(where)),total_rows=len(self._input),where=where)

    def inspect_rows(self,columns,where=None,n=5):
        if not isinstance(columns,list) or not 1<=len(columns)<=8 or len(set(columns))!=len(columns):raise ValueError('Select 1 to 8 distinct columns')
        for c in columns:self.column(c)
        if type(n) is not int or not 1<=n<=5:raise ValueError('n must be an integer from 1 to 5')
        selected=self.rows(where)[columns]
        return dict(matching_rows=len(selected),returned_rows=min(n,len(selected)),
            rows=[{c:scalar(v) for c,v in row.items()} for row in selected.head(n).to_dict(orient='records')],
            truncated=len(selected)>n)

    def group_profile(self,column,by,where=None):
        self.column(column)
        if not isinstance(by,list) or not 1<=len(by)<=2 or len(set(by))!=len(by):raise ValueError('by must contain 1 or 2 distinct input columns')
        for c in by:self.column(c)
        groups=self.rows(where).groupby(by[0] if len(by)==1 else by,dropna=False,sort=False,observed=True)
        if groups.ngroups>12:raise ValueError('More than 12 groups; narrow the query with where')
        result=[]
        for key,frame in groups:
            keys=(key,) if len(by)==1 else key
            result.append(dict(group={c:scalar(v) for c,v in zip(by,keys)},profile=self._profile(frame[column])))
        return dict(column=column,by=by,where=where,groups=result,missing_group_keys='included')

    def call(self,name,args):
        functions={'profile_column':self.profile_column,'count_rows':self.count_rows,'inspect_rows':self.inspect_rows,'group_profile':self.group_profile}
        try:
            if name not in functions:raise ValueError(f'Unknown tool: {name}')
            if not isinstance(args,dict):raise ValueError('args must be an object')
            result=functions[name](**args)
            encoded=json.dumps(result,ensure_ascii=False,allow_nan=False)
            if len(encoded)>MAX_RESULT_CHARS:raise ValueError('Result exceeds 6000 characters; narrow the condition, choose fewer columns or request fewer rows')
            return dict(ok=True,result=result)
        except (TypeError,ValueError,KeyError) as error:
            return dict(ok=False,error_type=type(error).__name__,error=str(error)[:500])
