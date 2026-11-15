"""Synthetic development fixtures: references are evaluation-only, never repair input."""
import copy
import pandas as pd

VERSION = 'repair-mechanisms-dev1'


def build_suite():
    cases = []
    def contract(change, preserve=(), except_=(), same_rows=False):
        return dict(contract_version='1.0', must_change=change,
                    must_preserve=[dict(type='columns_unchanged',target='*',**{'except':list(except_)}),
                                   *preserve,*([dict(type='row_count',op='unchanged')] if same_rows else [])])
    def add(name, request, data, spec, bad, good, control=False):
        cases.append(dict(id=name,request=request,data=data,contract=spec,initial_code=bad,
                          reference_code=good,initial_expected_correct=control))
    def code(body):
        return 'def transform(df):\n'+''.join('    '+line+'\n' for line in body.splitlines())
    def dedup(name,data,keys,keep,bad):
        good=code(f"return df.drop_duplicates(subset={keys!r}, keep={keep!r}).copy()")
        add(name,f'Keep the {keep} original record for each combination of {keys!r}, in input order. Preserve all columns and values.',
            data,contract([dict(type='duplicate_count',target=keys,op='eq',value=0,keep=keep)]),code(bad),good)
    data=dict(record=[11,12,13,14,15],account=['x','y','x','y','z'],kind=['a','b','a','b','a'],rank=[8,7,2,1,5],value=[None,4.,9.,None,6.])
    dedup('dedup_original_first',data,['account','kind'],'first',"return df.sort_values('rank').drop_duplicates(['account','kind'],keep='first')")
    dedup('dedup_original_last',data,['account'],'last',"return df.sort_values('rank').drop_duplicates(['account'],keep='last')")
    dedup('dedup_complete_record',data,['account'],'first',"return df.groupby('account',as_index=False).first()")
    dedup('dedup_keep_repeated_groups',data,['account','kind'],'last',"df = df[df.groupby(['account','kind'])['record'].transform('size').eq(1)]\nreturn df.drop_duplicates(['account','kind'],keep='last')")
    dedup('dedup_crashed_candidate',data,['account'],'first',"return df.groupby('account',keep=False).first()")
    for n,categories in enumerate([['bronze','silver','gold'],['low','middle','high']],1):
        values=dict(record=[1,2,3,4],tier=[categories[1],categories[0],None,categories[2]])
        prefix="df['tier'] = df['tier'].astype('category')\n"
        expression=f"df['tier'].cat.set_categories({categories!r},ordered=True)"
        spec=contract([dict(type='column_dtype',target='tier',dtype='category',ordered=True,categories=categories)],
                      [dict(type='column_values_unchanged',target='tier')],except_=['tier'],same_rows=True)
        add(f'category_assignment_{n}',f'Make tier an ordered category in this order: {categories!r}. Retain all values and rows.',values,spec,
            code(prefix+expression+'\nreturn df'),code(prefix+"df['tier'] = "+expression+'\nreturn df'))
    spec=contract([dict(type='columns_added',target=['total']),dict(type='column_value_relation',target='total',fn='add',args=[dict(col='units_a'),dict(col='units_b')])],same_rows=True)
    good=code("df['total'] = df['units_a'] + df['units_b']\nreturn df")
    add('derived_preserve_inputs','Add total equal to units_a plus units_b. Preserve every original column and row.',
        dict(record=[1,2,3],units_a=[3,7,1],units_b=[4,2,9]),spec,
        code("df['total'] = df['units_a'] + df['units_b']\nreturn df.drop(columns=['units_a','units_b'])"),good)
    for n,digits,values in [(1,1,[1.05,2.55,6.45,None,-0.95]),(2,2,[21.075,31.275,None,-9.475,5.0])]:
        spec=contract([dict(type='column_value_relation',target='reading',fn='round',args=[dict(col='reading'),dict(lit=digits)])],except_=['reading'],same_rows=True)
        add(f'rounding_convention_{n}',f'Round reading to {digits} decimal places using pandas/NumPy rounding. Keep missing values and other columns unchanged.',
            dict(record=list(range(5)),reading=values),spec,
            code(f"df['reading'] = df['reading'].apply(lambda x: round(x,{digits}))\nreturn df"),
            code(f"df['reading'] = df['reading'].round({digits})\nreturn df"))
    spec=contract([dict(type='rows_removed_where',condition=dict(column='measurement',op='<',value=10))],
        [dict(type='rows_retained_where',condition={'any':[dict(column='measurement',op='>=',value=10),dict(column='measurement',op='is_null')]})])
    add('filter_keep_unknown','Remove measurements below 10. Keep rows with an unknown measurement, unchanged.',
        dict(record=[1,2,3,4,5],measurement=[None,5.,10.,12.,None]),spec,
        code("return df[df['measurement'].notna() & (df['measurement'] >= 10)]"),
        code("return df[df['measurement'].isna() | (df['measurement'] >= 10)].copy()"))
    spec=contract([dict(type='rows_removed_where',condition=dict(column='stamp',op='<',value='2025-04-01'))],
        [dict(type='rows_retained_where',condition=dict(column='stamp',op='>=',value='2025-04-01'))])
    add('filter_extra_helper','Keep rows whose stamp is on or after 2025-04-01. Preserve the original text timestamps and original columns only.',
        dict(record=[1,2,3],stamp=['2025-03-31 23:00:00','2025-04-01 00:00:00','2025-04-04 10:00:00']),spec,
        code("df['temporary_stamp'] = pd.to_datetime(df['stamp'])\nreturn df[df['temporary_stamp'] >= pd.to_datetime('2025-04-01')]"),
        code("return df[pd.to_datetime(df['stamp']) >= pd.to_datetime('2025-04-01')].copy()"))
    # Already-correct regression controls, including schema edits and a legitimate empty result.
    base=dict(record=[1,2,3],quantity=[2.,None,6.])
    good=code("df['quantity'] = df['quantity'].fillna(4)\nreturn df")
    spec=contract([dict(type='column_value_relation',target='quantity',where=dict(column='quantity',op='is_null'),fn='const',args=[dict(lit=4)])],
        [dict(type='column_values_unchanged',target='quantity',where=dict(column='quantity',op='not_null'))],except_=['quantity'],same_rows=True)
    add('control_fill','Fill missing quantity with 4, preserving all other cells.',base,spec,good,good,True)
    spec=contract([dict(type='columns_removed',target=['quantity']),dict(type='columns_added',target=['amount']),
                   dict(type='column_value_relation',target='amount',fn='identity',args=[dict(col='quantity')])],except_=['quantity'],same_rows=True)
    good=code("return df.rename(columns={'quantity':'amount'})")
    add('control_rename','Rename quantity to amount and preserve values.',base,spec,good,good,True)
    good=code("return df.drop(columns=['quantity'])")
    spec=contract([dict(type='columns_removed',target=['quantity'])],except_=['quantity'],same_rows=True)
    add('control_drop','Remove quantity. Keep every other column and all rows.',base,spec,good,good,True)
    good=code("df['double'] = df['quantity'] * 2\nreturn df")
    spec=contract([dict(type='columns_added',target=['double']),dict(type='column_value_relation',target='double',fn='mul',args=[dict(col='quantity'),dict(lit=2)])],same_rows=True)
    add('control_add','Add double equal to quantity multiplied by 2. Preserve the input columns.',base,spec,good,good,True)
    good=code("return df[df['quantity'] <= 0].copy()")
    spec=contract([dict(type='rows_removed_where',condition=dict(column='quantity',op='>',value=0))],
                  [dict(type='rows_retained_where',condition=dict(column='quantity',op='<=',value=0))])
    add('control_empty','Remove all rows with quantity greater than zero, retaining the original columns.',
        dict(record=[1,2],quantity=[2,3]),spec,good,good,True)
    return copy.deepcopy(cases)


def frame(case):
    return pd.DataFrame(case['data'])
