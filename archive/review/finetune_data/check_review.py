import csv,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from igt.contracts import load_contract
from igt.sandbox import safe_run
from igt.verifier import verify
rows=list(csv.DictReader(open(ROOT/'finetune/data/review.csv')))
lookup={r['id']:r for r in map(json.loads,open(ROOT/'finetune/data/checked.jsonl'))}
results=[];frames={}
for i,r in enumerate(rows,1):
 ex=lookup[r['id']]
 assert r['request']==ex['request'] and json.loads(r['contract'])==ex['contract'] and r['reference_code']==ex['reference_code']
 f=ex['data_file']
 if f not in frames:frames[f]=pd.read_csv(ROOT/f)
 before=frames[f];c=load_contract(json.loads(r['contract']))
 run=safe_run(r['reference_code'],before,timeout_s=30,cpu_s=15,mem_gb=2)
 item=dict(row=i,id=r['id'],dataset=ex['source_dataset'],rows_before=len(before),run_status=run.status)
 if run.ran_ok:
  out=run.df_out;v,_=verify(c,before,out);v0,_=verify(c,before,before.copy())
  item.update(reference_verdict=v.status,reference_details=v.explain(),noop_verdict=v0.status,rows_after=len(out))
  if r['type']=='deduplication':
   ve,_=verify(c,before,before.iloc[:0].copy());vs,_=verify(c,before,out.iloc[:1].copy())
   item.update(empty_output_verdict=ve.status,one_row_output_verdict=vs.status)
  if r['id']=='student_col_0031':item['zero_denominators']=int((before.G1==0).sum())
  if r['id']=='bank_miss_0113':item['snapshot_median']=before.duration.median()
 results.append(item)
 if i%20==0:print('Checked',i,flush=True)
(ROOT/'tmp/review/check_results.json').write_text(json.dumps(results,indent=2))
print('Reference verdicts',Counter(r.get('reference_verdict') for r in results))
print('No-op verdicts',Counter(r.get('noop_verdict') for r in results))
for r in results:
 if r.get('empty_output_verdict')=='PASS' or r.get('reference_verdict')!='PASS' or r.get('rows_after')==0 or 'snapshot_median' in r or 'zero_denominators' in r:print(r)
