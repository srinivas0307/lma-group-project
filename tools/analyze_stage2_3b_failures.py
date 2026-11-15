import json,zipfile,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]; audit=json.loads((root/'results/stage2_dual_3b_analysis/audit.json').read_text())
notes={
'adult_col_005':'Map encodes an "Other" key as if it were a default. The contract does not represent the requested fallback for every unmatched marital-status value.',
'adult_conv_003':'Selective: invalid JSON contains an ellipsis in the category list. Tools: invented education labels and a mismatched mapping from education-num. Needs input-derived category/rank pairs and explicit ordered-category semantics.',
'adult_dedup_003':'Usable contract. Initial merge raises KeyError; repair inspects nonexistent last_output and keeps complicating the merge/helper-column approach. A fresh formulation should preserve original records and deduplicate on fnlwgt.',
'adult_filt_004':'Both contracts compare the categorical education strings with numeric thresholds; verification is uncertain. The request names two string categories.',
'adult_filt_005':'Selective binds self-employment to education=Private instead of workclass categories. Tools emits a predicate name as an unavailable tool and never produces a usable contract.',
'adult_miss_005':'Initially correct code is withheld. Filling null cells conflicts with preserving null count; tools additionally preserves the edited cell values. Resolve the contract, not the code alone.',
'airbnb_conv_005':'Usable contract. set_categories result is discarded; ordered=False remains. The repair agent repeats the same code despite an explicit ordered-flag failure.',
'airbnb_dedup_001':'Initially correct code is withheld because the contract deduplicates all columns instead of listing name.',
'airbnb_dedup_002':'Initially correct code is withheld; contract uses nonexistent listing_name and the wrong host identity field. Need exact input schema and request-to-key binding.',
'airbnb_dedup_003':'Selective withholds initially correct code because the contract uses all columns rather than host_id. Tools succeeds.',
'airbnb_dedup_004':'Selective omits host_id from the composite key. Tools has a reference-compatible contract but filters to groups of size one before keeping last, deleting all repeated groups. Adding a sort does not restore the lost groups.',
'airbnb_filt_002':'Both incorrectly accept an empty output. Contract selects neighbourhood instead of the borough-valued neighbourhood_group; correct reference has 927 rows. Relevant categorical-value evidence is missing.',
'taxi_col_004':'Usable contract under the current NumPy rounding convention. Scalar Python round(x,1) differs from pandas/NumPy round on 42 input rows; retries keep using scalar round.',
'taxi_col_005':'Selective contract attempts to preserve values of a newly added column that does not exist in the input. Tools succeeds. Zero-distance behavior must also remain represented.',
'taxi_conv_002':'Tools withholds initially correct code: it requires timestamp conversion while preserving the original string values under the current verifier semantics. Selective succeeds.',
'taxi_dedup_005':'Usable contract. Sorting by pickup timestamp before deduplication changes which original row is first; 99 key groups retain the wrong record. Repeated proposals preserve this error.',
'taxi_filt_004':'Both incorrectly accept the result with an extra pickup_datetime helper column. Existing columns_unchanged does not forbid extra output columns; both gold and predicted contracts pass the wrong output.',
'taxi_miss_003':'Tools withholds initially correct code: replacing zero with one conflicts with a required increase in null count while the remaining cells are preserved. Selective succeeds.',
'taxi_miss_004':'Tools withholds initially correct code: replacing 99 with one conflicts with a required increase in null count. Selective succeeds.',
'taxi_miss_005':'Selective router mistakes ordinary "mean" (signify) for a statistic and stops before parsing. Tools succeeds.',
'titanic_col_001':'Usable contract. Code adds num_relatives but removes SibSp/Parch (and sometimes more columns); repeated repair fails to preserve inputs.',
'titanic_col_003':'Usable contract under current NumPy rounding convention. Scalar Python round(x,2) differs from pandas/NumPy round on 29 rows; repeated proposals preserve the discrepancy.',
'titanic_col_005':'Selective encodes a threshold with a literal map key "<1"; tools multiplies Age by one instead of setting ages below one to one. Tools initially has correct code and loses it.',
'titanic_conv_002':'Selective emits invalid map arguments. Tools emits the wrong category order [1,3,2] and a spurious remapping. Request needs category order [3,2,1] with original values retained.',
'titanic_conv_003':'Both withhold initially correct code. Contract mapping uses string keys True/False rather than the input values 1/0 under current typed mapping semantics.',
'titanic_dedup_002':'Usable contract. Sorting Ticket before keeping last changes original record order/tie selection; repair repeats the initial program.',
'titanic_dedup_004':'Usable contract. Invalid groupby keep argument is followed by groupby.first(), which can aggregate first nonmissing values instead of retaining one complete original record.',
'titanic_filt_003':'Usable contract. Code drops unknown ages although request preserves them; the final failure report identifies 177 missing rows that should stay.',
'titanic_filt_005':'Selective emits invalid null operator; both contracts invent Cabin/Fare conditions unrelated to travelling alone. Tools initially has correct code but is withheld.',
'titanic_miss_005':'Selective cannot compute grouped statistics in its current facts router. Tools queries Pclass rather than Age grouped by Pclass and fills with 3. Existing group_profile can retrieve the required values but was not selected.'}
records={}; summary={}
for method in ['selective','tools']:
 with zipfile.ZipFile(root/f'kaggle_outputs/stage2/dual_parser_3b_fp16_v1/stage2_dual_3b_{method}_proposed.zip') as z:
  payload={n:json.loads(z.read(n)) for n in z.namelist() if n.endswith('.json')}
 bytask={r['record']['task_id']:r for n,r in payload.items() if n.startswith('tasks/')}
 parsed={r['task_id']:r for n,r in payload.items() if n.startswith('contracts/')}
 counts=collections.Counter()
 for tid,c in audit['cases'].items():
  d=c['methods'][method];r=d['record']
  if r['final_correct']:continue
  ref=(d['reference_verdict']or{}).get('status');category='incorrect_acceptance' if r['outcome']=='ACCEPT' else 'parser_stop' if r['stage_ended']=='parse' else 'contract_reference_mismatch' if ref!='PASS' else 'repair_with_reference_compatible_contract'
  counts[category]+=1
  events=(bytask[tid]['trace']or{}).get('events',[])
  records.setdefault(tid,dict(request=c['request'],diagnosis=notes[tid],methods={} ))['methods'][method]=dict(category=category,outcome=r['outcome'],initial_correct=r['initial_correct'],reference_status=ref,escalate_reason=r.get('escalate_reason'),repeated_code=any(e['event']=='repeated_code' for e in events),parser_status=parsed[tid]['extracted']['status'],parser_error=parsed[tid]['extracted'].get('error'),contract=d['contract'])
 summary[method]=dict(categories=dict(counts),withheld_initially_correct=sum(d['methods'][method]['record']['initial_correct'] is True and d['methods'][method]['record']['outcome']!='ACCEPT' for d in audit['cases'].values()),withheld_with_repeated_code=sum(v['methods'].get(method,{}).get('repeated_code',False) and v['methods'][method]['outcome']!='ACCEPT' for v in records.values()))
assert len(records)==30 and set(records)==set(notes)
out=root/'results/stage2_dual_3b_analysis/failure_review';out.mkdir(exist_ok=True)
(out/'inventory.json').write_text(json.dumps(dict(source='Existing CPU-verified 3B audit and sealed uploaded traces; diagnostic inspection, no new GPU inference',summary=summary,unique_tasks=len(records),cases=records),indent=2)+'\n')
lines=['# 3B incorrect-acceptance and withheld-case inventory','', 'All 54 unsuccessful method/task outcomes are covered below: 27 per method, across 30 distinct tasks. An omitted method succeeded on that task. Reference compatibility is audit information only and must not be used to route ordinary inference.','']
for tid,c in records.items():
 lines += ['## '+tid,'',c['request'].strip(),'',c['diagnosis'],'']
 for m,d in c['methods'].items():lines += [f"- {m}: {d['outcome']}; category={d['category']}; initially correct={d['initial_correct']}; reference={d['reference_status']}; repeated code={d['repeated_code']}."]
 lines += ['']
(out/'INVENTORY.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
