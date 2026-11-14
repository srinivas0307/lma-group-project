"""Validate and CPU-replay the four-arm parser diagnostic without model calls."""
import collections
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
from eval.stage2_independent_v2.diagnostic import ARMS,messages_for,evaluate,summarize
from eval.stage2_independent_v2.runner import load_tasks,write_json
from igt.agents.intent_parser import extract_contract
from igt.orchestrator import Limits
from igt.sandbox import safe_run


def members(source):
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:raise ValueError('CRC failure')
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate members')
        files={n:z.read(n) for n in z.namelist()}
    seal=json.loads(files.pop('bundle.json'))
    if not seal['complete'] or seal['sha256']!={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}:
        raise ValueError('Incomplete or changed bundle')
    return files


def main():
    path=ROOT/'kaggle_outputs/stage2/independent_v2/stage2_parser_diagnostic.zip';files=members(path)
    with tempfile.TemporaryDirectory(prefix='igt_parser_diagnostic_') as temp:
        for arm in ARMS: (Path(temp)/(arm+'.zip')).write_bytes(files[arm+'.zip'])
        summarize(temp)  # Checks method, current source/input hashes, full dev coverage and controlled settings.
        regenerated=json.loads((Path(temp)/'summary.json').read_text())
    supplied=json.loads(files['summary.json'])
    if supplied!=regenerated:raise ValueError('Supplied summary differs from validated arm data')
    expected_adapter=json.loads((ROOT/'eval/stage2_independent_v2/adapter.json').read_text())['sha256']
    if any(p['adapter_hashes']!=expected_adapter for p in supplied['plans']):raise ValueError('Wrong adapter provenance')
    notebook=json.loads((ROOT/'notebooks/executed/stage2/independent_v2/stage2_parser_diagnostic_executed.ipynb').read_text())
    template=json.loads((ROOT/'notebooks/kaggle/stage2_independent_v2/stage2_parser_diagnostic.ipynb').read_text())
    src=lambda c:''.join(c.get('source',[]))
    errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(notebook['cells']) for o in c.get('outputs',[]) if o.get('output_type')=='error']
    audit=dict(bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),strict_validation=True,summary=supplied['summary'],
               paired=supplied['paired_reference_pass'],notebook_errors=errors,
               notebook_source_matches=len(notebook['cells'])==len(template['cells']) and all(src(a)==src(b) for a,b in zip(notebook['cells'],template['cells'])),
               input_mismatches=[],extraction_mismatches=[],cpu_score_mismatches=[],cases={},by_type={},plans=supplied['plans'])
    rows={}
    for arm in ARMS:
        nested=members(io.BytesIO(files[arm+'.zip']))
        rows[arm]={r['task_id']:r for n,b in nested.items() if n.startswith('tasks/') and (r:=json.loads(b))}
    tasks=load_tasks('dev');data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks}
    for _,task in tasks:
        tid=task['id'];before=data[task['data_file']]
        ref=safe_run(task['reference_code'],before,**Limits().sandbox())
        if not ref.ran_ok:raise RuntimeError(ref.error)
        case=dict(request=task['request'],transformation_type=task['transformation_type'],arms={})
        for arm in ARMS:
            row=rows[arm][tid]
            if messages_for(task,before,arm.split('_')[0])!=row['messages']:audit['input_mismatches'].append([arm,tid])
            extracted=extract_contract(row['raw_output'])
            if extracted!=row['extracted']:audit['extraction_mismatches'].append([arm,tid])
            score=evaluate(extracted,before,ref.df_out)
            for key,value in score.items():
                if value!=row[key]:audit['cpu_score_mismatches'].append([arm,tid,key])
            case['arms'][arm]=dict(extracted=extracted,**score,tokens=row['tokens'],seconds=row['seconds'])
        audit['cases'][tid]=case
    for kind in sorted({t['transformation_type'] for _,t in tasks}):
        subset=[t['id'] for _,t in tasks if t['transformation_type']==kind]
        audit['by_type'][kind]={arm:sum((rows[arm][tid]['reference_verdict'] or {}).get('status')=='PASS' for tid in subset) for arm in ARMS}
    audit['totals']={arm:dict(tokens=sum(r['tokens'] for r in rows[arm].values()),inference_seconds=sum(r['seconds'] for r in rows[arm].values())) for arm in ARMS}
    out=ROOT/'results/stage2_independent_analysis/parser_diagnostic_v1'
    write_json(out/'audit.json',audit)
    lines=['# Parser diagnostic analysis','',
        'All four arms completed all 20 development tasks. Outer and nested ZIP seals, source/input hashes, adapter metadata, task coverage and supplied summaries validate. The executed notebook is the diagnostic despite its filename.',
        f'Notebook exception outputs: {len(errors)}; source matches template: {audit["notebook_source_matches"]}. Reconstructed-message mismatches: {len(audit["input_mismatches"])}; extraction mismatches: {len(audit["extraction_mismatches"])}; CPU score mismatches: {len(audit["cpu_score_mismatches"])}.','',
        '## Results','']
    for arm,s in audit['summary'].items():
        lines.append(f'- {arm}: {s["reference_pass"]}/20 reference passes; {s["valid_contracts"]}/20 schema-valid contracts; {s["wrong_noop_accepted"]}/20 wrong no-ops accepted; {s["tokens"]} tokens.')
    lines += ['', 'Reference pass means the generated contract accepts a correct reference output. This is not end-to-end Stage 2 accuracy or a complete semantic correctness score. There are 20 distinct tasks, evaluated four times.', '',
        '## Interpretation','',
        'At fixed NF4 loading, adding the profile loses nine reference passes and gains one (16 to 8). At fixed FP16 loading it also loses nine and gains one (15 to 7). NF4 gives only one net additional pass within either input format. Quantization affects individual outputs, but restoring NF4 does not resolve the profile regression in this sample.',
        'The profiles are derived from the input data and reconstructed exactly. Incorrect behavior is visible in generated contracts: wrong keys, literal/name changes, invalid functions and contradictory preservation. Input-format/distraction effects are a plausible explanation; this experiment does not isolate profile length, placement, contents or training mismatch separately.', '',
        '## Concrete failures','',
        '- `taxi_col_006`: request renames Airport_fee to airport_fee; profile/NF4 instead demands filling the column with zero and increasing nulls. Original/NF4 accepts the correct rename.',
        '- `taxi_dedup_006`: request specifies VendorID and pickup time, keeping last. Profile/NF4 uses all columns. Original/NF4 passes.',
        '- `titanic_col_006`: request names passenger_class; profile/NF4 invents pclass and also preserves the removed original column. Original/NF4 passes.',
        '- `airbnb_col_006`: profile/NF4 emits unsupported fn=rename and fails schema validation. Original/NF4 passes.',
        '- `taxi_miss_006`: request marks zero trip distances missing; profile/FP16 accepts the wrong unchanged input. Profile/NF4 also fails the correct reference.', '',
        '## What to do next','',
        'Do not treat the current full-profile v2 or untrained tool protocol as a validated improvement. Keep original-input NF4 as the current parser development control. Avoid spending on all six profile-based full runs before input-format work.',
        'Use the existing 80-task synthetic development suite (after review) to test a compact, request-relevant information format against that control, including both tasks requiring statistics and tasks that do not. Supply only relevant columns/statistics and full precision; do not inject gold or reference-derived hints. If performance still regresses, train on the selected format with explicit scope/key/preservation examples and select using semantic tests.',
        'These 20 development requests contain no median, mean, mode or group-statistic requests. Consequently this run demonstrates collateral damage on ordinary tasks, but does not test whether supplying genuinely unavailable statistics helps those harder tasks. A follow-up needs that coverage.',
        'Zero wrong no-op acceptances in three arms does not establish strong safety: unchanged input is only one wrong-output probe. Use the broader wrong-candidate probes before selecting a parser. The synthetic suite remains draft, and sharing templates limits independent generalization claims.', '',
        'The pip dependency-conflict messages are warnings printed in stream output; they did not stop these jobs. Torchao removal and the small LoRA probe succeeded. No GPU inference, retraining, implementation changes or original-result rewriting were performed by this audit.', '',
        'Reproduce: `.venv/bin/python tools/analyze_parser_diagnostic.py`. Detailed cases, paired gains/losses, family counts, cost and provenance are in `audit.json`.']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    print('AUDIT:',out)
    print('CHECKS:',{k:audit[k] for k in ['notebook_source_matches','input_mismatches','extraction_mismatches','cpu_score_mismatches']})
    print('FAMILY:',json.dumps(audit['by_type']))

if __name__=='__main__':main()
