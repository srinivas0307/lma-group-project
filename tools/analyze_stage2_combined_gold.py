"""Strict collection plus CPU replay of the completed combined gold diagnostic."""
import collections,dataclasses,hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
import nbformat
import pandas as pd
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from igt.agents.code_generator_v2 import build_generator_messages
from eval.stage2_repair_dev.guard import guarded_verifier
from eval.equivalence import outputs_equal
from eval.stage2_combined_gold.runner import load_tasks
from eval.stage2_combined_gold.artifacts import write,digest
from finetune.selective_facts_scopefix.context import schema_entry
OUT=ROOT/'results/stage2_combined_gold_analysis'

def main():
    gold=ROOT/'stage2_combined_gold_3b.zip'
    if not gold.exists():gold=ROOT/'kaggle_outputs/stage2/combined_gold_v1/stage2_combined_gold_3b.zip'
    notebook=ROOT/'stage2-combined-gold.ipynb'
    if not notebook.exists():notebook=ROOT/'notebooks/executed/stage2/combined_gold_v1/stage2_combined_gold_3b_kaggle_executed.ipynb'
    predicted=[ROOT/f'kaggle_outputs/stage2/combined_3b_v1/stage2_combined_3b_{m}_proposed.zip' for m in ['selective','tools']]
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='combined_gold_collect_') as temp:
        with zipfile.ZipFile(ROOT/'notebooks/kaggle/stage2_combined_gold/stage2_combined_gold_source.zip') as z:z.extractall(temp)
        isolated=Path(temp)/'stage2_combined_gold_source';(isolated/'_stage2_combined_gold_manifest.json').write_bytes((isolated/'manifest.json').read_bytes())
        script=(ROOT/'tools/collect_stage2_combined_gold.py').read_text()
        # Only the external compatibility shim is added, outside the sealed source inventory.
        shim=isolated/'tools/collect_stage2_combined_gold.py';shim.parent.mkdir(exist_ok=True);shim.write_text(script)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(isolated),str(isolated/'src')]))
        dest=OUT/'comparison' if not (OUT/'comparison').exists() else Path(temp)/'comparison'
        subprocess.run([sys.executable,str(shim),'--gold',str(gold),'--predicted',*map(str,predicted),'--out',str(dest)],cwd=isolated,env=env,check=True)
        comparison=json.loads((dest/'summary.json').read_text())
        if dest!=OUT/'comparison':assert comparison==json.loads((OUT/'comparison/summary.json').read_text())
    with zipfile.ZipFile(gold) as z:payload={n:json.loads(z.read(n)) for n in z.namelist() if n.endswith('.json') and n!='bundle.json'}
    plan=payload['plan.json']
    for name,sha in plan['runtime'].items():assert digest(ROOT/name)==sha,name
    tasks=load_tasks();data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks}
    book=nbformat.read(notebook,as_version=4);template=nbformat.read(ROOT/'notebooks/kaggle/stage2_combined_gold/stage2_combined_gold_3b_kaggle.ipynb',as_version=4)
    audit=dict(zip_sha256=digest(gold),comparison=comparison,notebook=dict(source_matches=[c.source for c in book.cells]==[c.source for c in template.cells],errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(book.cells) for o in c.get('outputs',[]) if o.output_type=='error']),
               collector_fix='Compare saved gold contracts with the exact JSON serialization of task gold; numeric mapping keys become strings. No inference, score or sealed runtime files changed.',mismatches=[],cases={},limitations=['CPU replay uses local Python 3.12 rather than recorded Kaggle Python 3.13.16.','Gold changes generation and verification; this is not a repair-only ablation.','Previously inspected test tasks are diagnostic.'])
    for index,(_,task) in enumerate(tasks,1):
        tid=task['id'];before=data[task['data_file']];item=payload['tasks/'+tid+'.json'];row=item['record'];trace=item['trace'];contract=task['gold_contract']
        ref=safe_run(task['reference_code'],before,**Limits().sandbox());assert ref.ran_ok,(tid,ref.error)
        checker=guarded_verifier(lambda after:verify(contract,before,after)[0],contract,before)
        v=dataclasses.asdict(checker(ref.df_out))
        if v!=row['combined_reference_verdict']:audit['mismatches'].append([tid,'reference_verdict'])
        g=payload['generation/'+tid+'.json']
        msg=build_generator_messages(dict(request=task['request'],data_file=task['data_file']),{task['data_file']:schema_entry(before)},contract)
        if g['messages']!=msg or g['raw_output']!=trace['generator_reply']:audit['mismatches'].append([tid,'generation_input'])
        case=dict(record=row,reference_verdict=v,request=task['request'],initial=None,final=None)
        cache={}
        def replay(code):
            if code not in cache:
                result=safe_run(code,before,**Limits().sandbox())
                correct,reason=outputs_equal(ref.df_out,result.df_out) if result.ran_ok else (False,None)
                cache[code]=dict(ran_ok=result.ran_ok,error=result.error,correct=correct,equivalence=reason,verdict=dataclasses.asdict(checker(result.df_out)) if result.ran_ok else None)
            return cache[code]
        case['initial']=replay(trace['initial_code'])
        if case['initial']['correct']!=row['initial_correct']:audit['mismatches'].append([tid,'initial_correct'])
        if trace.get('final_code'):case['final']=replay(trace['final_code'])
        if row['outcome']=='ACCEPT':
            if not case['final'] or not case['final']['correct'] or case['final']['verdict']['status']!='PASS':audit['mismatches'].append([tid,'accepted_final'])
        case['failure_evidence']=[e for e in trace['events'] if e['event'] in ['repair_evidence','verify','stop','fresh_recovery_requested']]
        audit['cases'][tid]=case
        if index%10==0:write(OUT/'audit.partial.json',audit);print('CPU audited',index,'/100',flush=True)
    old=[json.loads(l) for l in (ROOT/'results/stage2_clean_v1/gold_contract/runs.jsonl').read_text().splitlines()]
    now={i:c['record'] for i,c in audit['cases'].items()};historical={r['task_id']:r for r in old}
    audit['historical_comparison']=dict(note='Unmatched diagnostic: historical 7B NF4 repair versus combined 3B FP16; repair prompt/guard/runtime may differ.',only_historical_correct=[i for i in now if historical[i]['final_correct'] and not now[i]['final_correct']],only_current_correct=[i for i in now if now[i]['final_correct'] and not historical[i]['final_correct']])
    old_traces={t['task_id']:t for t in map(json.loads,(ROOT/'results/stage2_clean_v1/gold_contract/traces.jsonl').read_text().splitlines())}
    audit['historical_comparison']['same_initial_code_count']=sum(old_traces[i]['initial_code']==payload['tasks/'+i+'.json']['trace']['initial_code'] for i in now)
    audit['historical_comparison']['initial_code_total']=len(now)
    audit['summary']=dict(initially_correct=sum(r['initial_correct'] is True for r in now.values()),outcomes=dict(collections.Counter(r['outcome'] for r in now.values())),families={family:dict(tasks=sum(c['record']['transformation_type']==family for c in audit['cases'].values()),correct=sum(c['record']['final_correct'] for c in audit['cases'].values() if c['record']['transformation_type']==family)) for family in sorted({r['transformation_type'] for r in now.values()})})
    write(OUT/'audit.json',audit)
    print('Mismatches',audit['mismatches'],flush=True)
    assert not audit['mismatches']
if __name__=='__main__':main()
