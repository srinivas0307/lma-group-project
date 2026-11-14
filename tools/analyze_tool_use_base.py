"""Audit both unadapted tool parsers using saved transcripts and CPU semantic replay."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
import pandas as pd
from finetune.tool_use.config import MODELS,PROTOCOL
from finetune.tool_use.context import parser_messages
from finetune.tool_use.parser import parse_fresh,action_object
from finetune.tool_use.evaluate import development_tasks,summarize
from finetune.selective_facts.provenance import digest,write
from eval.contract_quality.harness import prepare,probe
from igt.agents.intent_parser import extract_contract
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from eval.equivalence import outputs_equal

OUT=ROOT/'results/tool_use_v2_analysis/base_comparison'

def load(size):
    path=ROOT/f'kaggle_outputs/tool_use_v2/tool_use_base_{size}.zip'
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:raise ValueError('Invalid archive')
        manifest=json.loads(z.read('bundle.json'))
        if not manifest['complete'] or set(manifest['sha256'])!=set(z.namelist())-{'bundle.json'}:raise ValueError('Invalid coverage')
        for n,h in manifest['sha256'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Changed bundle member: '+n)
        summary=json.loads(z.read('summary.json'))
        rows={r['task_id']:r for n in z.namelist() if n.startswith('tasks/') and (r:=json.loads(z.read(n)))}
    if not summary['complete'] or summary['smoke']:raise ValueError('Incomplete/smoke evaluation')
    prov=summary['provenance']
    if prov['base']!=MODELS[size] or prov['size']!=size or prov['adapter_hashes'] or prov['arm']!='base' or prov['protocol']!=PROTOCOL:raise ValueError('Unexpected model/protocol/adapter')
    for group in ['runtime','inputs','original_task_hashes']:
        for n,h in prov[group].items():
            if digest(ROOT/n)!=h:raise ValueError('Changed source/input: '+n)
    if digest(ROOT/'benchmark/contract_quality_v1/manifest.json')!=prov['synthetic_manifest']:raise ValueError('Changed synthetic suite')
    regenerated=summarize(list(rows.values()))
    for cohort,s in regenerated.items():
        group=[r for r in rows.values() if r['cohort']==cohort]
        s.update(tasks_with_tools=sum(r['tool_attempts']>0 for r in group),tool_attempts=sum(r['tool_attempts'] for r in group),tool_failures=sum(r['tool_failures'] for r in group))
    if regenerated!=summary['summary']:raise ValueError('Summary mismatch')
    return path,summary,rows

def main():
    tasks=development_tasks();expected={t['id'] for t,_,_ in tasks}
    template=nbformat.read(ROOT/'notebooks/kaggle/tool_use_v2/tool_use_base.ipynb',as_version=4)
    audit=dict(experiment='parser-tools-v2-base-comparison',models={},cases={},message_mismatches=[],trace_mismatches=[],score_mismatches=[],counter_mismatches=[],
      limits=['Open development data, not independent test accuracy.','Original tasks use only a wrong-no-op probe; synthetic tasks use targeted wrong outputs.','Exact per-call tokenizer counts are absent; token-budget stops are replayed as recorded rather than independently retokenized.','Recorded parser durations exclude setup/model loading, CPU semantic scoring and smoke.'])
    rows_by_size={}
    for size in MODELS:
        path,summary,rows=load(size);rows_by_size[size]=rows
        if set(rows)!=expected:raise ValueError('Incomplete task coverage')
        book=nbformat.read(ROOT/f'notebooks/executed/tool_use_v2/tool_use_base_{size}_executed.ipynb',as_version=4)
        if len(book.cells)!=len(template.cells):raise ValueError('Changed notebook cell structure')
        source_matches=all(a.source==b.source for i,(a,b) in enumerate(zip(book.cells,template.cells)) if i!=1)
        config=book.cells[1].source
        if f'MODEL_SIZE = "{size}"' not in config:raise ValueError('Notebook model size differs')
        errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(book.cells) for o in c.get('outputs',[]) if o.output_type=='error']
        stats=dict(tasks=len(rows),correct=sum(r['all_correct_accepted'] for r in rows.values()),wrong=sum(r['any_wrong_accepted'] for r in rows.values()),
          joint=sum(r['all_correct_accepted'] and not r['any_wrong_accepted'] for r in rows.values()),valid=sum(r['parse_status']=='ok' for r in rows.values()),
          statuses=dict(collections.Counter(r['parse_status'] for r in rows.values())),tokens=sum(r['total_tokens'] for r in rows.values()),seconds=sum(r['parser']['seconds'] for r in rows.values()),
          llm_calls=sum(r['parser']['llm_calls'] for r in rows.values()),tasks_with_tools=sum(r['tool_attempts']>0 for r in rows.values()),tool_attempts=sum(r['tool_attempts'] for r in rows.values()),tool_failures=sum(r['tool_failures'] for r in rows.values()),
          correct_tool_tasks=sum(r['tool_attempts']>0 and r['all_correct_accepted'] for r in rows.values()),tool_names=dict(collections.Counter(t['tool'] for r in rows.values() for t in r['parser']['tool_calls'])),
          tool_errors=dict(collections.Counter(t['observation']['error'] for r in rows.values() for t in r['parser']['tool_calls'] if not t['observation']['ok'])),
          wrong_and_correct=sum(r['all_correct_accepted'] and r['any_wrong_accepted'] for r in rows.values()),
          no_op_acceptances=sum(n['family']=='no_op' and n['status']=='PASS' for r in rows.values() for n in r.get('negative_statuses',[])))
        audit['models'][size]=dict(bundle_sha256=digest(path),notebook_errors=errors,notebook_source_matches_except_size=source_matches,summary=summary['summary'],stats=stats,by_pattern={})
    for index,(task,cohort,root) in enumerate(tasks,1):
        tid=task['id'];before=pd.read_csv(root/task['data_file']);prepared=None
        if cohort=='original':
            reference=safe_run(task['reference_code'],before,**Limits().sandbox())
            if not reference.ran_ok:raise ValueError(reference.error)
            if outputs_equal(reference.df_out,before)[0]:raise ValueError('No-op is not distinct')
        case=dict(request=task['request'],cohort=cohort,pattern=task.get('pattern',task.get('transformation_type')),models={})
        for size,rows in rows_by_size.items():
            record=rows[tid];saved=record['parser']
            if saved['messages'][:2]!=parser_messages(task,before):audit['message_mismatches'].append([size,tid])
            replies=[m['content'] for m in saved['messages'] if m['role']=='assistant']
            class Replay:
                model='transcript-replay'
                def __init__(self):self.index=0
                def prompt_tokens(self,messages):
                    return 24000 if self.index==len(replies) and saved['extracted']['status']=='parser_token_limit' else 0
                def complete(self,messages,*args,**kwargs):
                    reply=replies[self.index];self.index+=1;return reply,0,0
            replay=parse_fresh(task,before,Replay(),'p1-v2')
            for key in ['messages','raw_output','extracted','tool_calls','invalid_responses','stop_reason','llm_calls']:
                if replay[key]!=saved[key]:audit['trace_mismatches'].append([size,tid,key])
            if saved['tool_attempts']!=record['tool_attempts'] or saved['tool_failures']!=record['tool_failures'] or saved['tokens']!=record['total_tokens']:
                audit['counter_mismatches'].append([size,tid])
            extracted=saved['extracted'];detail=dict(status=record['parse_status'],correct=record['all_correct_accepted'],wrong=record['any_wrong_accepted'],tool_calls=saved['tool_calls'],contract=extracted.get('contract'),error=extracted.get('error'),raw_output=saved['raw_output'])
            if extracted['status']=='ok':
                if cohort=='synthetic':
                    if prepared is None:prepared=prepare(task,root)
                    score=probe(extracted['contract'],prepared)
                else:score=dict(all_correct_accepted=verify(extracted['contract'],before,reference.df_out)[0].status=='PASS',any_wrong_accepted=verify(extracted['contract'],before,before)[0].status=='PASS')
                for k,v in score.items():
                    if record.get(k)!=v:audit['score_mismatches'].append([size,tid,k])
                detail['semantic']=score
            elif record['all_correct_accepted'] or record['any_wrong_accepted']:audit['score_mismatches'].append([size,tid,'invalid_status'])
            case['models'][size]=detail
        audit['cases'][tid]=case
        if index%10==0:print(f'CPU replay {index}/100 tasks, two models',flush=True)
    for size in MODELS:
        for pattern in sorted({c['pattern'] for c in audit['cases'].values()}):
            group=[c['models'][size] for c in audit['cases'].values() if c['pattern']==pattern]
            audit['models'][size]['by_pattern'][pattern]=dict(tasks=len(group),correct=sum(r['correct'] for r in group),wrong=sum(r['wrong'] for r in group),joint=sum(r['correct'] and not r['wrong'] for r in group),valid=sum(r['status']=='ok' for r in group),with_tools=sum(bool(r['tool_calls']) for r in group))
    audit['paired']=dict(correct_1_5b_only=[tid for tid,c in audit['cases'].items() if c['models']['1_5b']['correct'] and not c['models']['3b']['correct']],correct_3b_only=[tid for tid,c in audit['cases'].items() if c['models']['3b']['correct'] and not c['models']['1_5b']['correct']])
    audit['trace_difference_notes']=[
      'warehouse_filter_3: Python 3.13 and local Python 3.12 produce different JSON syntax-error messages for the same invalid trailing comma; terminal status/counters match.',
      'titanic_conv_006: equal-frequency top_values order and the membership at the tenth-value boundary differ on local replay; returned counts/statistics/modes and terminal status match. The saved and local tied values were checked against input frequencies.'
    ]
    write(OUT/'audit.json',audit)
    print('Saved',OUT)
    print('CHECKS',{k:len(audit[k]) for k in ['message_mismatches','trace_mismatches','score_mismatches','counter_mismatches']})
    print('TOTALS',{size:audit['models'][size]['stats'] for size in MODELS})

if __name__=='__main__':main()
