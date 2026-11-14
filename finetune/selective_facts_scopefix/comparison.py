"""Historical paired comparison after inference, never input to model generation."""
import collections
import json
from pathlib import Path
from finetune.selective_facts.evaluate import summarize,eligible

def compare(rows,smoke=False):
    prior=json.loads((Path(__file__).parent/'prior_evaluation.json').read_text())
    ids={r['task_id'] for r in rows}
    if len(ids)!=len(rows):raise ValueError('Duplicate current task records')
    for r in rows:
        previous=prior['sf1']['records'].get(r['task_id'])
        if previous is None or previous['request']!=r['request'] or previous['cohort']!=r['cohort']:raise ValueError('Historical task identity differs')
    if not smoke and ids!=set(prior['sf1']['records']):raise ValueError('Full comparison requires the same 100 development tasks')
    result=dict(historical_note=prior['note'],current=summarize(rows),prior={arm:prior[arm]['summary'] for arm in ['original','sf1']},paired={})
    for arm in ['original','sf1']:
        result['paired'][arm]=dict(gained=[r['task_id'] for r in rows if r['all_correct_accepted'] and not prior[arm]['records'][r['task_id']]['all_correct_accepted']],
            lost=[r['task_id'] for r in rows if not r['all_correct_accepted'] and prior[arm]['records'][r['task_id']]['all_correct_accepted']])
    result['routing']=dict(fact_unavailable=sum(r['parse_status']=='fact_unavailable' for r in rows),tasks_with_facts=sum(bool(r['facts']) for r in rows),
         previously_blocked={r['task_id']:dict(parse_status=r['parse_status'],facts_delivered=bool(r['facts']),correct=r['all_correct_accepted'],wrong=r['any_wrong_accepted']) for r in rows if prior['sf1']['records'][r['task_id']]['parse_status']=='fact_unavailable'})
    result['passed_original_screen']=False if smoke else eligible(result['current'],prior['original']['summary'])
    result['no_cohort_regression_vs_sf1']=False if smoke else all(result['current'][c]['correct_accepted']>=prior['sf1']['summary'][c]['correct_accepted'] and result['current'][c]['wrong_accepted']<=prior['sf1']['summary'][c]['wrong_accepted'] for c in result['current'])
    result['smoke']=smoke
    return result
