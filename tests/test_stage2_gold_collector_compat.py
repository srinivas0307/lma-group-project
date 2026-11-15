"""A JSON key representation change must not invalidate an otherwise exact gold record."""
import copy,json
import pytest
from tools.collect_stage2_combined_gold import compatible_collector
from eval.stage2_combined_gold.runner import run,load_tasks
from eval.stage2_combined_gold.artifacts import export
from tests.test_stage2_combined_3b import factory


def test_numeric_mapping_key_roundtrip_and_real_tamper_rejection(tmp_path):
    backend,*_=factory('selective');out=run('gold',tmp_path/'gold',smoke=True,backend_factory=backend)
    read=compatible_collector().__globals__['read_bundle']
    tasks=copy.deepcopy(load_tasks('test'))
    tid='adult_col_001'
    # Use the actual numeric-key gold contract that exposed the collector bug.
    numeric=next(t['gold_contract'] for _,t in tasks if t['id']=='taxi_col_002')
    assert numeric!=json.loads(json.dumps(numeric))
    for _,t in tasks:
        if t['id']==tid:t['gold_contract']=numeric
    read.__globals__['load_tasks']=lambda split:tasks
    path=out/'contracts'/f'{tid}.json';item=json.loads(path.read_text());item['extracted']['contract']=copy.deepcopy(numeric);path.write_text(json.dumps(item));export(out,True)
    # The collector's authenticity comparison must use the serialized representation.
    read(out.with_suffix('.zip'),allow_scripted=True)
    item['extracted']['contract']['must_change']=[];path.write_text(json.dumps(item));export(out,True)
    with pytest.raises(ValueError,match='Gold contract differs'):read(out.with_suffix('.zip'),allow_scripted=True)
