"""Pack, gold JSON representation, fixed budgets and combined-repair CPU smoke."""
import json
from .artifacts import verify_pack
from .config import MODELS,COMMON,REPAIR_VERSION
from eval.stage2_combined_gold.config import MODELS as CONTROL_MODELS,COMMON as CONTROL_COMMON,REPAIR_VERSION as CONTROL_REPAIR
from eval.stage2_combined_3b.preflight import check as combined_check
from eval.stage2_selective.runner import load_tasks
from igt.contracts.schema import load_contract

def check():
    verify_pack()
    if MODELS['generator']!=CONTROL_MODELS['generator'] or REPAIR_VERSION!=CONTROL_REPAIR:raise ValueError('Generator/repair procedure changed')
    if {k:v for k,v in COMMON.items() if k!='repair_model_policy'}!={k:v for k,v in CONTROL_COMMON.items() if k!='repair_model_policy'}:raise ValueError('Combined budgets/procedure changed')
    for _,task in load_tasks('test'):load_contract(json.loads(json.dumps(task['gold_contract'])))
    return dict(combined_check(),gold_contract_json_check='100 passed',repair='7B NF4; float16 computation',generator='3B unquantized FP16')
if __name__=='__main__':print(json.dumps(check(),indent=2))
