from eval.stage2_repair_dev.agent import VERSION as EVIDENCE_VERSION
from eval.stage2_repair_hints.config import REPAIR_VERSION as HINTS_REPAIR_VERSION, HINTS
from .nocontract import VERSION as NO_CONTRACT_VERSION

EXPERIMENT = 'stage2-baselines-hints-3b-fp16-v1'
SYSTEMS = ('baseline2_repair', 'baseline3_repair', 'no_contract_to_generator')

# One notebook per run. `method` selects the parser adapter / protocol; baseline 2 has no parser and baseline 3 uses the
# prompted base 3B model (p1-v3), so for those two the method only labels the run (and sets the correction context).
RUNS = {
    'baseline2': dict(method='selective', preset='baseline2_repair', adapter=False, arm='control'),
    'baseline3': dict(method='selective', preset='baseline3_repair', adapter=False, arm='checked'),
    'no_contract_selective': dict(method='selective', preset='no_contract_to_generator', adapter=True, arm='checked'),
    'no_contract_tools': dict(method='tools', preset='no_contract_to_generator', adapter=True, arm='checked'),
}

REPAIR_VERSIONS = {
    'baseline2_repair': EVIDENCE_VERSION + '+' + NO_CONTRACT_VERSION,
    'baseline3_repair': HINTS_REPAIR_VERSION,
    'no_contract_to_generator': HINTS_REPAIR_VERSION,
}

# Baseline 2 thresholds exactly as in the reported six-system run (clean_v1 used eval/runner_stage2.make_checker,
# which reads results/stage1/baseline2/thresholds.json["tuned"]["thresholds"]; dev-tuned in Stage 1).
CONVENTIONAL_THRESHOLDS = {
    'max_dropped_columns': 1, 'max_row_ratio': 'inf', 'min_row_ratio': 0, 'max_index_not_subset': 0,
    'max_null_increase': 'inf', 'max_new_column_null_frac': 'inf', 'dtype_mode': 'off',
    'max_lost_unique_columns': 'inf', 'max_duplicate_rows_increase': 'inf', 'max_out_of_range_frac': 'inf',
    'max_mean_shift_z': 0.1, 'max_category_tvd': 'inf', 'max_new_value_frac': 'inf'}
CONVENTIONAL = 'stage1-dev-tuned (results/stage1/baseline2/thresholds.json, same as the six-system clean_v1 run)'

BASELINES = dict(
    systems=dict(
        baseline2_repair='no parser/contract; conventional checks; improved repair loop without contract-derived parts',
        baseline3_repair='prompted base Qwen2.5-3B parser (p1-v3, no adapter) + contract-check-v1; request-only generator; repair with hints',
        no_contract_to_generator='fine-tuned 1.5B parser + contract-check-v1; request-only generator; repair with hints'),
    generator_contract='proposed and gold only (six-system plan)',
    conventional_thresholds=CONVENTIONAL,
    repair_versions=REPAIR_VERSIONS,
    hints=HINTS,
)
