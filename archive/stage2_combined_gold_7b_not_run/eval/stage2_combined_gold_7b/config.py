from copy import deepcopy
from eval.stage2_combined_gold.config import METHODS,SYSTEMS,PROTOCOLS,PARSER_SETTINGS,THRESHOLDS,REPAIR_VERSION
from eval.stage2_combined_gold.config import MODELS as ORIGINAL_MODELS,COMMON as ORIGINAL_COMMON
EXPERIMENT='stage2-combined-gold-7b-nf4-v1'
MODELS=deepcopy(ORIGINAL_MODELS)
MODELS['agent']=dict(id='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',four_bit=True)
COMMON=dict(ORIGINAL_COMMON,repair_model_policy='7b-nf4-repair-with-3b-fp16-generator-v1')
