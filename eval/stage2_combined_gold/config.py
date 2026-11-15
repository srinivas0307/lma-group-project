from eval.stage2_combined_3b.config import COMMON,MODELS,THRESHOLDS,REPAIR_VERSION
EXPERIMENT='stage2-combined-gold-3b-fp16-v1'
METHODS=('gold',)
SYSTEMS=('gold_contract',)
PROTOCOLS={'gold':'provided-gold-contract-v1'}
PARSER_SETTINGS={'gold':{'source':'task.gold_contract','model_calls':0}}
