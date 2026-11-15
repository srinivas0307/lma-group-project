from eval.stage2_combined_3b.config import METHODS,MODELS,THRESHOLDS,REPAIR_VERSION,COMMON as COMBINED_COMMON
from eval.stage2_combined_3b.config import PROTOCOLS as COMBINED_PROTOCOLS,PARSER_SETTINGS as COMBINED_PARSER_SETTINGS
from .diagnostics import VERSION as CHECK_VERSION
EXPERIMENT='stage2-contract-check-3b-fp16-v1'
SYSTEMS=('proposed',)
# Arms: 'control' = frozen combined 3B parse; 'checked' = same first parse + diagnostics + at most one correction.
ARMS=('control','checked')
COMMON=dict(COMBINED_COMMON,contract_check=CHECK_VERSION,max_parser_corrections=1)
PROTOCOLS={m:p+'+'+CHECK_VERSION for m,p in COMBINED_PROTOCOLS.items()}
PARSER_SETTINGS={m:dict(s,check_version=CHECK_VERSION,max_corrections=1) for m,s in COMBINED_PARSER_SETTINGS.items()}
