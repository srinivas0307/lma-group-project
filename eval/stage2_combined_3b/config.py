from eval.stage2_dual_3b.config import METHODS,MODELS,PARSER_SETTINGS,PROTOCOLS,THRESHOLDS
from eval.stage2_dual_3b.config import COMMON as ORIGINAL_COMMON
from eval.stage2_repair_dev.agent import VERSION
EXPERIMENT='stage2-combined-3b-fp16-v1'
SYSTEMS=('proposed',)
REPAIR_VERSION=VERSION+'+declared-schema1'
COMMON=dict(ORIGINAL_COMMON,repair_arm='combined',repair_version=REPAIR_VERSION,schema_guard='declared-output-columns-v1')
