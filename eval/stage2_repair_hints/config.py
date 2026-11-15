from eval.stage2_combined_3b.config import REPAIR_VERSION as COMBINED_REPAIR_VERSION
from .agent import VERSION as HINTS_VERSION
from .recipes import VERSION as RECIPES_VERSION
from .lint import VERSION as LINT_VERSION
EXPERIMENT='stage2-repair-hints-3b-fp16-v1'
# Pipelines reuse the frozen runners: gold contracts (oracle diagnostic) and contract-check-v1 'checked' parses.
PIPELINES=('gold','selective','tools')
REPAIR_VERSION=COMBINED_REPAIR_VERSION+'+'+HINTS_VERSION
HINTS=dict(repair_hints=HINTS_VERSION,recipes=RECIPES_VERSION,lint=LINT_VERSION,repair_version=REPAIR_VERSION,
           change='repair user message adds contract-derived pandas recipes for failed predicates and contract-relevant code lints; nothing else changes')
