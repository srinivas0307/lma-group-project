"""Baseline 2 repair: the improved repair loop with everything that needs a contract removed.

Kept from a1-v4+repair-evidence-reset1: per-candidate evidence (failures, schema delta, output availability),
labelled tool-observation provenance, and the one fresh-recovery request when an approach repeats.
Removed (each is derived from an intent contract, which baseline 2 does not have): contract text in the prompt,
contract-aligned cell examples, the declared-output-schema guard and the repair hints.
The failures come from the conventional checks; the system prompt says so instead of naming a contract.
"""
import time

from igt.agents import repair_agent as original
from igt.orchestrator.memory import code_hash
from eval.stage2_repair_dev import agent as frozen

VERSION = 'repair-no-contract-v1'
CONTRACT_SENTENCE = 'The diagnostic is derived from the read-only contract and is not an independent answer. '
REPLACEMENT = ('There is no intent contract: the failures come from generic data-quality checks on the input and the '
               'candidate output, not from the request, and they are not an independent answer. ')
SEMANTICS_PREFIX = ('Contract operator conventions:', 'pandas conventions:')


def diagnose(before, after, report):
    result = dict(source='input + submitted candidate + conventional checks; no contract, no reference answer',
                  output_available=after is not None, failures=report['failures'], notes=[])
    if after is None:
        result['notes'].append('This candidate produced no usable output. Inspect before, not last_output.')
        return result
    result['schema_delta'] = dict(missing=[c for c in before if c not in after],
                                  added=[c for c in after if c not in before],
                                  duplicate=list(after.columns[after.columns.duplicated()]))
    result['rows'] = dict(input=len(before), output=len(after))
    return result


class _ConventionalRun(frozen._EvidenceRun):
    def evaluate(self, code, attempt_no):
        self.output = None
        self.ctx.last_output = None
        self.ctx.last_code = None
        original._Run.evaluate(self, code, attempt_no)
        start = time.monotonic()
        self.diagnostic = diagnose(self.before, self.output, self.report)
        self.ev('repair_evidence', code_hash=code_hash(code), seconds=time.monotonic() - start,
                diagnostic=self.diagnostic)

    def full_messages(self, observations):
        messages = super().full_messages(observations)
        system = messages[0]['content']
        if CONTRACT_SENTENCE not in system or SEMANTICS_PREFIX[0] not in system:
            raise RuntimeError('Frozen repair prompt changed; no-contract wording cannot be applied')
        system = system.replace(CONTRACT_SENTENCE, REPLACEMENT).replace(*SEMANTICS_PREFIX)
        messages[0] = dict(role='system', content=system)
        return messages


def run_repair(task, initial_code, before, verify_fn, client, config=None, contract_text=None, schema_text=None):
    if contract_text:
        raise ValueError('Baseline 2 repair runs without a contract')
    cfg = config or original.AgentConfig()
    if cfg.mode != 'full':
        raise ValueError('Baseline 2 uses full repair mode')
    if client.token_budget is None:
        client.token_budget = cfg.limits.token_budget
    return _ConventionalRun(task, initial_code, before, verify_fn, client, cfg, None, schema_text, contract=None).run()
