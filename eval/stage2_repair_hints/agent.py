"""Combined 3B repair (a1-v4+repair-evidence-reset1, schema guard) plus contract-derived hints.

Only the repair user message changes: after the existing evidence it adds
  * pandas recipes for the predicates that failed (all must_change recipes if the candidate crashed or
    produced no predicate failure), derived from the predicted contract and input columns only;
  * lints: contract-relevant pitfalls found statically in the submitted code.
Model, decoding, tools, budgets, attempts, fresh recovery and verifier are unchanged.
"""
import json

from igt.agents.repair_agent import AgentConfig
from eval.stage2_repair_dev import agent as frozen
from eval.stage2_repair_dev.guard import guarded_verifier
from . import lint, recipes

VERSION = 'repair-hints-v1'
HEADER = ('Contract-derived implementation notes (computed from the read-only contract and the input columns; '
          'not a reference answer). `df` is the unmodified input and `out = df.copy()` the result being built:')


def _failed_ids(report_or_ids):
    if isinstance(report_or_ids, (list, tuple)):
        return list(report_or_ids)
    ids = []
    for f in report_or_ids.get('failures', []):
        ids.append(f.get('predicate_id') or (f.get('check') if f.get('kind') == 'check' else 'execution'))
    return ids


def hint_block(contract, columns, code, failed):
    """The text appended to the repair prompt, or '' when nothing applies."""
    failed = _failed_ids(failed)
    table = recipes.recipes(contract, columns)
    chosen = {pid: r for pid, r in table.items() if pid in failed}
    if not chosen or 'execution' in failed:
        chosen.update({pid: r for pid, r in table.items() if r['direction'] == 'must_change'})
    parts = []
    text = recipes.render(chosen)
    if text:
        parts.append(HEADER + '\n' + text)
    if 'declared_output_schema' in failed:
        from eval.stage2_repair_dev.guard import schema_failures  # noqa: F401  (same declared column rule)
        parsed = recipes.load_contract(contract)
        added = [c for d, _, p in parsed.predicates(('must_change',)) if p.type == 'columns_added' for c in p.target]
        removed = {c for d, _, p in parsed.predicates(('must_change',)) if p.type == 'columns_removed' for c in p.target}
        expected = [c for c in columns if c not in removed] + [c for c in added if c not in columns]
        parts.append('Output columns must be exactly: ' + json.dumps(expected) + '. Drop any helper column before returning.')
    found = lint.hints(code, contract, columns) if code else []
    if found:
        parts.append('Problems found in the submitted code:\n' + lint.render(found))
    return '\n\n'.join(parts)


class _HintRun(frozen._EvidenceRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_columns = list(self.before.columns)

    def full_messages(self, observations):
        messages = super().full_messages(observations)
        try:
            block = hint_block(self.contract, self.hint_columns, self.code, self.report)
        except Exception as exc:  # hints are advisory; never break the frozen loop
            block = ''
            self.ev('repair_hints_error', error=type(exc).__name__ + ': ' + str(exc)[:200])
        if block:
            self.ev('repair_hints', chars=len(block), lints=[h['code'] for h in lint.hints(self.code, self.contract, self.hint_columns)]
                    if self.code else [])
            user = messages[1]['content']
            marker = '\n\n' + frozen.original.NEXT_ACTION
            head = user[:-len(marker)] if user.endswith(marker) else user
            messages[1] = dict(role='user', content=head + '\n\n' + block + marker)
        return messages


def run_repair(task, initial_code, before, verify_fn, client, config=None, contract_text=None, schema_text=None):
    if not contract_text:
        raise ValueError('Repair hints require an explicit intent contract')
    contract = json.loads(contract_text)
    cfg = config or AgentConfig()
    if cfg.mode != 'full':
        raise ValueError('Repair hints require full repair mode')
    verify_fn = guarded_verifier(verify_fn, contract, before)
    if client.token_budget is None:
        client.token_budget = cfg.limits.token_budget
    return _HintRun(task, initial_code, before, verify_fn, client, cfg, contract_text, schema_text,
                    contract=contract).run()
