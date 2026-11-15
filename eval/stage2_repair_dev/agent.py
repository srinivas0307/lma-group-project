"""a1-v4-compatible entry point with independently selectable evidence and schema guard."""
import json
import time
from igt.agents import repair_agent as original
from igt.orchestrator.memory import code_hash
from .evidence import diagnose, render, SEMANTICS
from .guard import guarded_verifier

VERSION = 'a1-v4+repair-evidence-reset1'
ARMS = {'control': (False,False), 'evidence': (True,False),
        'schema_guard': (False,True), 'combined': (True,True)}


class _ObservedRegistry:
    def __init__(self, base, run):
        self.base, self.run = base, run
    def __getattr__(self, name):
        return getattr(self.base,name)
    def call(self, call):
        if call.name == 'run_transformation':
            self.run.ctx.last_output = None
            self.run.ctx.last_code = None
        result = self.base.call(call)
        code = self.run.ctx.last_code
        observation = dict(tool=call.name, args=call.args, output_code_hash=code_hash(code) if code else None,
                           observation=result.to_observation())
        self.run.tool_history.append(observation)
        self.run.ev('tool_observation_provenance', tool=call.name,
                    output_code_hash=observation['output_code_hash'], output_available=self.run.ctx.last_output is not None)
        return result


class _EvidenceRun(original._Run):
    def __init__(self,*args,contract,**kwargs):
        super().__init__(*args,**kwargs)
        self.contract = contract
        self.fresh_used = False
        self.fresh_active = False
        self.tool_history = []
        self.diagnostic = None
        self.reg = _ObservedRegistry(self.reg,self)

    def evaluate(self,code,attempt_no):
        self.output = None
        self.ctx.last_output = None
        self.ctx.last_code = None
        super().evaluate(code,attempt_no)
        start = time.monotonic()
        self.diagnostic = diagnose(self.contract,self.before,self.output,self.report)
        self.ev('repair_evidence', code_hash=code_hash(code), seconds=time.monotonic()-start,
                diagnostic=self.diagnostic)

    def signature(self):
        return tuple(sorted((str(f.get('predicate_id') or f.get('check') or f.get('kind')),
                             str(f.get('status','')), json.dumps(f.get('observed',f.get('detail')),sort_keys=True,default=str)) for f in self.report['failures']))

    def propose(self,code,attempt_tool_calls):
        previous = self.signature()
        self.fresh_active = False
        outcome = super().propose(code,attempt_tool_calls)
        if (not self.fresh_used and self.n_fix < self.L.max_attempts and
                self.sm.state in ('FAILED','REPAIR') and
                (outcome == 'repeated' or self.signature() == previous)):
            self.fresh_used = self.fresh_active = True
            self.ev('fresh_recovery_requested', remaining_proposals=self.L.max_attempts-self.n_fix)
        return outcome

    def full_messages(self,observations):
        base = super().full_messages(observations)
        system = base[0]['content'] + '\n\n'+SEMANTICS
        system += ('\nUse the concrete failure evidence. Correct the cause and retain unrelated cells. '
                   'The diagnostic is derived from the read-only contract and is not an independent answer. '
                   'Do not invent output observations when a candidate crashed.')
        history = [dict(attempt=a.number, code_hash=a.code_hash, status=a.status,
                        report=a.report_text[:700]) for a in self.mem.attempts[-3:]]
        user = (self.task_text()+'\n\nLatest submitted code:\n```python\n'+self.code+'\n```\n'
                +'Latest candidate evidence:\n'+render(self.diagnostic)+'\n'
                +'Attempt summary:\n'+json.dumps(history)+'\n')
        if self.fresh_active:
            user += ('\nFRESH RECOVERY (once, within the existing attempt budget): the previous approach repeated '
                     'or left the same checks failing. Write the simplest fresh implementation from the request '
                     'and exact input schema. Replace the failed approach; do not make cosmetic edits.\n')
        current_hash = code_hash(self.ctx.last_code) if self.ctx.last_code else None
        user += '\nTool output state: '+json.dumps(dict(output_available=self.ctx.last_output is not None,
                                                        output_code_hash=current_hash))
        # Earlier tool observations remain labeled, not confused with the latest submitted candidate.
        if self.tool_history:
            history_text = [json.dumps(x,default=str)[:1800] for x in self.tool_history[-3:]]
            user += '\nRecent tool observations (long entries abbreviated):\n'+'\n'.join(history_text)
        user += '\n\n'+original.NEXT_ACTION
        return [dict(role='system',content=system),dict(role='user',content=user),*base[2:]]


def run_repair(task,initial_code,before,verify_fn,client,config=None,contract_text=None,schema_text=None,
               *,arm='combined'):
    if arm not in ARMS:
        raise ValueError('Unknown repair arm: '+arm)
    evidence,guard = ARMS[arm]
    if not evidence and not guard:
        return original.run_repair(task,initial_code,before,verify_fn,client,config,contract_text,schema_text)
    if not contract_text:
        raise ValueError('An explicit intent contract is required for this development version')
    contract = json.loads(contract_text)
    cfg = config or original.AgentConfig()
    if cfg.mode != 'full':
        raise ValueError('Development arms require full repair mode')
    if guard:
        verify_fn = guarded_verifier(verify_fn,contract,before)
    if client.token_budget is None:
        client.token_budget = cfg.limits.token_budget
    klass = _EvidenceRun if evidence else original._Run
    options = dict(contract=contract) if evidence else {}
    return klass(task,initial_code,before,verify_fn,client,cfg,contract_text,schema_text,**options).run()
