"""All limits from README §4.8, in one place (set from config; defaults shown in the README)."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Limits:
    max_attempts: int = 3              # propose_fix calls; exceeded -> REJECT
    max_tool_calls_per_attempt: int = 8  # then the next action must be propose_fix or escalate
    max_invalid_actions: int = 3       # malformed / disallowed actions in one attempt before the run is rejected
    sandbox_timeout_s: float = 30.0
    sandbox_mem_gb: float = 2.0
    sandbox_cpu_s: int = 20
    token_budget: int = 60_000         # per task, same for all systems; exceeded -> REJECT
    max_new_tokens: int = 768          # per agent step

    def sandbox(self):
        return dict(timeout_s=self.sandbox_timeout_s, cpu_s=self.sandbox_cpu_s, mem_gb=self.sandbox_mem_gb)

    def to_dict(self):
        return asdict(self)
