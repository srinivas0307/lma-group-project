"""Attempt history for the repair agent (README Phase 9): code, failure report and tool results per attempt,
plus repeated-code detection by hash of the normalised code."""
import ast
import hashlib
import warnings
from dataclasses import dataclass, field


def code_hash(code: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            norm = ast.unparse(ast.parse(code))      # ignore formatting and comments
    except SyntaxError:
        norm = code.strip()
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


@dataclass
class Attempt:
    number: int                    # 0 = the initial candidate
    code: str
    code_hash: str
    report_text: str = ""
    status: str = ""               # PASS / VIOLATION / UNCERTAIN / blocked / error ...
    tool_calls: list = field(default_factory=list)


class Memory:
    def __init__(self, enabled=True):
        self.enabled = enabled     # "no memory" ablation: the agent only sees the latest attempt
        self.attempts: list[Attempt] = []
        self.repeat_count = 0

    def add(self, attempt: Attempt):
        self.attempts.append(attempt)

    def seen(self, h) -> bool:
        return any(a.code_hash == h for a in self.attempts)

    def history_text(self, max_code_chars=1200):
        shown = self.attempts if self.enabled else self.attempts[-1:]
        out = []
        for a in shown:
            label = "Initial code" if a.number == 0 else f"Attempt {a.number}"
            code = a.code if len(a.code) <= max_code_chars else a.code[:max_code_chars] + "\n# ... (truncated)"
            out.append(f"### {label} (result: {a.status})\n```python\n{code}\n```\n{a.report_text}")
        return "\n\n".join(out)
