"""Shared types between team members (README Phase 0). Spec: docs/verdict_format.md (format v1.0).

Every verifier (none / conventional / intent) returns a `Verdict`. Systems write one JSON line per pool
candidate with `verdict_record(...)`; `eval/metrics.py` checks every file with `validate_record(...)`.
"""
from dataclasses import dataclass, field
from typing import Literal

FORMAT_VERSION = "1.0"

Status = Literal["PASS", "VIOLATION", "UNCERTAIN"]
Outcome = Literal["ACCEPT", "REJECT", "ESCALATE"]
STATUSES = ("PASS", "VIOLATION", "UNCERTAIN")
OUTCOMES = ("ACCEPT", "REJECT", "ESCALATE")
VERIFIERS = ("none", "conventional", "intent")
FAILURE_KINDS = ("execution", "check", "predicate", "uncertain")
DIRECTIONS = ("must_change", "must_preserve")
UNCERTAIN_REASONS = ("invalid_contract", "unknown_column", "predicate_not_evaluable", "rows_unmatched")
MAX_EVIDENCE = 5

# Standard system names (results-table row labels).
SYSTEMS = (
    "baseline1", "baseline2",
    "intent_oracle", "intent_oracle_change_only", "intent_oracle_preserve_only",
    "intent_prompted_7b", "intent_prompted_3b", "intent_prompted_1_5b",
    "intent_ft_1_5b", "intent_ft_3b", "intent_ft_3b_change_only", "intent_ft_3b_preserve_only",
)


# ---------------- failure entries (plain dicts, so they serialise as they are) ----------------
def execution_failure(status: str, detail: str | None) -> dict:
    """The code never produced a DataFrame: parse_error, blocked, error, bad_return, timeout, killed, memory."""
    return dict(kind="execution", status=status, detail=detail)


def check_failure(check: str, value, threshold, column=None, columns=None) -> dict:
    """Baseline 2: a conventional check failed."""
    f = dict(kind="check", check=check, value=value, threshold=threshold)
    if column:
        f["column"] = column
    if columns:
        f["columns"] = columns
    return f


def predicate_failure(predicate_id: str, predicate_type: str, direction: str, target, expected: str,
                      observed, detail: str = "", evidence_rows=()) -> dict:
    """Intent verifier: a contract predicate is violated."""
    assert direction in DIRECTIONS, direction
    return dict(kind="predicate", predicate_id=predicate_id, predicate_type=predicate_type, direction=direction,
                target=target, expected=expected, observed=observed, detail=detail,
                evidence_rows=list(evidence_rows)[:MAX_EVIDENCE])


def uncertain_failure(reason: str, detail: str = "", predicate_id: str | None = None) -> dict:
    """Intent verifier: it can't decide (README G7, G9)."""
    assert reason in UNCERTAIN_REASONS, reason
    f = dict(kind="uncertain", reason=reason, detail=detail)
    if predicate_id:
        f["predicate_id"] = predicate_id
    return f


# ---------------- verdict ----------------
@dataclass
class Verdict:
    status: Status
    failures: list[dict] = field(default_factory=list)
    evidence_rows: list = field(default_factory=list)

    @classmethod
    def combine(cls, failures: list[dict]) -> "Verdict":
        """Status from all failures: any violation -> VIOLATION (definite), else any uncertain -> UNCERTAIN,
        else PASS. Report every failure, not just the first."""
        definite = [f for f in failures if f["kind"] != "uncertain"]
        status = "VIOLATION" if definite else "UNCERTAIN" if failures else "PASS"
        ev = [r for f in definite for r in f.get("evidence_rows", [])]
        return cls(status, list(failures), list(dict.fromkeys(ev))[:MAX_EVIDENCE])

    def outcome(self, repair_on: bool = False) -> str:
        if self.status == "PASS":
            return "ACCEPT"
        if self.status == "UNCERTAIN":
            return "ESCALATE"
        return "REPAIR" if repair_on else "REJECT"      # Stage 2: a violation goes to the repair agent

    def to_dict(self) -> dict:
        return dict(status=self.status, failures=self.failures, evidence_rows=self.evidence_rows)

    def explain(self) -> str:
        if self.status == "PASS":
            return "PASS: every check holds"
        lines = [f"{self.status}: {len(self.failures)} problem(s)"]
        for f in self.failures:
            if f["kind"] == "predicate":
                lines.append(f"  - [{f['direction']}] {f['predicate_id']} {f['predicate_type']} on {f['target']}: "
                             f"expected {f['expected']}, observed {f['observed']}. {f.get('detail', '')}".rstrip())
            elif f["kind"] == "check":
                lines.append(f"  - check {f['check']}: value {f['value']} vs threshold {f['threshold']}"
                             + (f" ({f['column']})" if f.get("column") else ""))
            elif f["kind"] == "uncertain":
                lines.append(f"  - uncertain ({f['reason']}): {f.get('detail', '')}")
            else:
                lines.append(f"  - execution {f['status']}: {f.get('detail')}")
        if self.evidence_rows:
            lines.append(f"  evidence rows: {self.evidence_rows}")
        return "\n".join(lines)


def verdict_record(system: str, verifier: str, candidate: dict, verdict: Verdict) -> dict:
    """One line of verdicts_<system>.jsonl (Stage 1: repair off)."""
    return dict(system=system, verifier=verifier, candidate_id=candidate["candidate_id"],
                task_id=candidate["task_id"], split=candidate["split"], verdict=verdict.to_dict(),
                outcome=verdict.outcome())


# ---------------- validation (used by eval/metrics.py) ----------------
_REQUIRED_FAILURE_KEYS = {
    "execution": ("status",),
    "check": ("check", "value", "threshold"),
    "predicate": ("predicate_id", "predicate_type", "direction", "expected", "observed"),
    "uncertain": ("reason",),
}
_OUTCOME_FOR = {"PASS": "ACCEPT", "VIOLATION": "REJECT", "UNCERTAIN": "ESCALATE"}


def validate_record(r: dict, candidate: dict | None = None) -> list[str]:
    """Return a list of problems with one verdict line (empty = valid). Stage 1 rules (repair off)."""
    p = []
    for k in ("system", "verifier", "candidate_id", "task_id", "split", "verdict", "outcome"):
        if k not in r:
            p.append(f"missing field '{k}'")
    if p:
        return p
    v = r["verdict"]
    if r["verifier"] not in VERIFIERS:
        p.append(f"verifier '{r['verifier']}' not in {VERIFIERS}")
    if v.get("status") not in STATUSES:
        return p + [f"status '{v.get('status')}' not in {STATUSES}"]
    if r["outcome"] != _OUTCOME_FOR[v["status"]]:
        p.append(f"outcome {r['outcome']} does not match status {v['status']}")
    fails = v.get("failures", [])
    if (v["status"] == "PASS") != (not fails):
        p.append("PASS must have no failures, and VIOLATION/UNCERTAIN at least one")
    for f in fails:
        kind = f.get("kind")
        if kind not in FAILURE_KINDS:
            p.append(f"failure kind '{kind}' not in {FAILURE_KINDS}")
            continue
        p += [f"{kind} failure missing '{k}'" for k in _REQUIRED_FAILURE_KEYS[kind] if k not in f]
        if kind == "predicate" and f.get("direction") not in DIRECTIONS:
            p.append(f"predicate direction '{f.get('direction')}' not in {DIRECTIONS}")
        if kind == "uncertain" and f.get("reason") not in UNCERTAIN_REASONS:
            p.append(f"uncertain reason '{f.get('reason')}' not in {UNCERTAIN_REASONS}")
    if v["status"] == "UNCERTAIN" and any(f.get("kind") != "uncertain" for f in fails):
        p.append("UNCERTAIN may only contain uncertain failures (any definite failure makes it VIOLATION)")
    if len(v.get("evidence_rows", [])) > MAX_EVIDENCE:
        p.append(f"more than {MAX_EVIDENCE} evidence rows")
    if candidate is not None:
        for k in ("task_id", "split"):
            if r[k] != candidate[k]:
                p.append(f"{k} '{r[k]}' differs from the pool ('{candidate[k]}')")
        exec_fail = any(f.get("kind") == "execution" for f in fails)
        if not candidate["ran_ok"] and not (v["status"] == "VIOLATION" and exec_fail):
            p.append("candidate did not run: must be VIOLATION with an execution failure")
        if candidate["ran_ok"] and exec_fail:
            p.append("candidate ran: must not have an execution failure")
    return p
