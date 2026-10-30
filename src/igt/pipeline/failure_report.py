"""One failure-report format for every way a candidate can fail (README Phase 9, task 3):
safety block, sandbox crash/timeout, intent-predicate violations, uncertain predicates, Baseline 2 check failures.

    rep = failure_report(stage="verify", verdict=v)            # or stage="safety"/"execute" with exec_result
    rep["text"]   # what the repair agent reads
    rep["coarse"] # pass/fail only (for the "coarse report" ablation)

The report never contains reference code, reference outputs or correctness labels (README §4.8 invariant 2).
"""
from igt.types import Verdict


def _line(f):
    k = f.get("kind")
    if k == "predicate":
        s = (f"[{f['direction']}] {f['predicate_id']} {f['predicate_type']} on {f.get('target')}: expected "
             f"{f['expected']}, observed {f['observed']}.")
        if f.get("detail"):
            s += f" {f['detail']}"
        if f.get("evidence_rows"):
            s += f" Example input rows: {f['evidence_rows']}."
        return s
    if k == "uncertain":
        return f"could not check{' ' + f['predicate_id'] if f.get('predicate_id') else ''} ({f['reason']}): {f.get('detail', '')}"
    if k == "check":
        col = f" in column {f['column']}" if f.get("column") else ""
        return f"generic check {f['check']} failed{col}: value {f['value']}, limit {f['threshold']}"
    return f"{f.get('status', 'error')}: {f.get('detail')}"


def failure_report(stage, verdict: Verdict | None = None, exec_status=None, exec_error=None, code_hash=None):
    """stage: 'safety' (blocked by the AST guard), 'execute' (crash/timeout/bad return), or 'verify'."""
    if stage in ("safety", "execute"):
        failures = [dict(kind="execution", status=exec_status, detail=exec_error)]
        head = ("The code was blocked by the safety checker" if stage == "safety"
                else f"The code failed to run ({exec_status})")
        status = "VIOLATION"
    else:
        assert verdict is not None
        failures, status = verdict.failures, verdict.status
        head = {"VIOLATION": "The output violates the intent of the request",
                "UNCERTAIN": "The output could not be checked",
                "PASS": "The output passes every check"}[status]
    lines = [f"{head}."] + [f"- {_line(f)}" for f in failures]
    return dict(stage=stage, status=status, failures=failures, code_hash=code_hash, text="\n".join(lines),
                coarse="Verification passed." if status == "PASS" else "Verification failed.")
