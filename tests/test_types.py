import json
from pathlib import Path

import pytest

from eval import metrics
from igt.types import (Verdict, check_failure, execution_failure, predicate_failure, uncertain_failure,
                       validate_record, verdict_record)

ROOT = Path(__file__).resolve().parents[1]
CAND = dict(candidate_id="t1__c00", task_id="t1", split="test", ran_ok=True, is_correct=False,
            transformation_type="deduplication")


def pf(pid="c1", direction="must_change", rows=(3, 4)):
    return predicate_failure(pid, "duplicate_count", direction, ["host_id"], "eq 0", 182, "182 rows share a host_id", rows)


def test_combine_precedence():
    assert Verdict.combine([]).status == "PASS"
    assert Verdict.combine([uncertain_failure("rows_unmatched")]).status == "UNCERTAIN"
    v = Verdict.combine([uncertain_failure("unknown_column", predicate_id="p1"), pf()])
    assert v.status == "VIOLATION" and len(v.failures) == 2          # all failures kept, violation wins
    assert v.evidence_rows == [3, 4]


def test_outcomes():
    assert Verdict.combine([]).outcome() == "ACCEPT"
    assert Verdict.combine([pf()]).outcome() == "REJECT"
    assert Verdict.combine([pf()]).outcome(repair_on=True) == "REPAIR"
    assert Verdict.combine([uncertain_failure("invalid_contract")]).outcome() == "ESCALATE"


def test_evidence_capped_at_five():
    v = Verdict.combine([pf(rows=range(10)), pf("c2", rows=range(5, 20))])
    assert v.evidence_rows == [0, 1, 2, 3, 4]


def test_valid_records_pass():
    for fails in ([], [pf()], [uncertain_failure("predicate_not_evaluable", "x", "p2")],
                  [check_failure("nulls.null_increase", 0.8, 0.01, column="education")]):
        r = verdict_record("intent_oracle", "intent", CAND, Verdict.combine(fails))
        assert validate_record(r, CAND) == [], r


@pytest.mark.parametrize("mutate, problem", [
    (lambda r: r.pop("outcome"), "missing field 'outcome'"),
    (lambda r: r.update(outcome="ACCEPT"), "does not match status"),
    (lambda r: r["verdict"].update(status="FAIL"), "status 'FAIL'"),
    (lambda r: r["verdict"]["failures"][0].update(kind="oops"), "failure kind 'oops'"),
    (lambda r: r["verdict"]["failures"][0].pop("observed"), "missing 'observed'"),
    (lambda r: r["verdict"]["failures"][0].update(direction="both"), "direction 'both'"),
    (lambda r: r["verdict"].update(failures=[]), "PASS must have no failures"),
    (lambda r: r.update(split="dev"), "differs from the pool"),
    (lambda r: r["verdict"]["failures"].append(execution_failure("error", "x")), "must not have an execution failure"),
])
def test_invalid_records_are_caught(mutate, problem):
    r = verdict_record("intent_oracle", "intent", CAND, Verdict.combine([pf()]))
    mutate(r)
    assert any(problem in p for p in validate_record(r, CAND)), validate_record(r, CAND)


def test_crashed_candidate_must_be_execution_violation():
    crashed = dict(CAND, ran_ok=False)
    ok = verdict_record("x", "intent", crashed, Verdict.combine([execution_failure("error", "KeyError")]))
    assert validate_record(ok, crashed) == []
    bad = verdict_record("x", "intent", crashed, Verdict.combine([]))
    assert validate_record(bad, crashed)


def test_explain_mentions_direction_and_predicate():
    text = Verdict.combine([pf(), uncertain_failure("unknown_column", "no column 'hostid'")]).explain()
    assert "VIOLATION" in text and "[must_change] c1" in text and "unknown_column" in text


def test_metrics_on_toy_pool_hand_checked(tmp_path):
    # 2 correct (1 rejected), 3 runnable wrong (2 not accepted, one of them escalated), 1 crash
    spec = [("a", 1, 1, []), ("b", 1, 1, [pf()]), ("c", 1, 0, [pf()]), ("d", 1, 0, [uncertain_failure("rows_unmatched")]),
            ("e", 1, 0, []), ("f", 0, 0, [execution_failure("error", "x")])]
    pool = [dict(candidate_id=i, task_id=i, split="test", transformation_type="filtering", ran_ok=bool(r),
                 is_correct=bool(k)) for i, r, k, _ in spec]
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(c) + "\n" for c in pool))
    recs = [verdict_record("toy", "intent", c, Verdict.combine(f)) for c, (*_, f) in zip(pool, spec)]
    (tmp_path / "v.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rates = metrics.evaluate(tmp_path / "p.jsonl", [tmp_path / "v.jsonl"], 200)["test"]["toy"]["overall"]["rates"]
    assert rates == dict(recall=0.667, recall_all=0.75, false_rejection=0.5, silent_error_rate=0.5, escalate_rate=0.167)


def test_existing_verdict_files_are_valid():
    for pool, files in [("eval/pool_v2/pool_v2.jsonl", ["eval/pool_v2/verdicts_baseline1.jsonl",
                                                         "results/baseline2_v2/verdicts_baseline2.jsonl"])]:
        if not (ROOT / pool).exists():
            pytest.skip("pool v2 not present")
        P = {c["candidate_id"]: c for c in map(json.loads, (ROOT / pool).read_text().splitlines())}
        for f in files:
            metrics.load_verdicts(ROOT / f, P)          # raises SystemExit on any format problem


def test_mcnemar():
    assert metrics.mcnemar_exact(0, 13) == 2 / 2 ** 13 and metrics.mcnemar_exact(3, 3) == 1.0
