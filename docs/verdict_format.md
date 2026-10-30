# Verdict format v1.0

Every verifier (Baseline 1 "none", Baseline 2 "conventional", the intent verifier) gives its decision about
one candidate in the **same format**. That is what lets `eval/metrics.py` compare all systems on the same
pool with one command (README §4.4: all verifier modes return `Verdict(status, failures, evidence_rows)`).

- **Code:** `src/igt/types.py` (the `Verdict` class, helpers to build failure entries, and `validate_record`).
- **Enforced:** `eval/metrics.py` checks every verdicts file against this format and stops with a list of
  problems if anything is wrong. Baseline 1 and Baseline 2 files already pass.
- **Tests:** `tests/test_types.py`.

## 1. One verdict = one JSON line

Each system writes one file, `verdicts_<system>.jsonl`, with **one line per pool candidate** (all of them,
including ones that crashed or were blocked).

```json
{
  "system": "intent_oracle",
  "verifier": "intent",
  "candidate_id": "airbnb_dedup_003__c00",
  "task_id": "airbnb_dedup_003",
  "split": "test",
  "verdict": {
    "status": "VIOLATION",
    "failures": [
      {"kind": "predicate", "predicate_id": "c1", "predicate_type": "duplicate_count",
       "direction": "must_change", "target": ["host_id"], "expected": "eq 0", "observed": 182,
       "detail": "182 rows still share a host_id", "evidence_rows": [12, 87, 301]}
    ],
    "evidence_rows": [12, 87, 301]
  },
  "outcome": "REJECT"
}
```

| Field | Meaning |
|---|---|
| `system` | System name from §5. Used as the row label in the results table. |
| `verifier` | Verifier slot: `none`, `conventional` or `intent`. |
| `candidate_id`, `task_id`, `split` | Copied from the pool file. `candidate_id` is the join key. |
| `verdict.status` | `PASS`, `VIOLATION` or `UNCERTAIN` (§2). |
| `verdict.failures` | **Every** failed check or predicate (§3). Empty only when `PASS`. |
| `verdict.evidence_rows` | Up to 5 input row ids that show the problem, collected from the failures. May be empty. |
| `outcome` | Final decision: `ACCEPT`, `REJECT` or `ESCALATE`. **This is what the metrics use.** |
| `meta` (optional) | Extra run information, ignored by the metrics. The intent runner writes `{"alignment": "index"|"content", "n_predicates": n}`. |

## 2. Status and outcome

| `verdict.status` | When | `outcome` (Stage 1, repair off) |
|---|---|---|
| `PASS` | Every check holds | `ACCEPT` |
| `VIOLATION` | At least one check fails, or the code crashed / was blocked | `REJECT` |
| `UNCERTAIN` | The verifier can't decide anything definite (reasons in §3) | `ESCALATE` |

**Combining several results** (`Verdict.combine`): if any check or predicate is definitely violated, the
status is `VIOLATION`, even if other predicates were uncertain. A known violation means the output is wrong,
so it is rejected. Only when nothing is violated and something is uncertain is the status `UNCERTAIN`. Both
kinds of entries stay in `failures`.

In the metrics, `REJECT` and `ESCALATE` both count as "not accepted"; the ESCALATE rate is reported
separately (README G7). With repair on (Stage 2), `VIOLATION` goes to the repair agent instead
(`Verdict.outcome(repair_on=True)` returns `REPAIR`).

## 3. Failure entries

Failures are **plain JSON objects**. In Python they are plain dicts, built with the helper functions in
`src/igt/types.py` so the required keys are always present. Every entry has a `kind`:

| `kind` | Used by | Required keys | Helper |
|---|---|---|---|
| `execution` | all systems | `status` (+ `detail`) | `execution_failure(status, detail)` |
| `check` | Baseline 2 | `check`, `value`, `threshold` (+ `column`/`columns`) | `check_failure(...)` |
| `predicate` | intent verifier | `predicate_id`, `predicate_type`, `direction`, `expected`, `observed` (+ `target`, `detail`, `evidence_rows`) | `predicate_failure(...)` |
| `uncertain` | intent verifier | `reason` (+ `detail`, `predicate_id`) | `uncertain_failure(reason, detail, predicate_id)` |

Examples:
```json
{"kind": "execution", "status": "error", "detail": "AttributeError: 'float' object has no attribute 'upper'"}
{"kind": "check", "check": "nulls.null_increase", "value": 0.859, "threshold": 0.01, "column": "education"}
{"kind": "predicate", "predicate_id": "p1", "predicate_type": "columns_unchanged", "direction": "must_preserve",
 "target": "*", "expected": "no column added or removed", "observed": "added first_occurrence", "evidence_rows": []}
{"kind": "uncertain", "reason": "rows_unmatched", "detail": "index was reset and 14 rows match several input rows"}
```

- `direction` is `must_change` or `must_preserve`. It comes from the section of the gold contract the
  predicate is in. It powers the per-direction analysis and the change-only / preserve-only ablations (README G4).
- `uncertain` reasons (fixed list): `invalid_contract`, `unknown_column`, `predicate_not_evaluable`, `rows_unmatched`.

## 4. Rules

1. One line for **every** candidate in the pool, each exactly once. `metrics.py` stops on missing, duplicate or unknown ids.
2. The verifier only sees the input data, the candidate's output (`archive/pools/pool_v2/outputs/<candidate_id>.pkl`)
   and the contract. Never the reference code, the reference output or `is_correct`.
3. Candidates with `ran_ok = false` get `VIOLATION` with an `execution` failure. Candidates that ran never have one.
4. Report **all** failed predicates, not just the first.
5. Row ids are the **input index labels** (0..n-1 as loaded). If the generated code reset the index, match rows
   by content; if that is ambiguous, report `uncertain` with reason `rows_unmatched`.
6. Same input → same verdict (deterministic; no LLM inside the verifier).

## 5. System names

| Name | Parser | Contract | Verifier |
|---|---|---|---|
| `baseline1` | none | none | none |
| `baseline2` | none | none | conventional |
| `intent_oracle` | gold contract | full | intent |
| `intent_oracle_change_only` / `intent_oracle_preserve_only` | gold contract | one direction | intent |
| `intent_prompted_7b` | prompted Qwen2.5-7B (Baseline 3) | full | intent |
| `intent_prompted_3b`, `intent_prompted_1_5b` | same prompt on Qwen2.5-3B / 1.5B, no fine-tuning (size ablation) | full | intent |
| `intent_ft_1_5b`, `intent_ft_3b` | fine-tuned Qwen2.5 | full | intent |
| `intent_ft_3b_change_only` / `intent_ft_3b_preserve_only` | fine-tuned 3B | one direction | intent |

Sensitivity variants add a suffix, e.g. `baseline2_strict`. Parser prompt versions after the first add a suffix too, e.g. `intent_prompted_7b_p1v2`. Every intent system can also have `_change_only` /
`_preserve_only` variants (the runner writes all three).

**Why the parser grid:** the prompted 3B / 1.5B runs separate the effect of model size (prompted 7B vs 3B vs 1.5B)
from the effect of fine-tuning (prompted 3B vs fine-tuned 3B, same for 1.5B). Prompt, decoding, verifier and pool
are identical across all of them. The candidate code generator (Qwen2.5-3B + 1.5B) is not part of this comparison:
it only produces the pool every system judges.

## 6. Minimal example for the intent verifier

```python
from igt.types import Verdict, execution_failure, predicate_failure, uncertain_failure, verdict_record

failures = []
for pred in contract_predicates:                       # every predicate, don't stop at the first
    result = check_predicate(before, after, pred)      # your code
    if result.violated:
        failures.append(predicate_failure(pred.id, pred.type, pred.direction, pred.target,
                                          result.expected, result.observed, result.detail, result.rows))
    elif result.uncertain:
        failures.append(uncertain_failure("predicate_not_evaluable", result.detail, pred.id))

record = verdict_record("intent_oracle", "intent", candidate, Verdict.combine(failures))
```

Then compare all systems on the same pool:
```
.venv/bin/python eval/metrics.py --pool archive/pools/pool_v2/pool_v2.jsonl \
    --verdicts archive/pools/pool_v2/verdicts_baseline1.jsonl archive/results/baseline2_v2/verdicts_baseline2.jsonl \
               archive/results/intent_oracle_v2/verdicts_intent_oracle.jsonl
```

## 7. Decisions (2026-10-29)

| Question | Decision | Why |
|---|---|---|
| Plain dicts or typed classes for failures? | Plain dicts, built with helper functions | Baselines 1 and 2 already write dicts; JSON in and out with no conversion; the helpers and `validate_record` still guarantee the required keys. |
| All failed predicates, or stop at the first? | All | Needed to measure which predicates catch which errors and for the one-direction ablations. Costs almost nothing. |
| Violation and uncertain in the same candidate? | `VIOLATION` wins; both are kept in `failures` | A definite violation means the output is wrong; escalating it would hide a catch. |
| Does `UNCERTAIN` explain why? | Yes, `kind: "uncertain"` with a fixed `reason` | So the ESCALATE rate can be broken down by cause (README G7). |
| System names? | Table in §5 | One name per results-table row; ablations and sensitivity variants by suffix. |

Changes to this format after agreement get a new version number and a line here.
