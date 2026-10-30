"""Baseline 3 parser prompt, output extraction and contract metrics (README Phase 6 tests)."""
from pathlib import Path

import pytest

from eval.contract_metrics import aggregate, score
from eval.parse_contracts import load_tasks_with_contracts
from igt.agents.intent_parser import (FEW_SHOT_IDS, PROMPT_HASH, build_messages, contract_to_text,
                                      extract_contract, few_shot_examples, load_prompt_schema)
from igt.contracts.normalize import canon_condition, contract_keys
from igt.contracts.schema import Condition

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks_with_contracts(ROOT)
SCHEMA = load_prompt_schema(ROOT)
COLS = ["a", "b", "c", "w"]


def test_few_shot_are_dev_only_and_one_per_type():
    ex = few_shot_examples(TASKS)
    assert all(e["split"] == "dev" for e in ex)
    assert len({e["transformation_type"] for e in ex}) == 5


def test_prompt_never_contains_test_requests():
    test_requests = {t["request"] for t in TASKS if t["split"] == "test"}
    target = next(t for t in TASKS if t["split"] == "test")
    msgs = build_messages(target, SCHEMA, few_shot_examples(TASKS))
    shown = [m["content"].rsplit("Request: ", 1)[1] for m in msgs[1:-1] if m["role"] == "user"]
    assert not (set(shown) & test_requests)
    assert msgs[-1]["content"].endswith(target["request"])


def test_leave_one_out_for_dev_examples():
    ex = few_shot_examples(TASKS)
    msgs = build_messages(ex[0], SCHEMA, ex)
    assert sum(m["role"] == "assistant" for m in msgs) == len(ex) - 1


def test_prompt_size_fits_a_7b_context():
    for t in TASKS:
        n_chars = sum(len(m["content"]) for m in build_messages(t, SCHEMA, few_shot_examples(TASKS)))
        assert n_chars < 24_000, (t["id"], n_chars)          # roughly < 7k tokens


def test_examples_round_trip_through_extraction():
    for e in few_shot_examples(TASKS):
        ex = extract_contract(contract_to_text(e["gold_contract"]))
        assert ex["status"] == "ok"
        assert contract_keys(ex["contract"], COLS_OF(e)) == contract_keys(e["gold_contract"], COLS_OF(e))


def COLS_OF(t):
    return [c["name"] for c in SCHEMA[t["data_file"]]["columns"]]


@pytest.mark.parametrize("text, status", [
    ('Here it is:\n```json\n{"contract_version": "1.0", "must_change": [{"type": "row_count", "target": null, "op": "decreased"}], "must_preserve": []}\n```', "ok"),
    ('{"must_change": [{"type": "row_count", "op": "decreased"}]} trailing words', "ok"),
    ("no contract here", "no_json"),
    ('```json\n{"must_change": [{"type": "row_count", "op": "decreased",}]}\n```', "invalid_json"),
    ('```json\n{"must_change": [{"type": "remove_rows"}]}\n```', "invalid_contract"),
    ('```json\n{"must_change": [{"type": "row_count", "op": "eq"}]}\n```', "invalid_contract"),
])
def test_extract_contract(text, status):
    assert extract_contract(text)["status"] == status


def test_prompt_hash_is_stable():
    assert len(PROMPT_HASH) == 12 and FEW_SHOT_IDS[0] == "adult_dedup_006"


# ---------------- normalization (guide §5.8) ----------------
def test_condition_canonical_form():
    a = Condition.model_validate({"all": [{"column": "b", "op": "==", "value": 1.0}, {"column": "a", "op": "in", "value": [3, 1]}]})
    b = Condition.model_validate({"all": [{"column": "a", "op": "in", "value": [1, 3]}, {"all": [{"column": "b", "op": "==", "value": 1}]}]})
    assert canon_condition(a) == canon_condition(b)
    nn = Condition.model_validate({"not": {"not": {"column": "a", "op": "is_null"}}})
    assert canon_condition(nn) == {"column": "a", "op": "is_null"}


def test_star_expansion_gives_partial_credit():
    gold = {"must_preserve": [{"type": "columns_unchanged", "target": "*", "except": ["w"]}]}
    pred = {"must_preserve": [{"type": "columns_unchanged", "target": ["a", "b"]}]}
    g, p = contract_keys(gold, COLS), contract_keys(pred, COLS)
    assert len(g["must_preserve"]) == 3 and len(p["must_preserve"] & g["must_preserve"]) == 2


def test_ids_and_order_are_ignored():
    g = {"must_change": [{"id": "c1", "type": "row_count", "op": "decreased"},
                         {"id": "c2", "type": "null_count", "target": "a", "op": "eq", "value": 0}]}
    p = {"must_change": [{"type": "null_count", "target": "a", "op": "eq", "value": 0.0},
                         {"type": "row_count", "op": "decreased"}]}
    assert contract_keys(g, COLS) == contract_keys(p, COLS)


def test_contract_metrics_on_toy_parses():
    gold = {"t1": {"must_change": [{"type": "row_count", "op": "decreased"},
                                   {"type": "null_count", "target": "a", "op": "eq", "value": 0}],
                   "must_preserve": [{"type": "columns_unchanged", "target": "*", "except": []}]},
            "t2": {"must_change": [{"type": "row_count", "op": "decreased"}], "must_preserve": []}}
    cols = {"t1": COLS, "t2": COLS}
    rows = [dict(task_id="t1", split="test", transformation_type="x",
                 raw_output='{"must_change": [{"type": "row_count", "op": "decreased"}, {"type": "row_count", "op": "eq", "value": 3}],'
                            ' "must_preserve": [{"type": "columns_unchanged", "target": "*", "except": ["w"]}]}'),
            dict(task_id="t2", split="test", transformation_type="x", raw_output="sorry, no idea")]
    m = aggregate(score(rows, gold, cols))
    # must_change: gold {rc, nc} + {rc}; pred t1 {rc, rc_eq3}, t2 invalid -> tp 1, fp 1, fn 2
    assert m["must_change"]["tp"] == 1 and m["must_change"]["fp"] == 1 and m["must_change"]["fn"] == 2
    # must_preserve: gold 4 columns, pred 3 of them
    assert (m["must_preserve"]["tp"], m["must_preserve"]["fp"], m["must_preserve"]["fn"]) == (3, 0, 1)
    assert m["invalid_rate"] == 0.5 and m["exact_match"] == 0.0


def test_prompt_versions():
    from igt.agents.intent_parser import PROMPTS, prompt_hash
    assert prompt_hash("p1-v1") == "1b8d8783f898"                 # the first Kaggle run must stay reproducible
    assert PROMPTS["p1-v2"].startswith(PROMPTS["p1-v1"]) and "Format rules" in PROMPTS["p1-v2"]
    t = next(t for t in TASKS if t["split"] == "test")
    assert build_messages(t, SCHEMA, few_shot_examples(TASKS), "p1-v1")[0]["content"] == PROMPTS["p1-v1"]


# ---------------- p1-v3 reference card ----------------
def _card_snippets():
    import json as _json
    from igt.agents.intent_parser import REFERENCE_CARD
    out = []
    for line in REFERENCE_CARD.splitlines():
        line = line.strip()
        if line.startswith("{"):
            obj, _ = _json.JSONDecoder().raw_decode(line)
            out.append(obj)
    return out


def test_card_snippets_are_valid():
    from igt.contracts import load_contract
    for s in _card_snippets():
        if "type" in s:
            load_contract({"must_change": [s]})
        else:
            load_contract({"must_change": [{"type": "rows_removed_where", "target": None, "condition": s}]})


def test_prompt_v3_covers_whole_vocabulary():
    """Design rule of p1-v3: every vocabulary element appears in exact JSON (few-shot examples + card)."""
    import json as _json
    from igt.agents.intent_parser import PROMPTS
    from igt.contracts.schema import COMPARE_OPS, COUNT_OPS_ABSOLUTE, COUNT_OPS_RELATIVE, DTYPE_FAMILIES, FUNCTIONS, PREDICATE_TYPES
    shown = PROMPTS["p1-v3"] + "".join(contract_to_text(e["gold_contract"]) for e in few_shot_examples(TASKS))
    for t in PREDICATE_TYPES:
        assert f'"type": "{t}"' in shown, t
    for f in FUNCTIONS:
        assert f'"fn": "{f}"' in shown, f
    for op in list(COMPARE_OPS) + ["in", "not_in", "is_null", "not_null"]:
        assert f'"op": "{op}"' in shown, op
    for k in ("all", "any", "not"):
        assert f'"{k}": ' in shown, k
    for op in COUNT_OPS_RELATIVE + COUNT_OPS_ABSOLUTE:
        assert f'"{op}"' in shown, op
    for d in DTYPE_FAMILIES:
        assert f'"{d}"' in shown, d
    assert '"keep": "first"' in shown and '"keep": "last"' in shown


def test_card_uses_no_evaluation_columns_or_phrasing():
    from igt.agents.intent_parser import REFERENCE_CARD
    eval_cols = {c["name"] for ds in SCHEMA.values() for c in ds["columns"]}
    def names(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("column", "col", "target"):
                    yield from ([v] if isinstance(v, str) else v if isinstance(v, list) else [])
                else:
                    yield from names(v)
        elif isinstance(o, list):
            for x in o:
                yield from names(x)
    card_cols = {n for s in _card_snippets() for n in names(s)}
    assert not (card_cols & eval_cols), card_cols & eval_cols
    import re as _re
    words = lambda t: _re.findall(r"[a-z0-9_]+", t.lower())
    grams = lambda t, n=5: {tuple(words(t)[i:i + n]) for i in range(len(words(t)) - n + 1)}
    test_grams = set().union(*(grams(t["request"]) for t in TASKS if t["split"] == "test"))
    assert not (grams(REFERENCE_CARD) & test_grams)


def test_prompt_v3_fits_context():
    from igt.agents.intent_parser import build_messages
    worst = max(sum(len(m["content"]) for m in build_messages(t, SCHEMA, few_shot_examples(TASKS), "p1-v3")) for t in TASKS)
    assert worst < 24_000, worst
