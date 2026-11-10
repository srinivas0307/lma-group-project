"""Extensions #9 (pc1 pre-check) and #8 (rq1 re-ask), and #13 data / prompt pieces. No GPU."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eval.ext_precheck import apply_precheck, flagged_note
from eval.ext_reask import reask_rows
from igt.agents.intent_parser import PROMPTS, build_messages, extract_contract, few_shot_examples, load_prompt_schema
from igt.agents.reask import original_messages, parser_kind
from igt.contracts.precheck import precheck
from igt.llm import ScriptedBackend

ROOT = Path(__file__).resolve().parents[2]
DF = pd.DataFrame({"price": [10.0, None, 30.0], "room": ["Entire home", "Private room", None], "host": [1, 1, 2]})
KEEP = [{"type": "row_count", "op": "unchanged"}]


def C(change, preserve=KEEP):
    return {"contract_version": "1.0", "must_change": change, "must_preserve": preserve}


def checks(c):
    return sorted({i["check"] for i in precheck(c, DF)})


def test_each_precheck():
    fill = lambda v: C([{"type": "column_value_relation", "target": "price", "where": {"column": "price", "op": "is_null"},
                         "fn": "const", "args": [{"lit": v}]}])
    assert checks(fill(20.0)) == []
    assert checks(fill("median")) == ["stat_name"]
    assert checks(fill("Most common")) == ["stat_name"]
    rm = lambda cond: C([{"type": "rows_removed_where", "condition": cond}])
    assert checks(rm({"column": "price", "op": ">", "value": 100})) == ["zero_match"]
    assert checks(rm({"column": "price", "op": ">", "value": 20})) == []
    assert "type_mismatch" in checks(rm({"column": "price", "op": "==", "value": "10"}))
    assert "type_mismatch" in checks(rm({"column": "room", "op": "==", "value": 3}))
    assert checks(rm({"column": "room", "op": "in", "value": ["Shared room"]})) == ["absent_value", "zero_match"]
    assert checks(rm({"column": "room", "op": "in", "value": ["Private room"]})) == []
    dup = lambda t: C([{"type": "duplicate_count", "target": t, "op": "eq", "value": 0}])
    assert checks(dup(["host"])) == [] and checks(dup(["host_id"])) == ["unknown_dup_key"]


def test_gold_contracts_pass_the_precheck():
    data = {}
    for f in sorted((ROOT / "benchmark" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        df = data.setdefault(t["data_file"], pd.read_csv(ROOT / t["data_file"]))
        assert precheck(t["gold_contract"], df) == [], t["id"]


def test_pc1_flagged_parse_becomes_invalid_and_keeps_the_original():
    tasks = {"t1": dict(data_file="x.csv")}
    good = "```json\n" + json.dumps(C([{"type": "rows_removed_where", "condition": {"column": "price", "op": ">", "value": 20}}])) + "\n```"
    bad = "```json\n" + json.dumps(C([{"type": "rows_removed_where", "condition": {"column": "price", "op": ">", "value": 99}}])) + "\n```"
    rows = apply_precheck([dict(task_id="t1", raw_output=good), dict(task_id="t1", raw_output=bad)], tasks, {"x.csv": DF})
    assert rows[0]["raw_output"] == good and not rows[0]["precheck_issues"]
    assert extract_contract(rows[1]["raw_output"])["status"] == "no_json" and rows[1]["raw_output_original"] == bad
    assert "{" not in flagged_note([dict(where="w", message="value {'a': 1}")])


# ---------------- rq1 ----------------
def _setup():
    from eval.ext_common import tasks_and_data
    from eval.parse_contracts import load_tasks_with_contracts
    tl = load_tasks_with_contracts(ROOT)
    tasks, _ = tasks_and_data()                     # raw task files, as eval/ext_reask.py loads them
    data = {f: pd.read_csv(ROOT / f) for f in {t["data_file"] for t in tl[:3]}}
    return tl, tasks, data, load_prompt_schema(ROOT), few_shot_examples(tl)


def test_reask_only_invalid_and_uses_the_original_conversation():
    tl, tasks, data, schema, ex = _setup()
    t_ok, t_bad = tl[0], tl[1]
    gold = "```json\n" + json.dumps(t_bad["gold_contract"]) + "\n```"
    rows = [dict(task_id=t_ok["id"], raw_output="```json\n" + json.dumps(t_ok["gold_contract"]) + "\n```",
                 parser_prompt_version="p1-v3", shots="few", adapter=None),
            dict(task_id=t_bad["id"], raw_output='```json\n{"must_change": [{"type": "nope"}]}\n```',
                 parser_prompt_version="p1-v3", shots="few", adapter=None)]
    backends = []

    def backend_for(t):
        backends.append(ScriptedBackend([gold], model="s"))
        return backends[-1]
    out, n = reask_rows(rows, tasks, data, schema, ex, backend_for, use_precheck=False)
    assert n == 1 and not out[0]["reasked"] and out[0]["raw_output"] == rows[0]["raw_output"]
    assert out[1]["reasked"] and out[1]["raw_output"] == gold and out[1]["raw_output_first"] == rows[1]["raw_output"]
    msgs = backends[0].calls[0]
    want = build_messages(t_bad, schema, ex, "p1-v3")         # what eval/parse_contracts.py sent
    assert msgs[: len(want)] == want and msgs[-1]["content"].startswith("Your contract is invalid")


def test_reask_kinds():
    assert parser_kind(dict(parser_prompt_version="p1-v2", adapter="a")) == "ft"
    assert parser_kind(dict(parser_prompt_version="t1-v1")) == "tools"
    assert parser_kind(dict(parser_prompt_version="ft-t1", adapter="a")) == "tools_ft"
    tl, tasks, _, schema, ex = _setup()
    t = tl[5]
    m = original_messages(dict(parser_prompt_version="p1-v2", adapter="x"), t, schema, ex)
    assert m == build_messages(t, schema, [], "p1-v2")
    m = original_messages(dict(parser_prompt_version="t1-v1", tool_calls=[dict(tool="count_rows", args={}, result={"n": 1})]),
                          t, schema, ex)
    assert m[-2]["role"] == "assistant" and m[-1]["content"].startswith("Tool result:")


def test_precheck_reask_second_answer_still_flagged_is_invalid():
    tl, tasks, data, schema, ex = _setup()
    t = next(x for x in tl if x["data_file"] in data)
    df = data[t["data_file"]]
    col = next(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    flagged = "```json\n" + json.dumps(C([{"type": "rows_removed_where",
                                            "condition": {"column": col, "op": ">", "value": 1e12}}])) + "\n```"
    rows = [dict(task_id=t["id"], raw_output=flagged, parser_prompt_version="p1-v3", shots="few", adapter=None)]
    out, n = reask_rows(rows, tasks, data, schema, ex, lambda _: ScriptedBackend([flagged]), use_precheck=True)
    assert n == 1 and out[0]["reask_reason"] == "precheck" and out[0]["parse_status"] == "precheck_flagged"
    assert extract_contract(out[0]["raw_output"])["status"] != "ok"


# ---------------- ft-t1 prompt ----------------
def test_ft_t1_prompt_is_p1v2_plus_tools_and_existing_prompts_unchanged():
    from igt.agents.tool_parser import TOOL_ADDENDUM
    from igt.agents.tool_parser_ft import SYSTEM_PROMPT_FT_TOOLS, build_ft_tool_messages
    assert SYSTEM_PROMPT_FT_TOOLS == PROMPTS["p1-v2"] + TOOL_ADDENDUM
    tl, _, _, schema, _ = _setup()
    m = build_ft_tool_messages(tl[0], schema)
    assert len(m) == 2 and m[1] == build_messages(tl[0], schema, [], "p1-v2")[1]
