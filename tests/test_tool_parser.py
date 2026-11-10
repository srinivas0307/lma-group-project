"""Tool-using intent parser (extra experiment): tools, protocol, limits, and that it differs from Baseline 3 only by
the tool instructions."""
import json
from pathlib import Path

import pandas as pd

from eval.parse_contracts import load_tasks_with_contracts
from igt.agents.intent_parser import PROMPTS, build_messages, few_shot_examples, load_prompt_schema
from igt.agents.tool_parser import (MAX_TOOL_CALLS, SYSTEM_PROMPT, DataTools, build_tool_messages, parse_with_tools,
                                    tool_call_of)
from igt.llm import LLMClient, ScriptedBackend

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks_with_contracts(ROOT)
EX = few_shot_examples(TASKS)
SCHEMA = load_prompt_schema(ROOT)
T = next(t for t in TASKS if t["id"] == "titanic_miss_004")
DF = pd.read_csv(ROOT / T["data_file"])
FINAL = "```json\n" + json.dumps(T["gold_contract"]) + "\n```"


def test_same_as_baseline3_except_tool_instructions():
    b3, tp = build_messages(T, SCHEMA, EX, "p1-v3"), build_tool_messages(T, SCHEMA, EX)
    assert tp[1:] == b3[1:] and SYSTEM_PROMPT.startswith(PROMPTS["p1-v3"]) and "profile_column" in tp[0]["content"]


def test_tools_return_the_values_the_gold_contract_uses():
    tools = DataTools(DF)
    assert tools.call("profile_column", {"column": "Age"})[1]["median"] == 28.0
    first = tools.call("profile_column", {"column": "Age", "where": {"column": "Pclass", "op": "==", "value": 1}})[1]
    assert first["median"] == 37.0
    assert tools.call("count_rows", {"where": {"column": "Age", "op": "is_null"}})[1]["matching_rows"] == 177
    assert tools.call("profile_column", {"column": "Embarked"})[1]["top_values"][0]["value"] == "S"


def test_tool_errors_are_returned_not_raised():
    tools = DataTools(DF)
    assert tools.call("drop_table", {})[0] is False
    assert "no column" in tools.call("profile_column", {"column": "nope"})[1]["error"]
    assert tools.call("count_rows", {"where": {"column": "Age", "op": "<"}})[0] is False
    assert tools.call("inspect_rows", "notadict")[0] is False


def test_protocol_tool_then_answer():
    be = ScriptedBackend([json.dumps({"tool": "profile_column", "args": {"column": "Age"}}), FINAL])
    res = parse_with_tools(T, SCHEMA, EX, DF, LLMClient(be))
    assert res["raw_output"] == FINAL and len(res["tool_calls"]) == 1 and res["tool_calls"][0]["result"]["median"] == 28.0
    assert '"median": 28.0' in be.calls[1][-1]["content"]          # the result went back to the model


def test_limit_forces_an_answer():
    call = json.dumps({"tool": "count_rows", "args": {"where": {"column": "Age", "op": "is_null"}}})
    be = ScriptedBackend([call] * (MAX_TOOL_CALLS + 1) + [FINAL])
    res = parse_with_tools(T, SCHEMA, EX, DF, LLMClient(be))
    assert len(res["tool_calls"]) == MAX_TOOL_CALLS and res["forced_answer"] and res["raw_output"] == FINAL
    be2 = ScriptedBackend([call] * (MAX_TOOL_CALLS + 2))            # keeps calling after the last warning: stop
    assert parse_with_tools(T, SCHEMA, EX, DF, LLMClient(be2))["forced_answer"]


def test_a_contract_is_not_mistaken_for_a_tool_call():
    assert tool_call_of(FINAL) is None and tool_call_of("no json") is None
    assert tool_call_of('{"tool": "count_rows", "args": {}}') == ("count_rows", {})


def test_runner_output_is_scored_like_any_parser(tmp_path):
    from eval import contract_metrics, parse_contracts_tools
    rows = parse_contracts_tools.main(["--backend", "gold", "--out", str(tmp_path), "--limit", "3"])
    assert len(rows) == 3 and all(r["parse_status"] == "ok" and r["n_tool_calls"] == 1 for r in rows)
    assert json.loads((tmp_path / "env.json").read_text())["valid_result"] is False
    per_task = contract_metrics.score(rows, {t["id"]: t["gold_contract"] for t in TASKS},
                                      {t["id"]: list(pd.read_csv(ROOT / t["data_file"], nrows=1).columns) for t in TASKS})
    assert all(d["exact"] for d in per_task)
