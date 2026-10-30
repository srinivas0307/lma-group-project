"""LLM client (cache, scripted backend, budget) and the agent tool registry (README Phases 3-4 tests)."""
import numpy as np
import pandas as pd
import pytest

from igt.llm import BudgetExceeded, LLMClient, ScriptedBackend, ScriptExhausted
from igt.pipeline.failure_report import failure_report
from igt.tools import FaultInjector, ToolCall, ToolContext, build_registry
from igt.types import Verdict, predicate_failure
from igt.verifier import verify

MSG = [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]
B = pd.DataFrame({"host": ["a", "a", "b", "c"], "price": [10, 0, 30, 0], "rpm": [1.0, np.nan, 2.0, np.nan]})
CONTRACT = {"must_change": [{"type": "rows_removed_where", "condition": {"column": "price", "op": "==", "value": 0}}],
            "must_preserve": [{"type": "rows_retained_where", "condition": {"column": "price", "op": "!=", "value": 0}},
                              {"type": "columns_unchanged", "target": "*", "except": []}]}
GOOD = "def transform(df):\n    return df[df['price'] != 0].copy()\n"
BAD = "def transform(df):\n    return df[df['price'] > 10].copy()\n"


# ---------------- LLM client ----------------
def test_scripted_backend_replays_in_order_and_records_calls():
    be = ScriptedBackend(["one", lambda m: "echo:" + m[-1]["content"]])
    c = LLMClient(be)
    assert c.generate(MSG).text == "one" and c.generate(MSG).text == "echo:hello"
    assert len(be.calls) == 2 and be.calls[0][-1]["content"] == "hello"
    with pytest.raises(ScriptExhausted):
        c.generate(MSG)


def test_cache_hit_returns_identical_output_without_calling_backend(tmp_path):
    be = ScriptedBackend(["first answer"])
    c = LLMClient(be, cache_dir=tmp_path)
    r1 = c.generate(MSG, seed=1)
    r2 = LLMClient(ScriptedBackend([], model="scripted"), cache_dir=tmp_path).generate(MSG, seed=7)   # greedy: seed ignored
    assert (r1.text, r1.cached) == ("first answer", False) and (r2.text, r2.cached) == ("first answer", True)
    with pytest.raises(ScriptExhausted):          # sampling: the seed is part of the key -> miss -> backend called
        LLMClient(ScriptedBackend([], model="scripted"), cache_dir=tmp_path).generate(MSG, temperature=0.7, seed=7)


def test_token_budget():
    c = LLMClient(ScriptedBackend(["x" * 800] * 5), token_budget=150)   # ~200 tokens per call
    c.generate(MSG)
    with pytest.raises(BudgetExceeded):
        c.generate(MSG)


# ---------------- tools ----------------
@pytest.fixture
def reg_ctx():
    ctx = ToolContext(before=B, verify_fn=lambda after: verify(CONTRACT, B, after)[0],
                      sandbox_limits=dict(timeout_s=20, cpu_s=10, mem_gb=2))
    return build_registry(ctx), ctx


def test_five_tools_with_schemas(reg_ctx):
    reg, _ = reg_ctx
    assert sorted(reg.tools) == ["diff_datasets", "inspect_dataset", "profile_column", "run_transformation", "run_verifier"]
    assert all("args" in s and s["description"] for s in reg.specs())


def test_inspect_and_profile(reg_ctx):
    reg, _ = reg_ctx
    r = reg.call(ToolCall("inspect_dataset", {"n_rows": 2}))
    assert r.ok and r.output["n_rows"] == 4 and len(r.output["head"]) == 2 and r.output["head"][0]["_row_id"] == 0
    p = reg.call(ToolCall("profile_column", {"column": "rpm"})).output
    assert p["nulls"] == 2 and p["family"] == "float" and p["min"] == 1.0


def test_errors_are_structured_observations(reg_ctx):
    reg, _ = reg_ctx
    assert reg.call(ToolCall("delete_files", {})).error["type"] == "unknown_tool"
    e = reg.call(ToolCall("profile_column", {"colum": "price"})).error
    assert e["type"] == "invalid_args" and "missing required argument 'column'" in e["message"] and "unknown argument 'colum'" in e["message"]
    assert reg.call(ToolCall("profile_column", {"column": 3})).error["type"] == "invalid_args"
    assert reg.call(ToolCall("profile_column", {"column": "nope"})).error["type"] == "tool_error"
    assert reg.call(ToolCall("diff_datasets", {})).error["type"] == "tool_error"        # nothing run yet
    assert reg.call(ToolCall("inspect_dataset", {"which": "after"})).error["type"] == "invalid_args"
    assert [x["error_type"] for x in reg.log] == ["unknown_tool", "invalid_args", "invalid_args", "tool_error",
                                                  "tool_error", "invalid_args"]


def test_run_diff_verify_cycle(reg_ctx):
    reg, ctx = reg_ctx
    r = reg.call(ToolCall("run_transformation", {"code": BAD})).output
    assert r["status"] == "ok" and r["n_rows"] == 1
    d = reg.call(ToolCall("diff_datasets", {})).output
    assert d["rows_removed"] == 3 and d["removed_examples"] == [0, 1, 3] and d["alignment"] == "index"
    v = reg.call(ToolCall("run_verifier", {})).output
    assert v["status"] == "VIOLATION" and "rows_retained_where" in v["report"]
    reg.call(ToolCall("run_transformation", {"code": GOOD}))
    assert reg.call(ToolCall("run_verifier", {})).output["status"] == "PASS"


def test_run_transformation_goes_through_the_guard_and_never_changes_the_input(reg_ctx):
    reg, ctx = reg_ctx
    before = B.copy()
    r = reg.call(ToolCall("run_transformation", {"code": "import os\ndef transform(df):\n    return df"})).output
    assert r["status"] == "blocked" and ctx.safety_log and ctx.last_output is None
    reg.call(ToolCall("run_transformation", {"code": "def transform(df):\n    df['price'] = -1\n    return df"}))
    pd.testing.assert_frame_equal(B, before)
    assert reg.call(ToolCall("inspect_dataset", {})).output["head"][0]["price"] == 10     # snapshot untouched
    crash = reg.call(ToolCall("run_transformation", {"code": "def transform(df):\n    return df['nope']"})).output
    assert crash["status"] == "error" and "KeyError" in crash["error"]


def test_diff_reports_changed_cells_and_dtype(reg_ctx):
    reg, _ = reg_ctx
    reg.call(ToolCall("run_transformation", {"code": "def transform(df):\n    out = df.copy()\n    out['rpm'] = out['rpm'].fillna(0)\n    out['price'] = out['price'].astype(float)\n    return out"}))
    d = reg.call(ToolCall("diff_datasets", {})).output
    assert d["changed_cells"]["rpm"]["n_cells"] == 2 and d["dtype_changes"] == {"price": "int -> float"}


# ---------------- fault hooks ----------------
def test_faults_fire_exactly_when_configured(reg_ctx):
    reg, _ = reg_ctx
    fi = FaultInjector(reg, [{"tool": "profile_column", "on_call": 2, "kind": "error"},
                             {"tool": "inspect_dataset", "on_call": 1, "kind": "timeout"},
                             {"tool": "inspect_dataset", "on_call": 3, "kind": "malformed"}])
    pc = [fi.call(ToolCall("profile_column", {"column": "price"})) for _ in range(3)]
    assert [r.ok for r in pc] == [True, False, True] and pc[1].error["type"] == "injected_fault"
    ins = [fi.call(ToolCall("inspect_dataset", {})) for _ in range(3)]
    assert ins[0].error["type"] == "timeout" and ins[1].ok and "garbled" in ins[2].output
    assert fi.fired == [dict(tool="profile_column", on_call=2, kind="error"), dict(tool="inspect_dataset", on_call=1, kind="timeout"),
                        dict(tool="inspect_dataset", on_call=3, kind="malformed")]
    with pytest.raises(ValueError):
        FaultInjector(reg, [{"tool": "x", "on_call": 1, "kind": "explode"}])


# ---------------- failure report ----------------
def test_failure_report_formats():
    v = Verdict.combine([predicate_failure("c1", "duplicate_count", "must_change", ["host_id"], "eq 0", 182,
                                           "182 rows share a host_id", [4, 9])])
    rep = failure_report("verify", verdict=v, code_hash="abc")
    assert rep["status"] == "VIOLATION" and "[must_change] c1 duplicate_count" in rep["text"] and "[4, 9]" in rep["text"]
    assert rep["coarse"] == "Verification failed."
    s = failure_report("safety", exec_status="blocked", exec_error="import os")
    assert "safety checker" in s["text"] and s["failures"][0]["kind"] == "execution"
    assert "failed to run (timeout)" in failure_report("execute", exec_status="timeout", exec_error="30s")["text"]
