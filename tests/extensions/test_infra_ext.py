"""Extensions #1 (configs), #7 (vLLM stub), #5 (Docker; skipped without it), #3 (LLM verifier), #6 (demo), #13
(training tokenisation, tool data). No GPU."""
import json
import shlex
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from igt.config import all_configs, load_system_config, to_argv

ROOT = Path(__file__).resolve().parents[2]


# ---------------- #1 configs ----------------
def test_every_config_loads_and_its_runner_exists():
    cfgs = all_configs()
    assert len(cfgs) >= 30
    for name, c in cfgs.items():
        assert (ROOT / c["runner"]).exists(), name


def test_stage2_configs_equal_the_presets():
    from eval.runner_stage2 import PRESETS, parse_args
    for p, want in PRESETS.items():
        args = parse_args(to_argv(load_system_config(f"stage2_{p}.yaml")))
        assert args.preset == p and {k: getattr(args, k) for k in want} == want
        assert args.backend == "hf" and args.agent_4bit and args.agent_model == "Qwen/Qwen2.5-7B-Instruct"


def test_agent_configs_use_the_notebook_flags():
    for name, mode in [("agent_full", "full"), ("agent_feedback", "feedback"), ("agent_naive", "naive")]:
        argv = to_argv(load_system_config(f"{name}.yaml"))
        s = " ".join(argv)
        assert f"--system {name} --mode {mode}" in s and "--load-in-4bit" in argv and "--categories R C T E S" in s


def test_overrides_come_last():
    argv = to_argv(load_system_config("stage2_baseline1.yaml"), ["--backend", "pipeline_check"])
    from eval.runner_stage2 import parse_args
    assert parse_args(argv).backend == "pipeline_check"
    np1 = to_argv(load_system_config("ext_agent_full_np1.yaml"))
    assert np1[np1.index("--stop-after-no-progress") + 1] == "3"


# ---------------- #7 vLLM (stub) ----------------
class _Out:
    def __init__(self, text, n_prompt):
        self.outputs = [types.SimpleNamespace(text=text, token_ids=[1, 2, 3])]
        self.prompt_token_ids = list(range(n_prompt))


class _StubLLM:
    def __init__(self, **kw):
        self.kw, self.calls = kw, []

    def get_tokenizer(self):
        return types.SimpleNamespace(apply_chat_template=lambda m, tokenize, add_generation_prompt: "|".join(x["content"] for x in m))

    def generate(self, prompts, sampling_params=None, use_tqdm=False, lora_request=None):
        self.calls.append((prompts, sampling_params, lora_request))
        return [_Out("hello", 7)]


@pytest.fixture
def fake_vllm(monkeypatch):
    v = types.ModuleType("vllm")
    v.SamplingParams = lambda **kw: kw
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = lambda name, i, path: ("lora", name, i, path)
    monkeypatch.setitem(sys.modules, "vllm", v)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req)
    from igt.llm.vllm_backend import VLLMChatBackend
    monkeypatch.setattr(VLLMChatBackend, "_llm_factory", staticmethod(lambda **kw: _StubLLM(**kw)))
    return VLLMChatBackend


def test_vllm_backend_interface(fake_vllm):
    from igt.llm import LLMClient
    b = fake_vllm("Qwen/X", load_in_4bit=True, adapter="/ad")
    assert b.model == "vllm:Qwen/X+adapter:/ad" and b.llm.kw["quantization"] == "bitsandbytes" and b.llm.kw["enable_lora"]
    text, n_in, n_out = b.complete([{"role": "user", "content": "hi"}], 16, 0.0, None, 3)
    assert (text, n_in, n_out) == ("hello", 7, 3)
    prompts, sp, lora = b.llm.calls[0]
    assert sp["temperature"] == 0.0 and sp["max_tokens"] == 16 and lora[0] == "lora"
    r = LLMClient(b).generate([{"role": "user", "content": "hi"}])
    assert r.text == "hello" and r.model.startswith("vllm:")


# ---------------- #5 Docker ----------------
def _docker():
    from igt.sandbox.docker_executor import docker_available
    return docker_available()


@pytest.mark.skipif(not _docker(), reason="Docker or the image igt-sandbox:docker1 is not available")
def test_docker_sandbox_matches_subprocess_and_isolates():
    from igt.sandbox import safe_run
    from igt.sandbox.docker_executor import docker_run, docker_safe_run
    df = pd.DataFrame({"a": [1, 2, None], "b": list("xyz")})
    code = "def transform(df):\n    out = df.copy()\n    out['a'] = out['a'].fillna(0)\n    return out[out['a'] < 2]\n"
    a, b = safe_run(code, df), docker_safe_run(code, df)
    assert a.status == b.status == "ok" and a.df_out.equals(b.df_out)
    assert docker_safe_run("import os\ndef transform(df):\n    return df\n", df).status == "blocked"
    net = 'def transform(df):\n    import _socket\n    s = _socket.socket()\n    s.settimeout(2)\n    s.connect(("1.1.1.1", 53))\n    return df\n'
    r = docker_run(net, df, timeout_s=20)
    assert r.status == "error" and "unreachable" in r.error.lower()
    r = docker_run(f'def transform(df):\n    open("{ROOT / "README.md"}").read()\n    return df\n', df, timeout_s=20)
    assert r.status == "error" and "No such file" in r.error
    assert docker_run("def transform(df):\n    return 3\n", df).status == "bad_return"


# ---------------- #3 LLM verifier ----------------
def test_llm_verifier_reply_and_format():
    from igt.baselines.llm_verifier import VERIFIER_SLOT, build_messages, change_summary, judge_reply
    from igt.types import validate_record, verdict_record
    b = pd.DataFrame({"p": [1, 0, 3]})
    a = b[b.p != 0]
    s = change_summary(b, a)
    assert s["rows_removed"] == 1 and s["rows_after"] == 2
    m = build_messages("drop zeros", "schema", b, a)
    assert "rows_removed" in m[1]["content"] and m[0]["role"] == "system"
    cand = dict(candidate_id="c", task_id="t", split="test", ran_ok=True)
    for text, outcome in [('{"verdict": "ACCEPT", "reason": "ok"}', "ACCEPT"), ('{"verdict": "reject", "reason": "x"}', "REJECT"),
                          ("hmm", "ESCALATE")]:
        v, info = judge_reply(text)
        rec = verdict_record("llm_verifier_7b_lv1", VERIFIER_SLOT, cand, v)
        assert rec["outcome"] == outcome and validate_record(rec, cand) == []
        assert info["malformed"] == (outcome == "ESCALATE")


def test_llm_verifier_pipeline_check(tmp_path):
    from eval.run_llm_verifier import main
    p = main(["--backend", "accept_all", "--limit", "8", "--out", str(tmp_path)])
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert len(rows) == 8 and all(r["meta"]["verifier_kind"] == "llm" for r in rows)
    assert all(r["outcome"] == ("ACCEPT" if r["verdict"]["status"] == "PASS" else "REJECT") for r in rows)


# ---------------- #6 demo ----------------
def test_demo_core_task_and_app_import():
    sys.path.insert(0, str(ROOT / "app"))
    import demo_core
    text, res = demo_core.task("titanic_dedup_002")
    assert "Intent contract" in text and res["intent"] in ("ACCEPT", "REJECT", "ESCALATE")
    assert demo_core.candidates("titanic_dedup_002") and demo_core.task_ids()
    text, res = demo_core.task("no_such_task")
    assert res is None and "[error]" in text
    import importlib
    importlib.import_module("demo_app")                     # imports without gradio or a GPU


# ---------------- #13 training tokenisation + tool data ----------------
class StubTok:
    """Character-level tokenizer with a Qwen-like chat template."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_single_turn_tokenisation_is_unchanged_and_multi_turn_trains_every_assistant_turn():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tqt", ROOT / "finetune" / "train_qlora_tools.py")
    tqt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tqt)
    single = [dict(messages=[dict(role="system", content="S"), dict(role="user", content="U"),
                             dict(role="assistant", content="CONTRACT")])]
    assert tqt.tokenize_multi_turn(single, StubTok(), 10_000) == tqt.tq.tokenize(single, StubTok(), 10_000)
    multi = [dict(messages=[dict(role="system", content="S"), dict(role="user", content="U"),
                            dict(role="assistant", content="CALL"), dict(role="user", content="RESULT"),
                            dict(role="assistant", content="CONTRACT")])]
    (ex,), dropped = tqt.tokenize_multi_turn(multi, StubTok(), 10_000)
    trained = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100)
    assert trained == "CALL<|im_end|>\nCONTRACT<|im_end|>\n" and dropped == 0
    assert "".join(map(chr, ex["input_ids"])) == StubTok().apply_chat_template(multi[0]["messages"])
    assert tqt.tokenize_multi_turn(multi, StubTok(), 20)[1] == 1


def test_tool_data_check():
    from finetune.data_gen.build_tools_data import tool_check
    assert tool_check("median", 2.5, {"median": 2.5}) is None
    assert tool_check("median", 2.5, {"median": 2.50001}) is not None
    assert tool_check("mode", "a", {"top_values": [{"value": "a", "count": 3}, {"value": "b", "count": 1}]}) is None
    assert "tie" in tool_check("mode", "a", {"top_values": [{"value": "a", "count": 3}, {"value": "b", "count": 3}]})


def test_v3_tools_data_if_built():
    p = ROOT / "finetune" / "data" / "v3_tools" / "sft_train.jsonl"
    if not p.exists():
        pytest.skip("finetune/data/v3_tools not built")
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert {r["prompt_version"] for r in rows} == {"ft-t1"}
    multi = [r for r in rows if r["tool_turns"]]
    assert multi and all(r["pattern"] in ("fill_median", "fill_most_common") for r in multi)
    for r in multi:
        call = json.loads(r["messages"][2]["content"])
        assert call["tool"] == "profile_column" and r["messages"][3]["content"].startswith("Tool result:")
    v2 = [json.loads(l)["id"] for l in (ROOT / "finetune/data/v2/sft_train.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows if "__rep" not in r["id"]] == v2


def test_full_precision_tools():
    from igt.agents.tool_parser import DataTools
    from igt.agents.tool_parser_ft import DataToolsFT
    df = pd.DataFrame({"x": [1.123456789, 2.0, None, 3.0]})
    assert DataToolsFT(df).profile_column("x")["median"] == 2.0
    df = pd.DataFrame({"x": [0.1234567891, 0.1234567891, 5.0]})
    assert DataToolsFT(df).profile_column("x")["median"] == 0.1234567891
    assert DataTools(df).profile_column("x")["median"] == round(0.1234567891, 6)     # t1-v1 unchanged
