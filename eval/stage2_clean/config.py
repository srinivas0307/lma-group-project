"""One frozen method for all six independent notebooks."""
import hashlib
import json
from pathlib import Path

EXPERIMENT = "stage2-clean-v1"
REVISION = "e9aaa31b85225fc14792c3d34b9f24de149f92b1"
REPO_URL = "https://github.com/srinivas0307/lma-group-project.git"
SYSTEMS = ["baseline1", "baseline2_repair", "baseline3_repair", "gold_contract",
           "proposed", "no_contract_to_generator"]
MODELS = {
    "generator": {"id": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                  "device": 1, "four_bit": False},
    "agent": {"id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
              "device": 0, "four_bit": True},
}
EXPECTED = {"pandas": "2.3.3", "numpy": "2.1.3", "torch": "2.11.0+cu130",
            "transformers": "5.20.2", "bitsandbytes": "0.50.2", "peft": "0.21.2"}
PARSES = {"baseline3_repair": "kaggle_outputs/parsing/7b_p1-v3/parses.jsonl",
          "proposed": "kaggle_outputs/parsing/ft_1_5b/parses.jsonl",
          "no_contract_to_generator": "kaggle_outputs/parsing/ft_1_5b/parses.jsonl"}
METHOD = dict(split="test", seeds=[0], temperature=0.0, dtype="float16", models=MODELS,
              generator_max_new_tokens=512, max_attempts=3, max_tool_calls=8, token_budget=60000,
              agent_max_new_tokens=768, max_invalid_actions=3,
              sandbox_timeout_s=30.0, sandbox_mem_gb=2.0, sandbox_cpu_s=20,
              generator_prompt="b2-v2", request_only_prompt="b1-v1", agent_prompt="a1-v4",
              prompt_hashes={"b2-v2": "ed85f90bc61a", "b1-v1": "ebbe8bceae71", "a1-v4": "af55b596106e"},
              parses=PARSES, attention="efficient_cuda_explicit_gqa_v1", sandbox="sb1")
FROZEN = json.loads((Path(__file__).parent / "frozen_inputs.json").read_text())
TEST_IDS = FROZEN["test_ids"]
# Infrastructure regression first, counted exactly once; quality is never used to select the runtime.
ORDER = ["taxi_dedup_003"] + [tid for tid in TEST_IDS if tid != "taxi_dedup_003"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def runtime_hashes():
    directory = Path(__file__).parent
    return {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir())
            if p.suffix in (".py", ".json")}


def common_plan(environment, backend="hf", task_ids=None):
    selected = TEST_IDS if task_ids is None else list(task_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(TEST_IDS):
        raise ValueError("Invalid task selection")
    if backend == "hf" and selected != TEST_IDS:
        raise ValueError("Real evaluations require all 100 test tasks")
    return dict(experiment=EXPERIMENT, source_revision=REVISION, method=METHOD,
                inputs=FROZEN, execution_order=ORDER, runtime=runtime_hashes(),
                environment=environment, backend=backend, selected_test_ids=selected)


def validate_plan(plan, allow_scripted=False):
    if plan != common_plan(plan.get("environment"), plan.get("backend"), plan.get("selected_test_ids")):
        raise ValueError("Bundle method, source, task inventory or runtime differs from this workflow")
    if plan["backend"] != "hf" and not (allow_scripted and plan["backend"] == "pipeline_check"):
        raise ValueError("Scripted pipeline checks are not model-quality results")
    if plan["backend"] == "hf":
        actual = plan["environment"]["packages"]
        mismatch = {k: (actual.get(k), v) for k, v in EXPECTED.items() if actual.get(k) != v}
        if mismatch:
            raise ValueError(f"Incorrect inference versions: {mismatch}")


def verify_sources(repo):
    bad = [name for name, expected in FROZEN["sources"].items()
           if not (repo / name).is_file() or digest((repo / name).read_bytes()) != expected]
    if bad:
        raise ValueError(f"Frozen source/input checks failed: {bad[:10]}")


def generator_version(preset):
    return "b2-v2" if preset in ("gold_contract", "proposed") else "b1-v1"
