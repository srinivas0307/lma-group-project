"""Run one standalone arm with the frozen evaluator and atomic per-task commits."""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

from .artifacts import bundle_name, export_arm, task_payloads, write_json
from .config import (EXPERIMENT, METHOD, MODELS, ORDER, PARSES, SYSTEMS, TEST_IDS,
                     canonical, common_plan, digest, generator_version, verify_sources)


def evaluate(task, preset, before, reference, schema, parsed, generator, agent, cache):
    # Reuse the frozen implementation for prompts, parsers, tools, safety, verification and equivalence.
    from eval import runner_stage2 as old
    from igt.agents.code_generator_v2 import build_generator_messages

    started = time.monotonic()
    flags = old.PRESETS[preset]
    limits = old.Limits(max_attempts=METHOD["max_attempts"], max_tool_calls_per_attempt=METHOD["max_tool_calls"],
                        token_budget=METHOD["token_budget"])
    request = " ".join(str(task["request"]).split())
    rec = dict(task_id=task["id"], split="test", dataset=task["dataset"],
               transformation_type=task["transformation_type"], difficulty=task.get("difficulty"),
               seed=0, system=preset, preset=preset, experiment=EXPERIMENT,
               contract_source=flags["contract"], generator_contract=flags["generator_contract"],
               verifier=flags["verifier"], repair_mode=flags["repair_mode"],
               generator_prompt=generator_version(preset), generator_model=MODELS["generator"]["id"],
               agent_model=MODELS["agent"]["id"], agent_prompt_version="a1-v4")
    contract, status = None, "not_used"
    if flags["contract"] == "oracle":
        contract, status = task["gold_contract"], "ok"
    elif flags["contract"] == "parsed":
        extracted = parsed.get(task["id"])
        status = "missing" if extracted is None else extracted["status"]
        contract = extracted["contract"] if extracted and extracted["status"] == "ok" else None
    rec["contract_status"] = status
    if flags["contract"] != "none" and contract is None:
        rec.update(stage_ended="parse", outcome="ESCALATE", final_correct=False, initial_correct=None,
                   first_check=None, no_repair_outcome="ESCALATE", no_repair_correct=False, initial_parse_ok=None,
                   attempts=0, tool_calls=0, llm_calls=0, tokens=0, invalid_actions=0, unsafe_attempts=0,
                   unknown_tool_calls=0, initial_blocked=False, escalate_reason=f"contract {status}",
                   path=["PREPARE", "PARSE", "ESCALATED"], tool_trace=[], latency_s=round(time.monotonic()-started, 2))
        return dict(record=rec, trace=None)
    gen_client = old.LLMClient(generator, cache_dir=cache)
    messages = build_generator_messages(dict(request=request, data_file=task["data_file"]), schema,
                                        contract if flags["generator_contract"] else None)
    generated = gen_client.generate(messages, max_new_tokens=METHOD["generator_max_new_tokens"],
                                    temperature=0.0, seed=0)
    extracted = old.extract_code(generated.text)
    initial = extracted["code"] if extracted["parse_ok"] else generated.text
    checker = old.make_checker(flags["verifier"], contract)
    client = old.LLMClient(agent, cache_dir=cache, token_budget=limits.token_budget)
    result = old.run_repair(dict(request=request), initial, before, lambda after: checker(before, after), client,
                            old.AgentConfig(mode=flags["repair_mode"], limits=limits, temperature=0.0),
                            contract_text=json.dumps(contract) if contract and flags["verifier"] == "intent" else None,
                            schema_text=old.schema_text(schema[task["data_file"]]))
    first = old.safe_run(initial, before, **limits.sandbox())
    correct0 = bool(first.ran_ok and old.outputs_equal(reference, first.df_out)[0])
    correct = bool(result.outcome == "ACCEPT" and old.outputs_equal(reference, result.final_output)[0])
    first_status = old.first_check(result.trace)
    without = "ACCEPT" if first_status == "PASS" else "ESCALATE" if first_status == "UNCERTAIN" else "REJECT"
    rec.update(stage_ended="repair" if result.attempts else "check", outcome=result.outcome, final_correct=correct,
               initial_correct=correct0, initial_parse_ok=extracted["parse_ok"], first_check=first_status,
               no_repair_outcome=without, no_repair_correct=without == "ACCEPT" and correct0,
               final_status=result.final_status, attempts=result.attempts, tool_calls=result.tool_calls,
               llm_calls=result.llm_calls + gen_client.n_calls, tokens=result.tokens + gen_client.tokens_used,
               invalid_actions=result.invalid_actions, unsafe_attempts=result.unsafe_attempts,
               unknown_tool_calls=result.unknown_tool_calls, initial_blocked=result.initial_blocked,
               escalate_reason=result.escalate_reason,
               path=["PREPARE"] + (["PARSE"] if contract else []) + result.path,
               tool_trace=[dict(tool=e["tool"], ok=e["ok"], error_type=e["error_type"])
                           for e in result.trace if e["event"] == "tool"],
               latency_s=round(time.monotonic() - started, 2))
    return dict(record=rec, trace=dict(task_id=task["id"], seed=0, generator_reply=generated.text,
                                      initial_code=initial, trace=result.trace))


def run(preset, repo, output, packages=None, backend="hf", task_ids=None):
    if preset not in SYSTEMS or backend not in ("hf", "pipeline_check"):
        raise ValueError("Unknown system/backend")
    repo, output = Path(repo).resolve(), Path(output).resolve()
    sys.path[:0] = [str(repo), str(repo / "src")]
    verify_sources(repo)
    from eval import runner_stage2 as old
    from igt.agents import code_generator_v2, repair_agent
    from igt.sandbox.environment import sandbox_environment
    from . import backend as gpu
    import hashlib
    import yaml

    if code_generator_v2.PROMPT_HASH != METHOD["prompt_hashes"]["b2-v2"]:
        raise RuntimeError("Generator prompt changed")
    if hashlib.sha256(repair_agent.FULL_SYSTEM.encode()).hexdigest()[:12] != METHOD["prompt_hashes"]["a1-v4"]:
        raise RuntimeError("Repair prompt changed")
    defaults = old.Limits()
    for name in ("max_invalid_actions", "sandbox_timeout_s", "sandbox_mem_gb", "sandbox_cpu_s"):
        if getattr(defaults, name) != METHOD[name]:
            raise RuntimeError(f"Frozen agent limit changed: {name}")
    if defaults.max_new_tokens != METHOD["agent_max_new_tokens"]:
        raise RuntimeError("Frozen agent generation limit changed")
    env = gpu.environment() if backend == "hf" else {"pipeline_check": True}
    plan = common_plan(env, backend, task_ids)
    manifest = dict(preset=preset, plan=plan)
    directory = output / preset
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "arm_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing outputs use another configuration; start a fresh notebook session")
    elif any(directory.iterdir()):
        raise ValueError("Output directory is nonempty but has no matching manifest")
    write_json(manifest_path, manifest)
    bundle = output.parent / bundle_name(preset)
    done = {p["record"]["task_id"] for p in task_payloads(directory, manifest)}
    selected = set(plan["selected_test_ids"])
    if done == selected:
        export_arm(directory, bundle, "complete")
        print(f"COMPLETE: {preset} {len(done)}/{len(selected)} (already saved)", flush=True)
        return directory
    export_arm(directory, bundle)
    active = None

    def record(event):
        item = dict(task_id=active, time=time.time(), **{k: v for k, v in event.items() if k not in ("task_id", "time")})
        if event.get("task_id") is not None:
            item["task_id"] = event["task_id"]
        with (directory / "diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
            stream.flush()
        if event["event"] in ("kernel_self_test_passed", "model_loaded", "generation_failed", "attention_failure"):
            print("GPU:", json.dumps(item), flush=True)

    try:
        tasks = {t["id"]: t for p in sorted((repo / "benchmark/tasks").glob("*/*.yaml"))
                 if (t := yaml.safe_load(p.read_text()))["split"] == "test"}
        if set(tasks) != set(TEST_IDS):
            raise ValueError("Test task inventory changed")
        schema = old.load_prompt_schema(repo)
        data = {f: old.pd.read_csv(repo / f) for f in {tasks[tid]["data_file"] for tid in selected}}
        parsed = {}
        if preset in PARSES:
            rows = old.load_jsonl(repo / PARSES[preset])
            ids = [r["task_id"] for r in rows if r["split"] == "test"]
            if len(ids) != len(TEST_IDS) or set(ids) != set(TEST_IDS):
                raise ValueError("Incomplete/duplicate parser inputs")
            parsed = {r["task_id"]: old.extract_contract(r["raw_output"]) for r in rows}
        with sandbox_environment(packages) as sandbox:
            write_json(directory / "sandbox.json", sandbox)
            if backend == "hf":
                gpu.preflight(record)
                generator = gpu.Backend("generator", record)
                agent = gpu.Backend("agent", record) if preset != "baseline1" else old.ScriptedBackend([], model="unused")
            cache = output.parent / "cache" / digest(canonical(plan))[:16] / preset
            for task_id in ORDER:
                if task_id not in selected or task_id in done:
                    continue
                active = task_id
                task = tasks[task_id]
                print(f"TASK START: {preset} {task_id} ({len(done)}/{len(selected)} saved)", flush=True)
                before = data[task["data_file"]]
                reference = old.safe_run(task["reference_code"], before, **old.Limits().sandbox())
                if not reference.ran_ok:
                    raise RuntimeError(f"Reference evaluator failed for {task_id}: {reference.status}: {reference.error}")
                if backend == "pipeline_check":
                    generator = old.ScriptedBackend([old.NOOP_CODE], model="pipeline_check_generator")
                    agent = old.ScriptedBackend([
                        json.dumps({"action": "tool", "tool": "diff_datasets", "args": {}}),
                        '{"action": "propose_fix"}\n```python\n' + task["reference_code"] + '```'] * 5,
                        model="pipeline_check_agent")
                else:
                    generator.task_id = task_id
                    agent.task_id = task_id
                payload = evaluate(task, preset, before, reference.df_out, schema, parsed, generator, agent, cache)
                # Trace and result commit together; interruption cannot leave a half-written JSONL record.
                write_json(directory / "tasks" / f"{task_id}__0.json", payload)
                done.add(task_id)
                export_arm(directory, bundle, "running")
                print(f"TASK DONE: {preset} {len(done)}/{len(selected)} {task_id} {payload['record']['outcome']}", flush=True)
                if backend == "hf":
                    gc.collect()
                    generator.torch.cuda.empty_cache()
        export_arm(directory, bundle, "complete")
        print(f"COMPLETE: {preset} {len(done)}/{len(selected)} -> {bundle}", flush=True)
        return directory
    except BaseException as error:
        write_json(directory / "failure.json", dict(task_id=active, error_type=type(error).__name__,
                                                    message=str(error), traceback=traceback.format_exc()))
        try:
            export_arm(directory, bundle, "failed")
        except Exception as export_error:
            print(f"Checkpoint export failed; retaining last safe ZIP: {export_error}", file=sys.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=SYSTEMS)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    args = parser.parse_args()
    run(args.preset, args.repo, args.out, args.packages)


if __name__ == "__main__":
    main()
