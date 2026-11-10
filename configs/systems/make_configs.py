"""Writes configs/systems/*.yaml (extension #1). The configs only write down the commands the project already uses;
tests/extensions/test_configs.py checks that the Stage 2 ones equal runner_stage2.PRESETS and that every runner and
flag exists. Rerun after changing this file:  .venv/bin/python configs/systems/make_configs.py
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
K = "/kaggle/working"
AGENT = dict(backend="hf", model="Qwen/Qwen2.5-7B-Instruct", **{"load-in-4bit": True}, verifier="oracle",
             splits=["test"], categories=["R", "C", "T", "E", "S"], **{"cache-dir": f"{K}/llm_cache"},
             out=f"{K}/results/stage2_agent")
S2 = dict(backend="hf", **{"generator-model": "Qwen/Qwen2.5-3B-Instruct", "agent-model": "Qwen/Qwen2.5-7B-Instruct",
                           "agent-4bit": True}, splits=["test"], seeds=[0], **{"cache-dir": f"{K}/llm_cache"},
          out=f"{K}/results/stage2")
B3 = "kaggle_outputs/parsing/7b_p1-v3/parses.jsonl"
FT = "kaggle_outputs/parsing/ft_1_5b/parses.jsonl"
STAGE2_PRESETS = {
    "baseline1": {}, "baseline2_repair": {}, "baseline3_repair": dict(parses=B3), "gold_contract": {},
    "proposed": dict(parses=FT), "no_contract_to_generator": dict(parses=FT),
}

C = {}
# ---- Stage 1 (CPU; what reproduce_all.sh runs) ----
C["stage1_baseline1"] = dict(system="baseline1", runner="eval/baseline1_by_dataset.py", needs="cpu",
                             description="Baseline 1 (execution success) by dataset")
C["stage1_baseline2"] = dict(system="baseline2", runner="eval/baseline2.py", needs="cpu",
                             args=dict(config="configs/baseline2_v4.yaml"), description="Baseline 2, conventional checks")
C["stage1_oracle"] = dict(system="intent_oracle", runner="eval/run_intent_oracle.py", needs="cpu",
                          description="intent verifier with gold contracts")
C["stage1_score_parsers"] = dict(system="baseline3 + parser ablations", runner="eval/score_parsers.py", needs="cpu",
                                 description="score every kaggle_outputs/parsing/* run (Baseline 3 = intent_prompted_7b_p1v3)")
# ---- parser runs (GPU) ----
for size, model, v, q in [("7b", "Qwen/Qwen2.5-7B-Instruct", "p1-v3", True),
                          ("3b", "Qwen/Qwen2.5-3B-Instruct", "p1-v2", False),
                          ("1_5b", "Qwen/Qwen2.5-1.5B-Instruct", "p1-v3", False)]:
    C[f"parse_{size}_{v.replace('-', '')}"] = dict(
        system=f"intent_prompted_{size}_{v.replace('-', '')}", runner="eval/parse_contracts.py", needs="gpu",
        args={"model": model, "load-in-4bit": q, "prompt-version": v, "out": f"{K}/parse_{size}_{v}"})
C["parse_tools_7b_t1v1"] = dict(system="intent_tools_7b_t1v1", runner="eval/parse_contracts_tools.py", needs="gpu",
                                args={"model": "Qwen/Qwen2.5-7B-Instruct", "load-in-4bit": True,
                                      "out": f"{K}/parse_tools_7b", "cache-dir": f"{K}/llm_cache"})
for size in ("1_5b", "3b"):
    C[f"finetune_{size}"] = dict(system=f"intent_ft_{size}", runner="finetune/train_qlora.py", needs="gpu",
                                 args=dict(config=f"finetune/configs/qwen25_{size}.yaml"))
# ---- agent systems (GPU) ----
for name, mode in [("agent_full", "full"), ("agent_feedback", "feedback"), ("agent_naive", "naive")]:
    C[name] = dict(system=name, runner="eval/runner_agent.py", needs="gpu", args=dict(system=name, mode=mode, **AGENT))
# ---- Stage 2 systems (GPU), one per preset ----
for p, extra in STAGE2_PRESETS.items():
    C[f"stage2_{p}"] = dict(system=p, runner="eval/runner_stage2.py", needs="gpu", args=dict(preset=p, **extra, **S2))

# ---- extensions (docs/extensions.md) ----
C["ext_pc1_precheck"] = dict(system="<parser>_pc1", extension="pc1", runner="eval/ext_precheck.py", needs="cpu",
                             positional=["7b_p1-v3", "3b_p1-v2", "1_5b_p1-v3"])
C["ext_rq1_7b"] = dict(system="intent_prompted_7b_p1v3_rq1", extension="rq1", runner="eval/ext_reask.py", needs="gpu",
                       positional=["7b_p1-v3"], args={"backend": "hf", "model": "Qwen/Qwen2.5-7B-Instruct",
                                                      "load-in-4bit": True, "out-root": f"{K}/ext"})
C["ext_pc1rq1_7b"] = dict(system="intent_prompted_7b_p1v3_pc1rq1", extension="pc1rq1", runner="eval/ext_reask.py",
                          needs="gpu", positional=["7b_p1-v3"],
                          args={"precheck": True, "backend": "hf", "model": "Qwen/Qwen2.5-7B-Instruct",
                                "load-in-4bit": True, "out-root": f"{K}/ext"})
C["ext_np1_replay"] = dict(system="agent_*_np1 (offline)", extension="np1", runner="eval/ext_replay_no_progress.py",
                           needs="cpu", args=dict(ks=[1, 2, 3]))
C["ext_agent_full_np1"] = dict(system="agent_full_np1_k3", extension="np1", runner="eval/runner_agent_ext.py",
                               needs="gpu", args=dict(system="agent_full", mode="full", stop_after_no_progress=3, **AGENT))
C["ext_agent_full_fs1"] = dict(system="agent_full_fs1", extension="fs1", runner="eval/runner_agent_ext.py", needs="gpu",
                               args=dict(system="agent_full", mode="full", stop_after_no_progress=0,
                                         feasibility_check=True, **AGENT))
C["ext_stage2_proposed_np1"] = dict(system="proposed_np1_k3", extension="np1", runner="eval/runner_stage2_ext.py",
                                    needs="gpu", args=dict(preset="proposed", parses=FT, stop_after_no_progress=3, **S2))
C["ext_lv1_7b"] = dict(system="llm_verifier_7b_lv1", extension="lv1", runner="eval/run_llm_verifier.py", needs="gpu",
                       args={"backend": "hf", "model": "Qwen/Qwen2.5-7B-Instruct", "load-in-4bit": True,
                             "out": f"{K}/ext/llm_verifier_7b_lv1", "cache-dir": f"{K}/llm_cache"})
C["ext_ui_demo"] = dict(system="demo UI", extension="ui", runner="app/demo_app.py", needs="cpu",
                        notes="the Live tab needs a GPU (notebooks/kaggle/demo_ui_kaggle.ipynb)")
C["ext_docker_redteam"] = dict(system="docker sandbox, red-team set", extension="docker",
                               runner="eval/ext_docker_check.py", needs="docker")
for size in ("1_5b", "3b"):
    C[f"ext_ft_tools_{size}_data"] = dict(system=f"intent_ft_tools_{size}", extension="ft-t1",
                                          runner="finetune/data_gen/build_tools_data.py", needs="cpu")
    C[f"ext_ft_tools_{size}_train"] = dict(system=f"intent_ft_tools_{size}", extension="ft-t1",
                                           runner="finetune/train_qlora_tools.py", needs="gpu",
                                           args=dict(config=f"finetune/configs/qwen25_{size}_tools.yaml"))
    C[f"ext_ft_tools_{size}_parse"] = dict(
        system=f"intent_ft_tools_{size}", extension="ft-t1", runner="eval/parse_contracts_tools_ft.py", needs="gpu",
        args={"model": f"Qwen/Qwen2.5-{'1.5B' if size == '1_5b' else '3B'}-Instruct",
              "adapter": f"finetune/runs/qwen25_{size}_tools/adapter", "out": f"{K}/parse_ft_tools_{size}",
              "cache-dir": f"{K}/llm_cache"})

if __name__ == "__main__":
    for name, cfg in C.items():
        (HERE / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{len(C)} configs in {HERE}")
