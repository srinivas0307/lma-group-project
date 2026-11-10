"""Shared by eval/runner_agent_ext.py and eval/runner_stage2_ext.py: switch a runner module to the extension agent
(np1, fs1) and/or the vLLM backend for one process, without editing the runner. The existing runners keep their
code and defaults; without these wrappers nothing changes.
"""
import argparse
import functools
import json
from pathlib import Path

from igt.agents.repair_agent_ext import DEFAULT_NO_PROGRESS_K, AgentConfigExt, ext_version, run_repair_ext


def ext_arg_parser():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stop-after-no-progress", type=int, default=DEFAULT_NO_PROGRESS_K, metavar="K",
                    help=f"np1: stop (ESCALATE) after K consecutive no-progress attempts; default {DEFAULT_NO_PROGRESS_K}, "
                         "0 = off")
    ap.add_argument("--feasibility-check", action="store_true", help="fs1: one 'can this be done?' call before repair")
    ap.add_argument("--ext-help", action="store_true", help="show these extension flags and exit")
    return ap


def tags(ext):
    out = []
    if ext.stop_after_no_progress:
        out.append(f"np1_k{ext.stop_after_no_progress}")
    if ext.feasibility_check:
        out.append("fs1")
    if getattr(ext, "vllm", False):
        out.append("vllm")
    return out


def with_tags(system, tag_list):
    for t in tag_list:
        if t not in system:
            system += "_" + t
    return system


def replace_flag(argv, flag, value):
    """--flag X / --flag=X -> value (append if missing)."""
    argv = list(argv)
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            argv[i + 1] = value
            return argv
        if a.startswith(flag + "="):
            argv[i] = f"{flag}={value}"
            return argv
    return argv + [flag, value]


def get_flag(argv, flag, default=None):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def patch_runner(module, ext, vllm=False):
    """Point the runner module's AgentConfig / run_repair / AGENT_PROMPT_VERSION to the extension agent, and (vllm)
    igt.llm.hf.HFChatBackend to VLLMChatBackend. Returns the recorded agent version."""
    k = ext.stop_after_no_progress or None
    cfg_cls = functools.partial(AgentConfigExt, stop_after_no_progress=k, feasibility_check=ext.feasibility_check)
    version = ext_version(AgentConfigExt(stop_after_no_progress=k, feasibility_check=ext.feasibility_check))
    if vllm:
        import igt.llm.hf as hf
        from igt.llm.vllm_backend import VLLMChatBackend
        hf.HFChatBackend = VLLMChatBackend               # the runners import it inside main()
        version += "+vllm"
    module.AgentConfig = cfg_cls
    module.run_repair = run_repair_ext
    module.AGENT_PROMPT_VERSION = version
    return version


def write_ext_info(out_dir, **info):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info_ext.json").write_text(json.dumps(info, indent=2, default=str))
