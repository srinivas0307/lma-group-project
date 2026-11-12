"""Versioned Stage 2 prompt comparison with verified sandbox packages.

Existing runners and prompt defaults remain unchanged. Example (CPU wiring
check, NOT a model-quality result):
    python eval/runner_stage2_prompts.py --preset gold_contract --splits dev \
        --generator-prompt b2-v2 --agent-prompt a1-v4 --backend pipeline_check \
        --limit 2 --out /tmp/stage2_prompt_dev
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval import runner_stage2 as runner  # noqa: E402
from igt.agents import code_generator, code_generator_v2, repair_agent, repair_agent_v5  # noqa: E402
from igt.sandbox.environment import sandbox_environment  # noqa: E402
from igt.sandbox.environment import EXPECTED  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--generator-prompt", choices=["b2-v1", "b2-v2"], default="b2-v2")
    parser.add_argument("--agent-prompt", choices=["a1-v4", "a1-v5"], default="a1-v4")
    parser.add_argument("--sandbox-packages", type=Path, default=None)
    parser.add_argument("--prompt-help", action="store_true")
    selected, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if selected.prompt_help:
        parser.print_help()
        return None
    args = runner.parse_args(rest)
    if selected.agent_prompt == "a1-v5" and args.repair_mode not in ("full", "none"):
        parser.error("a1-v5 applies to the full repair agent, not naive/feedback modes")
    gen = code_generator_v2 if selected.generator_prompt == "b2-v2" else code_generator
    agent = repair_agent_v5 if selected.agent_prompt == "a1-v5" else repair_agent
    gen_info = gen.generator_prompt_info(args.generator_contract)
    system = args.system + "__" + gen_info["version"].replace("-", "") + "__" + selected.agent_prompt.replace("-", "") + "__sb1"
    # Appending wins over a previously supplied --system, just as in the existing config runner.
    rest += ["--system", system]
    names = ("build_generator_messages", "generator_prompt_info", "run_repair", "AGENT_PROMPT_VERSION", "config_signature")
    saved = {name: getattr(runner, name) for name in names}
    versions = {}
    for name in ("pandas", "numpy", "torch", "transformers", "bitsandbytes", "peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import numpy as np
    actual = {"pandas": runner.pd.__version__, "numpy": np.__version__}
    if actual != EXPECTED:
        raise RuntimeError(f"parent data libraries are {actual}, expected {EXPECTED}; set PYTHONPATH before starting Python")
    agent_hash = (repair_agent_v5.PROMPT_HASH if selected.agent_prompt == "a1-v5" else
                  hashlib.sha256(repair_agent.FULL_SYSTEM.encode()).hexdigest()[:12])
    try:
        with sandbox_environment(selected.sandbox_packages) as sandbox:
            # These fields enter the resume signature, not only a sidecar file.
            def signature(args, info):
                return dict(saved["config_signature"](args, info), sandbox_environment=sandbox,
                            package_versions=versions, agent_prompt_hash=agent_hash)
            runner.build_generator_messages = gen.build_generator_messages
            runner.generator_prompt_info = gen.generator_prompt_info
            runner.run_repair = agent.run_repair
            runner.AGENT_PROMPT_VERSION = selected.agent_prompt
            runner.config_signature = signature
            print("Verified sandbox:", json.dumps(sandbox, sort_keys=True), flush=True)
            out = runner.main(rest)
            (out / "prompt_run_info.json").write_text(json.dumps(dict(
                generator_prompt=gen_info, agent_prompt=selected.agent_prompt,
                agent_prompt_hash=agent_hash, sandbox_environment=sandbox,
                package_versions=versions), indent=2))
            return out
    finally:
        for name, value in saved.items():
            setattr(runner, name, value)


if __name__ == "__main__":
    main()
