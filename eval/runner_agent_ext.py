"""Agent runs with the opt-in extensions (np1 no-progress stop, fs1 feasibility step, vllm backend). Every other flag
is passed unchanged to eval/runner_agent.py, which is not edited.

    # Kaggle GPU: full agent, stop after 3 no-progress attempts (the default; any K with --stop-after-no-progress K)
    python eval/runner_agent_ext.py --system agent_full --mode full --backend hf --model Qwen/Qwen2.5-7B-Instruct \\
        --load-in-4bit --out results/stage2_agent                       # -> results/stage2_agent/agent_full_np1_k3/
    # + feasibility step, on vLLM
    python eval/runner_agent_ext.py --system agent_full --mode full --backend vllm --feasibility-check ...
                                                                        # -> agent_full_np1_k3_fs1_vllm/
    # pipeline check on the laptop (NOT a result)
    python eval/runner_agent_ext.py --system smoke --mode full --backend reference_policy --limit 6 --out /tmp/agent

Extension flags: --stop-after-no-progress K (default 3; 0 = off), --feasibility-check, --backend vllm.
The system name gets the extension tags appended (np1_k<K>, fs1, vllm), so runs with different settings never share
a folder, and the recorded agent version is e.g. a1-v4+np1-k3 / a1-v4-fs1+np1-k3+vllm (the runner refuses to mix
versions in one folder). Settings also go to <out>/<system>/run_info_ext.json.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval import runner_agent  # noqa: E402
from eval.ext_runner_patch import (ext_arg_parser, get_flag, patch_runner, replace_flag, tags, with_tags,  # noqa: E402
                                   write_ext_info)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ext, rest = ext_arg_parser().parse_known_args(argv)
    if ext.ext_help:
        ext_arg_parser().print_help()
        return None
    ext.vllm = get_flag(rest, "--backend") == "vllm"
    if ext.vllm:
        rest = replace_flag(rest, "--backend", "hf")
    system = get_flag(rest, "--system")
    if not system:
        raise SystemExit("--system is required")
    t = tags(ext)
    if not t:
        print("note: no extension switched on; this is the a1-v4 agent (use eval/runner_agent.py for that)")
    system = with_tags(system, t)
    rest = replace_flag(rest, "--system", system)
    version = patch_runner(runner_agent, ext, vllm=ext.vllm)
    out = Path(get_flag(rest, "--out"))
    sys.argv = ["runner_agent.py", *rest]
    runner_agent.main()
    write_ext_info(out / system, system=system, agent_version=version,
                   stop_after_no_progress=ext.stop_after_no_progress or None,
                   feasibility_check=ext.feasibility_check, backend="vllm" if ext.vllm else get_flag(rest, "--backend"),
                   argv=argv)
    return out / system


if __name__ == "__main__":
    main()
