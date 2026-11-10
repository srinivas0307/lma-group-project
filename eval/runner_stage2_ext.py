"""Stage 2 runs with the opt-in extensions. Every other flag goes unchanged to eval/runner_stage2.py (not edited).

    # proposed system, agent stops after 3 no-progress attempts (default), contracts pre-checked + re-asked (#9 + #8)
    python eval/runner_stage2_ext.py --preset proposed \\
        --parses results/stage1/extensions/intent_ft_1_5b_pc1rq1/parses.jsonl \\
        --backend hf --generator-model Qwen/Qwen2.5-3B-Instruct --agent-model Qwen/Qwen2.5-7B-Instruct --agent-4bit \\
        --out results/stage2                       # -> results/stage2/proposed_pc1rq1_np1_k3/
    # pipeline check on the laptop (NOT a result)
    python eval/runner_stage2_ext.py --preset gold_contract --backend pipeline_check --limit 3 --out /tmp/stage2

Extensions:
    np1   --stop-after-no-progress K (default 3; 0 = off)
    fs1   --feasibility-check
    vllm  --backend vllm (generator and agent)
    rq1 / pc1 / pc1rq1   contracts from an extension parse folder (eval/ext_reask.py, eval/ext_precheck.py): pass its
          parses.jsonl as --parses; the folder's tag (_rq1, _pc1, _pc1rq1) is added to the system name.
The system name (default: the preset) gets every tag appended, so extension runs never share a folder with the
existing systems; the recorded agent version (e.g. a1-v4+np1-k3) enters the run's config signature, which the runner
already refuses to mix. Settings also go to <out>/<system>/run_info_ext.json.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval import runner_stage2  # noqa: E402
from eval.ext_runner_patch import (ext_arg_parser, get_flag, patch_runner, replace_flag, tags, with_tags,  # noqa: E402
                                   write_ext_info)


def parse_tag(parses):
    if not parses:
        return []
    name = Path(parses).parent.name
    for t in ("pc1rq1", "rq1", "pc1"):
        if name.endswith("_" + t):
            return [t]
    return []


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ext, rest = ext_arg_parser().parse_known_args(argv)
    if ext.ext_help:
        ext_arg_parser().print_help()
        return None
    ext.vllm = get_flag(rest, "--backend") == "vllm"
    if ext.vllm:
        rest = replace_flag(rest, "--backend", "hf")
    system = get_flag(rest, "--system") or get_flag(rest, "--preset")
    if not system:
        raise SystemExit("--preset is required")
    t = parse_tag(get_flag(rest, "--parses")) + tags(ext)
    if not t:
        print("note: no extension switched on; this equals eval/runner_stage2.py")
    system = with_tags(system, t)
    rest = replace_flag(rest, "--system", system)
    version = patch_runner(runner_stage2, ext, vllm=ext.vllm)
    out_dir = runner_stage2.main(rest)
    write_ext_info(out_dir, system=system, agent_version=version,
                   stop_after_no_progress=ext.stop_after_no_progress or None, feasibility_check=ext.feasibility_check,
                   backend="vllm" if ext.vllm else get_flag(rest, "--backend"), contracts=get_flag(rest, "--parses"),
                   argv=argv)
    return out_dir


if __name__ == "__main__":
    main()
