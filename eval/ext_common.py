"""Shared helpers for the optional extensions (docs/extensions.md). Nothing in the existing systems imports this.

Extension outputs never go to an existing results folder:
    Stage 1 variants   results/stage1/extensions/<system>/
    agent variants     results/stage2_agent/<system>/        (system names carry the extension id, e.g. agent_full_np1_k3)
    Stage 2 variants   results/stage2/<system>/
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.paths import PARSING, RESULTS, STAGE1  # noqa: E402

EXT_STAGE1 = STAGE1 / "extensions"
EXT_SUMMARY = EXT_STAGE1 / "summary"
EXT_IDS = ("rq1", "pc1", "pc1rq1", "np1", "fs1", "cfg", "vllm", "lv1", "ui", "docker", "ft-t1")


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def ext_dir(system):
    """Results folder of a Stage 1 extension system; refuses names that would point into an existing folder."""
    if not any(tok in system for tok in ("_rq1", "_pc1", "intent_ft_tools_", "llm_verifier_", "_vllm")):
        raise SystemExit(f"{system}: an extension system name must carry its extension id (e.g. _pc1, _rq1)")
    return EXT_STAGE1 / system


def parse_folder(arg):
    """'7b_p1-v3' or a path -> (folder, base system name from system.txt)."""
    d = Path(arg)
    if not d.exists():
        d = PARSING / arg
    if not (d / "parses.jsonl").exists():
        raise SystemExit(f"{arg}: no parses.jsonl (looked in {d})")
    st = d / "system.txt"
    system = st.read_text().strip() if st.exists() else d.name
    return d, system


def tasks_and_data():
    import pandas as pd
    import yaml
    tasks, data = {}, {}
    for f in sorted((ROOT / "benchmark" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        tasks[t["id"]] = t
        if t["data_file"] not in data:
            data[t["data_file"]] = pd.read_csv(ROOT / t["data_file"])
    return tasks, data


__all__ = ["ROOT", "RESULTS", "EXT_STAGE1", "EXT_SUMMARY", "load_jsonl", "write_jsonl", "ext_dir", "parse_folder",
           "tasks_and_data"]
