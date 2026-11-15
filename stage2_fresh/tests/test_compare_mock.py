"""Simulate two small runs with fake LLMs, then execute the comparison cell on them."""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness
from test_pipeline_mock import load_variant, FakeLM, fence
SCR = Path("/tmp/claude-1000/-home-srinivas-billuri-4-1-LMA-intent-parser/7fa0dbd3-d71d-47bf-8585-883ceb58c1ab/scratchpad/compare_sim")
shutil.rmtree(SCR, ignore_errors=True)
ids = ["adult_filt_001", "titanic_miss_004", "airbnb_dedup_001", "taxi_conv_001", "adult_conv_003", "airbnb_col_005"]
for kind in ("selective_facts", "tool_use"):
    ns = load_variant(kind)
    tasks = {t["id"]: t for t in ns["ALL_TASKS"]}
    out = SCR / f"stage2_fresh_{kind}_v3" / "tasks"; out.mkdir(parents=True)
    for i in ids:
        t = tasks[i]
        g = t["gold_contract"]
        if kind == "tool_use" and i == "airbnb_col_005":
            ns["PARSER_LM"] = FakeLM(lambda m, tl: '{"action": "escalate", "reason": "x"}')
        else:
            ns["PARSER_LM"] = FakeLM(lambda m, tl, g=g: fence(g))
        ns["CODER_LM"] = FakeLM(lambda m, tl, t=t: "```python\n" + t["reference_code"] + "\n```")
        r = ns["run_task"](t)
        (out / f"{i}.json").write_text(json.dumps(r, default=str))
cns = {"display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x)}
src = (harness.HERE / "cells" / "c16_compare.py").read_text().replace(
    'SEARCH = [Path("/kaggle/input"), Path("."), Path("/kaggle/working")]', f'SEARCH = [Path("{SCR}")]')
exec(src, cns)
