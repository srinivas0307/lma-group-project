"""Build the Stage 2 (fresh) notebooks from cells/*.py.

Both pipeline notebooks share every cell except the configuration and the INTENT PARSER cell, so a
side-by-side diff of the two .ipynb files shows exactly the parser difference.

    python stage2_fresh/build_notebooks.py
"""
import json
import zipfile
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CELLS = HERE / "cells"
OUT = HERE / "notebooks"
UPLOAD = HERE / "kaggle_upload"

VARIANTS = {
    "selective_facts": {
        "file": "01_stage2_proposed_selective_facts.ipynb",
        "title": "Stage 2 proposed system — parser A: fine-tuned 1.5B **selective facts**",
        "protocol": "p1-v2+sf1",
        "prompt": "system_selective_facts.txt",
        "parser_cell": "c09_parser_selective_facts.py",
        "how": ("A deterministic interpreter reads the request, computes **only** the input statistics it needs "
                "(median / mean / mode — full column, one equality group, or one fact per group) and inserts them into the "
                "prompt as `Required input facts (sf1)`. The adapter writes the contract in one reply, without tools. "
                "If the request needs a statistic the interpreter cannot compute unambiguously (for example a tied mode), the task "
                "is withheld before inference (`fact_unavailable`)."),
        "adapter_hint": "`selective_facts_1_5b.zip` (training output; contains `adapter/` with `igt_prompt_version.txt` = `p1-v2+sf1`)",
    },
    "tool_use": {
        "file": "02_stage2_proposed_tool_use.ipynb",
        "title": "Stage 2 proposed system — parser B: fine-tuned 1.5B **tool use**",
        "protocol": "parser-tools-v2",
        "prompt": "system_tool_use.txt",
        "parser_cell": "c09_parser_tool_use.py",
        "how": ("The adapter sees only the schema and three rows and **decides itself** whether to query the input data with "
                "read-only tools (`profile_column`, `count_rows`, `inspect_rows`, `group_profile`) before writing the contract, "
                "using the multi-turn JSON protocol it was trained on (at most 6 tool attempts, 10 replies, 24k parser tokens). "
                "It may also escalate with `{\"action\": \"escalate\"}`."),
        "adapter_hint": "`tool_use_train_1_5b.zip` (training output; contains `adapter/` with `igt_prompt_version.txt` = `parser-tools-v2`)",
    },
}

ARCH = r"""
```
User request + input CSV
        │
        ▼
┌──────────────────────────────┐   feedback loop (≤ MAX_PARSER_FEEDBACK rounds):
│ INTENT PARSER  (1.5B + LoRA) │◄── schema errors · unknown columns/values ·
│  ← this notebook's variant   │    contract already satisfied by unchanged data ·
└──────────────┬───────────────┘    contract not checkable on this data
               ▼
        Intent contract  ──►  must_change  /  must_preserve
               ▼
┌──────────────────────────────┐
│ PLANNER + GENERATOR (3B fp16)│  plan as comments + def transform(df)
└──────────────┬───────────────┘
               ▼
   Safety checker (AST) ─► Sandbox execute (subprocess, timeout, memory cap)
               ▼
   Before/after profiler ─► Intent verifier (deterministic, 11 predicates)
        ┌──────┼──────────────┐
      PASS  VIOLATION /     UNCERTAIN
        │   crash / unsafe     │
     ACCEPT    ▼            WITHHOLD (flag)
        ┌──────────────────────────────┐
        │ REPAIR AGENT (3B fp16)       │  tools: inspect_rows, column_summary
        │ diagnose → fix → re-execute  │  ≤ MAX_REPAIR_ATTEMPTS, then WITHHOLD
        │ → re-verify                  │
        └──────────────────────────────┘
```
"""

SECTIONS = [
    ("c02_setup.py", "## 1. Setup and input discovery\nInstalls `peft`/`bitsandbytes` if missing and finds the attached datasets automatically."),
    ("c03_tasks.py", "## 2. Benchmark tasks\nThe system sees only `request` and the input CSV. `reference_code` and `gold_contract` are used **only by the evaluator**."),
    ("c04_contract.py", "## 3. Intent contract: extraction, schema validation, condition semantics\nWritten fresh from the contract specification (closed predicate vocabulary, null-safe conditions, value equality with `atol = 1e-9`)."),
    ("c05_verifier.py", "## 4. Before/after profiler and deterministic intent verifier\n`PASS` → accept, `VIOLATION` → repair, `UNCERTAIN` → flag (withhold)."),
    ("c06_sandbox.py", "## 5. Safety checker and sandbox"),
    ("c13_evaluate.py", "## 6. Evaluator (answer key)\nCompares the final output with the reference output and the predicted contract with the gold contract. Never visible to the system."),
    ("c13b_selftest.py", "## 7. Verifier self-test (CPU)\nRuns before any model is loaded: every gold contract must accept its reference output and reject the unchanged input."),
    ("c08_parser_common.py", "## 8. Parser input format and contract feedback checks (shared by both parser variants)"),
    ("c07_llm.py", "## 9. Load models\nParser: Qwen2.5-1.5B-Instruct NF4 + the fine-tuned LoRA adapter on GPU 0. Generator **and** repair agent: one Qwen2.5-3B-Instruct FP16 on GPU 1."),
    ("__PARSER__", "## 10. INTENT PARSER — the only cell that differs between the two notebooks"),
    ("c10_generator.py", "## 11. Transformation planner + code generator (3B)"),
    ("c11_repair.py", "## 12. Repair agent (3B, tool-using, bounded)"),
    ("c12_pipeline.py", "## 13. Stage 2 pipeline and metrics"),
    ("c14_run.py", "## 14. Run: smoke → dev (20) → test (100)\nResumable: each finished task is saved as JSON and skipped if the cell is re-run."),
    ("c15_package.py", "## 15. Package results"),
]


def config_cell(key, v):
    return f'''# ===== Configuration =====
SYSTEM_NAME = "stage2_proposed_{key}"
PARSER_PROTOCOL = "{v["protocol"]}"          # adapter is auto-detected by this protocol string
PARSER_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
PARSER_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"   # base revision the adapter was trained on
PARSER_ADAPTER = None                          # None = auto-detect under /kaggle/input; or a path to the adapter/ folder
CODER_BASE = "Qwen/Qwen2.5-3B-Instruct"       # code generator AND repair agent (FP16)

DATA_ROOT = None                               # None = auto-detect (folder containing benchmark/tasks and data/)
INPUT_SEARCH_ROOTS = ["/kaggle/input"]
WORK_DIR = "/kaggle/working"
RUN_NAME = "stage2_fresh_{key}_v3"

RUN_MODELS = True
RUN_SMOKE = True                               # 2 dev tasks first, written to RUN_NAME/smoke
SMOKE_TASKS = 2
RUN_SPLITS = ["dev", "test"]                   # 20 dev + 100 test tasks
MAX_TASKS_PER_SPLIT = None                     # e.g. 5 for a quick trial

MAX_PARSER_FEEDBACK = 2                        # parser correction rounds (format / grounding / vacuity)
MAX_REPAIR_ATTEMPTS = 3                        # repair → re-execute → re-verify cycles
MAX_REPAIR_TOOL_CALLS = 3                      # inspection tool calls per repair attempt
PARSER_MAX_NEW_TOKENS = 1024
GEN_MAX_NEW_TOKENS = 1024
REPAIR_MAX_NEW_TOKENS = 1024
EXEC_TIMEOUT_S = 60
REPETITION_PENALTY = 1.0                       # greedy decoding, no repetition penalty (keeps JSON intact)
RETRY_TEMPERATURE = 0.7                        # seeded sampling, used only for parser retries that would otherwise repeat
ATTN_IMPL = "sdpa"                             # use "eager" if your GPU/driver has SDPA problems
'''


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_variant(key, v):
    other = [k for k in VARIANTS if k != key][0]
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    intro = f"""# {v["title"]}

Fresh implementation of the **Stage 2 proposed architecture** (no code reused from earlier runs).
The only component that differs from the companion notebook (`{VARIANTS[other]["file"]}`) is the **intent parser** (section 10) and its configuration.

**This notebook's parser:** {v["how"]}

**Shared by both notebooks:** contract feedback loop to the parser · Qwen2.5-3B-Instruct FP16 planner/generator · AST safety checker · subprocess sandbox · deterministic verifier · tool-using 3B repair agent (≤ 3 attempts) · identical evaluator and metrics.
{ARCH}
### How to run on Kaggle
1. Settings: **GPU T4 x2**, **Internet on**.
2. Attach the datasets: **`stage2_fresh_data.zip`** (benchmark tasks + 4 CSVs) and {v["adapter_hint"]}. Paths are detected automatically.
3. **Run All**. A smoke run (2 dev tasks) runs first, then 20 dev and 100 test tasks. Re-running resumes where it stopped.
4. Download **`/kaggle/working/stage2_fresh_{key}_v3.zip`** and compare both runs with `03_compare_parsers.ipynb`.

Outcome per task: **correct accepted** (verifier PASS and output equals the reference) · **incorrect accepted** (silent error) · **withheld** (parser escalation/invalid, verifier UNCERTAIN, or repair budget exhausted).
"""
    nb.cells = [md(intro), md("## 0. Configuration"), code(config_cell(key, v))]
    for fname, header in SECTIONS:
        nb.cells.append(md(header))
        if fname == "__PARSER__":
            src = (CELLS / v["parser_cell"]).read_text()
            prompt = (HERE / "prompts" / v["prompt"]).read_text()
            src = src.replace("__SYSTEM_PROMPT__", repr(prompt))
            nb.cells.append(md(f"**Variant: `{key}` — protocol `{v['protocol']}`.** {v['how']}\n\n"
                               "The system prompt below is the exact prompt the adapter was fine-tuned with."))
        else:
            src = (CELLS / fname).read_text()
        nb.cells.append(code(src))
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / v["file"])
    return OUT / v["file"]


def build_compare():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [md("""# Stage 2 proposed system — parser A (selective facts) vs parser B (tool use)

CPU only. Attach (or place next to this notebook) the two result ZIPs produced by notebooks 01 and 02:
`stage2_fresh_selective_facts.zip` and `stage2_fresh_tool_use.zip`. Everything except the intent parser is identical
between the runs, so every difference below comes from the parser."""),
                code((CELLS / "c16_compare.py").read_text())]
    nbf.write(nb, OUT / "03_compare_parsers.ipynb")
    return OUT / "03_compare_parsers.ipynb"


def build_data_bundle():
    UPLOAD.mkdir(exist_ok=True)
    out = UPLOAD / "stage2_fresh_data.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted((ROOT / "benchmark" / "tasks").rglob("*")):
            if f.is_file() and f.suffix in (".yaml", ".md"):
                zf.write(f, f.relative_to(ROOT))
        for name in ("adult_dev.csv", "airbnb_dev.csv", "Titanic-Dataset.csv", "yellow_tripdata_2026-01.csv"):
            zf.write(ROOT / "data" / name, f"data/{name}")
    return out


if __name__ == "__main__":
    for k, v in VARIANTS.items():
        print("wrote", build_variant(k, v))
    print("wrote", build_compare())
    print("wrote", build_data_bundle())
