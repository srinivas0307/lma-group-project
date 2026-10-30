"""Every project folder in one place. Scripts and tests take paths from here, so the layout is changed in one file.

    benchmark/      frozen inputs: tasks, prompt schema, candidate pool, agent scenarios, red-team samples
    kaggle_outputs/ downloads from Kaggle (generation, parsing, agent zips) + their local labels (candidates.jsonl)
    results/        everything computed from them: results/stage1/<system>/, results/stage2_agent/<system>/
    archive/        superseded pools (v1, v2), their results and old configs, kept unchanged
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BENCHMARK = ROOT / "benchmark"
TASKS = BENCHMARK / "tasks"                         # <dataset>/<task_id>.yaml
PROMPTS = BENCHMARK / "prompts"                     # schema_b1-v1.json (frozen dataset schema in the prompts)
POOL_VERSION = "v4"                                 # v4: built from the two all-task Baseline 1 runs (3B, 1.5B)
POOL_DIR = BENCHMARK / "pool"                       # pool_v4.jsonl, manifest.json, outputs/*.pkl
POOL = POOL_DIR / f"pool_{POOL_VERSION}.jsonl"
AGENT_SCENARIOS = BENCHMARK / "agent_scenarios"     # rct_v4.jsonl (from pool v4; T faults on the first tool call), E/, S/
RCT = AGENT_SCENARIOS / "rct_v4.jsonl"
REDTEAM = BENCHMARK / "redteam"

KAGGLE = ROOT / "kaggle_outputs"
GENERATION = KAGGLE / "generation"                  # 3b_all_tasks, 1_5b_all_tasks (the pool v4 sources)
PARSING = KAGGLE / "parsing"                        # <model>_<prompt version>/parses.jsonl + system.txt

RESULTS = ROOT / "results"
STAGE1 = RESULTS / "stage1"
STAGE1_SYSTEM_DIRS = {                              # system name -> results folder
    "baseline1": STAGE1 / "baseline1",
    "baseline2": STAGE1 / "baseline2",
    "intent_oracle": STAGE1 / "oracle",
    "intent_prompted_7b_p1v3": STAGE1 / "baseline3",
    "intent_prompted_3b_p1v2": STAGE1 / "size_ablation" / "intent_prompted_3b_p1v2",
    "intent_prompted_1_5b_p1v3": STAGE1 / "size_ablation" / "intent_prompted_1_5b_p1v3",
}
STAGE1_SUMMARY = STAGE1 / "summary"
AGENT_RESULTS = RESULTS / "stage2_agent"

ARCHIVE = ROOT / "archive"
CONFIGS = ROOT / "configs"


def stage1_dir(system):
    """Results folder of a Stage 1 system. Parser runs that were not selected go to stage1/other_prompts/<system>;
    fine-tuned parsers (intent_ft_*) to stage1/finetuned/<system>."""
    if system in STAGE1_SYSTEM_DIRS:
        return STAGE1_SYSTEM_DIRS[system]
    if system.startswith("intent_ft_"):
        return STAGE1 / "finetuned" / system
    return STAGE1 / "other_prompts" / system


def rel(p):
    """Path relative to the project root, for printing and for files written into results."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)
