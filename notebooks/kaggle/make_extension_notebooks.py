"""Writes the 7 extension notebooks (docs/extensions.md). Existing notebooks are not touched.
    .venv/bin/python notebooks/kaggle/make_extension_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = 'REPO_URL = "https://github.com/srinivas0307/lma-group-project.git"   # <- your repo\nBRANCH = "main"'

CLONE = '''# ---------------- Get the code ----------------
import os, shutil, subprocess, time, json, collections, zipfile, datetime
url = REPO_URL
try:
    from kaggle_secrets import UserSecretsClient
    url = REPO_URL.replace("https://", f"https://{UserSecretsClient().get_secret('GITHUB_TOKEN')}@")
except Exception:
    print("no GITHUB_TOKEN secret; cloning without a token (public repo)")
WORK = "/kaggle/working"
os.chdir(WORK)
shutil.rmtree(f"{WORK}/repo", ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, f"{WORK}/repo"], check=True)
os.chdir(f"{WORK}/repo")
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
missing = [f for f in NEEDED if not os.path.exists(f)]
assert not missing, f"missing (push them first): {missing}"
print("commit", COMMIT)'''

LIBS = '''# ---------------- Libraries: pinned pandas for the sandbox / verifier / tools (own folder) + bitsandbytes ----------------
subprocess.run(["pip", "install", "-q", "--target", f"{WORK}/evalpkgs", "pandas==2.3.3", "numpy==2.1.3",
                "pyyaml==6.0.2", "pydantic==2.14.1"], check=True)
subprocess.run(["pip", "install", "-q", "bitsandbytes"], check=True)
if BACKEND == "vllm":
    subprocess.run(["pip", "install", "-q", "-r", "requirements-vllm.txt"], check=True)
ENV = dict(os.environ, PYTHONPATH=f"{WORK}/evalpkgs")
# pip may print "ERROR: pip's dependency resolver ..." about Kaggle's pre-installed packages: only a warning.
probe = ("import pandas, numpy, torch, transformers; print('pandas', pandas.__version__, '| numpy', numpy.__version__, "
         "'| torch', torch.__version__, '| GPU', torch.cuda.is_available())")
p = subprocess.run(["python", "-c", probe], env=ENV, capture_output=True, text=True)
print(p.stdout, p.stderr[-1500:])
assert p.returncode == 0 and "pandas 2.3.3" in p.stdout and "GPU True" in p.stdout, "environment check failed"'''

SH = '''def sh(cmd, log=None):
    """Run a step, stream its output live, keep the full log."""
    t0 = time.time()
    p = subprocess.Popen(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if "Loading weights" not in line:
            print(line, end="", flush=True)
    rc = p.wait()
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        open(log, "a").write(f"$ {' '.join(cmd)}\\n" + "".join(lines) + f"\\n--- exit {rc}, {(time.time() - t0) / 60:.1f} min ---\\n")
    assert rc == 0, f"failed: {' '.join(cmd)}"
    return (time.time() - t0) / 60'''


def nb(cells):
    out = []
    for kind, src in cells:
        c = dict(cell_type=kind, metadata={}, source=src.strip("\n").splitlines(keepends=True))
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return dict(cells=out, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)


def bundle(name, folders):
    return f'''# ---------------- ONE zip with everything needed on the laptop (unzip at the project root) ----------------
BUNDLE = f"{{WORK}}/{name}.zip"
with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as z:
    for base in {folders}:
        for dp, _, fs in os.walk(base):
            for f in fs:
                full = os.path.join(dp, f)
                z.write(full, os.path.relpath(full, WORK))
    z.writestr("{name}_info.json", json.dumps(dict(commit=COMMIT, created=datetime.datetime.now().isoformat(timespec="seconds")), indent=2))
print("download:", BUNDLE, f"({{os.path.getsize(BUNDLE) / 1e6:.1f}} MB)")'''


NOTEBOOKS = {}

# ---------------------------------------------------------------- agent extensions (#11, #12, #7)
NOTEBOOKS["agent_extensions_kaggle.ipynb"] = nb([
    ("markdown", """# Agent extensions (GPU): no-progress stop (np1), feasibility step (fs1), vLLM backend

Runs `eval/runner_agent_ext.py` (a wrapper around the unchanged `eval/runner_agent.py`) on the agent scenarios.
Every extension is off unless set below; the system name gets the extension tags (`agent_full_np1_k3`,
`agent_full_fs1`, `..._vllm`), so nothing mixes with the existing `agent_full` / `agent_feedback` / `agent_naive` runs.

- **np1** `NO_PROGRESS_K` (default 3; 0 = off): stop with ESCALATE after K consecutive repair attempts without progress.
  With `max_attempts` = 3, K = 3 only turns a REJECT into an ESCALATE; K = 1 or 2 also saves calls.
  The offline replay (`eval/ext_replay_no_progress.py`, CPU) already gives the exact numbers for K = 1, 2, 3 on the
  recorded test runs; this notebook confirms them with real runs.
- **fs1** `FEASIBILITY = True`: one "can this be done?" call before the first repair (prompt a1-v4-fs1).
- **vllm** `BACKEND = "vllm"`: same runs on vLLM (its own version; outputs may differ from transformers).

After the run: download `agent_ext_bundle.zip`, unzip it at the project root (results land in
`results/stage2_agent/<system>/`), then `.venv/bin/python eval/agent_metrics.py results/stage2_agent test`.
Settings: GPU T4 x1, Internet on, secret `GITHUB_TOKEN` for a private repo. Resumable and cached."""),
    ("code", REPO + '''
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"; LOAD_IN_4BIT = True
STAGE = "test"                 # "dev" for a quick check
NO_PROGRESS_K = 3              # np1: 0 = off
FEASIBILITY = False            # fs1
BACKEND = "hf"                 # "hf" | "vllm" | "reference_policy" (pipeline check only, NOT results)
VERIFIER = "oracle"
SYSTEMS = [("agent_full", "full", [])]       # e.g. add ("agent_feedback", "feedback", []) for np1 on a baseline
CATEGORIES = ["R", "C", "T", "E", "S"]
LIMIT = None
OUT = "/kaggle/working/results/stage2_agent"
CACHE = "/kaggle/working/llm_cache"
NEEDED = ["eval/runner_agent_ext.py", "src/igt/agents/repair_agent_ext.py", "eval/runner_agent.py"]'''),
    ("code", CLONE), ("code", LIBS),
    ("code", SH + '''

for system, mode, flags in SYSTEMS:
    cmd = ["python", "eval/runner_agent_ext.py", "--system", system, "--mode", mode, "--backend", BACKEND,
           "--model", MODEL_ID, "--verifier", VERIFIER, "--splits", STAGE, "--categories", *CATEGORIES,
           "--cache-dir", CACHE, "--out", OUT, "--stop-after-no-progress", str(NO_PROGRESS_K), *flags]
    cmd += ["--load-in-4bit"] if LOAD_IN_4BIT else []
    cmd += ["--feasibility-check"] if FEASIBILITY else []
    cmd += ["--limit", str(LIMIT)] if LIMIT else []
    print(f"== {system}: {(sh(cmd, f'{OUT}/logs/{system}_ext_{STAGE}.log')):.1f} min")'''),
    ("code", '''# ---------------- Summary ----------------
p = subprocess.run(["python", "eval/agent_metrics.py", OUT, STAGE], env=ENV, capture_output=True, text=True)
print(p.stdout[-4000:], p.stderr[-1500:])
for d in sorted(os.listdir(OUT)):
    if os.path.isdir(f"{OUT}/{d}") and d != "logs":
        open(f"{OUT}/{d}/git_commit.txt", "w").write(COMMIT)
        if os.path.exists(f"{OUT}/{d}/run_info_ext.json"):
            print(d, json.load(open(f"{OUT}/{d}/run_info_ext.json"))["agent_version"])'''),
    ("code", bundle("agent_ext_bundle", '["/kaggle/working/results/stage2_agent"]')),
])

# ---------------------------------------------------------------- Stage 2 extensions (#11, #12, #7, #8, #9)
NOTEBOOKS["stage2_extensions_kaggle.ipynb"] = nb([
    ("markdown", """# Stage 2 extensions (GPU): np1, fs1, vLLM, and contracts with pre-check / re-ask

Runs `eval/runner_stage2_ext.py` (a wrapper around the unchanged `eval/runner_stage2.py`). Every extension is off
unless set below; the system name gets the tags (`proposed_pc1rq1_np1_k3__gen_3b`, ...), so nothing mixes with the
existing Stage 2 systems.

- **np1** `NO_PROGRESS_K` (default 3; 0 = off), **fs1** `FEASIBILITY`, **vllm** `BACKEND = "vllm"`.
- **pc1 / rq1 / pc1rq1** (contracts): set `PARSES` to an extension parse folder's `parses.jsonl`:
  `results/stage1/extensions/<parser>_pc1/parses.jsonl` (pre-check, already computed offline and in the repo), or the
  `_rq1` / `_pc1rq1` folders produced by `reask_kaggle.ipynb` (copy them into the repo first, or attach them as a
  Kaggle dataset and give the path).

After the run: download `stage2_ext_bundle.zip`, unzip it at the project root, then
`.venv/bin/python eval/stage2_metrics.py results/stage2 test`."""),
    ("code", REPO + '''
STAGE = "test"
GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"; GENERATOR_4BIT = False
AGENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"; AGENT_4BIT = True
NO_PROGRESS_K = 3              # np1: 0 = off
FEASIBILITY = False            # fs1
BACKEND = "hf"                 # "hf" | "vllm" | "pipeline_check" (NOT results)
# (preset, parses.jsonl or None). Contract extensions: point PARSES at an extension parse folder.
RUNS = [
    ("proposed", "results/stage1/extensions/intent_prompted_7b_p1v3_pc1/parses.jsonl"),
    ("gold_contract", None),
]
SEEDS = [0]; TEMPERATURE = 0.0; LIMIT = None
OUT = "/kaggle/working/results/stage2"
CACHE = "/kaggle/working/llm_cache"
GEN_TAG = GENERATOR_MODEL.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "").lower().replace(".", "_")
NEEDED = ["eval/runner_stage2_ext.py", "eval/runner_stage2.py", "src/igt/agents/repair_agent_ext.py"] + \\
         [p for _, p in RUNS if p]'''),
    ("code", CLONE), ("code", LIBS),
    ("code", SH + '''

for preset, parses in RUNS:
    tag = f"__{os.path.basename(os.path.dirname(parses))}" if parses else ""
    system = f"{preset}__gen_{GEN_TAG}{tag}"
    cmd = ["python", "eval/runner_stage2_ext.py", "--preset", preset, "--system", system, "--backend", BACKEND,
           "--generator-model", GENERATOR_MODEL, "--agent-model", AGENT_MODEL, "--splits", STAGE,
           "--seeds", *map(str, SEEDS), "--temperature", str(TEMPERATURE), "--cache-dir", CACHE, "--out", OUT,
           "--stop-after-no-progress", str(NO_PROGRESS_K)]
    cmd += ["--generator-4bit"] if GENERATOR_4BIT else []
    cmd += ["--agent-4bit"] if AGENT_4BIT else []
    cmd += ["--feasibility-check"] if FEASIBILITY else []
    cmd += ["--parses", parses] if parses else []
    cmd += ["--limit", str(LIMIT)] if LIMIT else []
    print(f"== {system}: {sh(cmd, f'{OUT}/logs/{system}_{STAGE}.log'):.1f} min")'''),
    ("code", '''# ---------------- Summary ----------------
p = subprocess.run(["python", "eval/stage2_metrics.py", OUT, STAGE], env=ENV, capture_output=True, text=True)
print(p.stdout[-4000:], p.stderr[-1500:])
for d in sorted(os.listdir(OUT)):
    if os.path.isdir(f"{OUT}/{d}") and d != "logs":
        open(f"{OUT}/{d}/git_commit.txt", "w").write(COMMIT)'''),
    ("code", bundle("stage2_ext_bundle", '["/kaggle/working/results/stage2"]')),
])

# ---------------------------------------------------------------- re-ask (#8, #9+#8)
NOTEBOOKS["reask_kaggle.ipynb"] = nb([
    ("markdown", """# Re-ask once on an invalid contract (rq1), optionally after the data pre-check (pc1rq1) (GPU)

`eval/ext_reask.py` reads an existing parser run (only read), sends **only** the flagged tasks back to the same model
once ("Your contract is invalid: <error> ..." or, with the pre-check, "Your contract does not fit the input data:
<issues> ..."), and writes a new parse folder `<system>_rq1` / `<system>_pc1rq1`. One extra model call per re-asked task.
For the 7B p1-v3 run that is 18 tasks (rq1) or 20 (pc1rq1).

After the run: download `reask_bundle.zip`, unzip it at the project root (folders land in
`results/stage1/extensions/`), then `.venv/bin/python eval/ext_score.py`.
Fine-tuned parsers: add their parse folder (in the repo under `kaggle_outputs/parsing/`) and their adapter path
(attach the adapter zip as a Kaggle dataset)."""),
    ("code", REPO + '''
BACKEND = "hf"                  # "hf" | "vllm" | "gold" (pipeline check only, NOT results)
# (parse folder under kaggle_outputs/parsing/, model that wrote it, 4-bit, adapter path or None)
RUNS = [
    ("7b_p1-v3", "Qwen/Qwen2.5-7B-Instruct", True, None),
    # ("3b_p1-v2", "Qwen/Qwen2.5-3B-Instruct", False, None),
    # ("1_5b_p1-v3", "Qwen/Qwen2.5-1.5B-Instruct", False, None),
    # ("ft_1_5b", "Qwen/Qwen2.5-1.5B-Instruct", True, "/kaggle/input/<your-adapter-dataset>/adapter"),
]
MODES = ["rq1", "pc1rq1"]       # rq1 = invalid contracts only; pc1rq1 = + valid contracts that fail the pre-check
OUT_ROOT = "/kaggle/working/results/stage1/extensions"
CACHE = "/kaggle/working/llm_cache"
NEEDED = ["eval/ext_reask.py", "src/igt/agents/reask.py", "src/igt/contracts/precheck.py"] + \\
         [f"kaggle_outputs/parsing/{r[0]}/parses.jsonl" for r in RUNS]'''),
    ("code", CLONE), ("code", LIBS),
    ("code", SH + '''

for folder, model, q, adapter in RUNS:
    for mode in MODES:
        cmd = ["python", "eval/ext_reask.py", folder, "--backend", BACKEND, "--out-root", OUT_ROOT, "--cache-dir", CACHE]
        cmd += ["--model", model] if BACKEND != "gold" else []
        cmd += ["--load-in-4bit"] if q and BACKEND != "gold" else []
        cmd += ["--adapter", adapter] if adapter else []
        cmd += ["--precheck"] if mode == "pc1rq1" else []
        print(f"== {folder} {mode}: {sh(cmd, f'{OUT_ROOT}/logs/{folder}_{mode}.log'):.1f} min")
for d in sorted(os.listdir(OUT_ROOT)):
    if os.path.exists(f"{OUT_ROOT}/{d}/env.json"):
        e = json.load(open(f"{OUT_ROOT}/{d}/env.json"))
        open(f"{OUT_ROOT}/{d}/git_commit.txt", "w").write(COMMIT)
        print(f"{d}: re-asked {e['n_reasked']}, valid {e['valid_before']} -> {e['valid_after']}")'''),
    ("code", bundle("reask_bundle", '["/kaggle/working/results/stage1/extensions"]')),
])

# ---------------------------------------------------------------- LLM verifier (#3)
NOTEBOOKS["llm_verifier_kaggle.ipynb"] = nb([
    ("markdown", """# LLM-as-verifier baseline (lv1) over the frozen pool (GPU)

A model judges each of the 803 pool candidates from the request, the input schema and a summary of what the code
changed (`src/igt/baselines/llm_verifier.py`); no contract. One call per candidate that ran (about 760).
Same model as Baseline 3 by default.

After the run: download `llm_verifier_bundle.zip`, unzip it at the project root (lands in
`results/stage1/extensions/llm_verifier_<size>_lv1/`), then `.venv/bin/python eval/ext_score.py` (recall, false
rejections, McNemar vs Baselines 2 and 3)."""),
    ("code", REPO + '''
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"; LOAD_IN_4BIT = True
BACKEND = "hf"                  # "hf" | "vllm" | "accept_all" (pipeline check only, NOT results)
LIMIT = None                    # e.g. 20 for a quick check
SIZE = MODEL_ID.split("-")[-2].lower().replace(".", "_")
SYSTEM = f"llm_verifier_{SIZE}_lv1"
OUT = f"/kaggle/working/results/stage1/extensions/{SYSTEM}"
CACHE = "/kaggle/working/llm_cache"
NEEDED = ["eval/run_llm_verifier.py", "src/igt/baselines/llm_verifier.py", "benchmark/pool/pool_v4.jsonl"]'''),
    ("code", CLONE), ("code", LIBS),
    ("code", SH + '''

cmd = ["python", "eval/run_llm_verifier.py", "--backend", BACKEND, "--model", MODEL_ID, "--system", SYSTEM,
       "--out", OUT, "--cache-dir", CACHE]
cmd += ["--load-in-4bit"] if LOAD_IN_4BIT else []
cmd += ["--limit", str(LIMIT)] if LIMIT else []
print(f"done in {sh(cmd, f'{OUT}/run.log'):.1f} min")
open(f"{OUT}/git_commit.txt", "w").write(COMMIT)
print(json.load(open(f"{OUT}/run_info.json")))'''),
    ("code", bundle("llm_verifier_bundle", '[OUT]')),
])

# ---------------------------------------------------------------- demo UI (#6)
NOTEBOOKS["demo_ui_kaggle.ipynb"] = nb([
    ("markdown", """# Demo UI (Gradio) on Kaggle: Task, Replay and Live tabs (GPU for Live)

Starts `app/demo_app.py` with a temporary public link (`share=True`; the link is printed below and works while this
session runs). Task and Replay use the recorded data in the repo; Live loads the models on its first run and keeps
them loaded. To stop: interrupt the last cell.
Settings: GPU T4 x1 (two T4s if parser and agent are different large models), Internet on."""),
    ("code", REPO + '''
PARSER_MODEL = "Qwen/Qwen2.5-7B-Instruct"     # or "Qwen/Qwen2.5-1.5B-Instruct" with ADAPTER
ADAPTER = ""                                   # fine-tuned parser adapter folder (attach it as a Kaggle dataset)
GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
AGENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BACKEND = "hf"
NEEDED = ["app/demo_app.py", "app/demo_core.py", "eval/demo.py"]'''),
    ("code", CLONE), ("code", LIBS),
    ("code", '''# ---------------- Gradio (installed next to the pinned packages) ----------------
subprocess.run(["pip", "install", "-q", "--target", f"{WORK}/demopkgs", "gradio>=5,<6"], check=True)
ENV_UI = dict(ENV, PYTHONPATH=f"{WORK}/evalpkgs:{WORK}/demopkgs", IGT_DEMO_PARSER=PARSER_MODEL, IGT_DEMO_ADAPTER=ADAPTER,
              IGT_DEMO_GENERATOR=GENERATOR_MODEL, IGT_DEMO_AGENT=AGENT_MODEL)'''),
    ("code", '''# ---------------- Launch (the public link is printed; interrupt this cell to stop) ----------------
p = subprocess.Popen(["python", "app/demo_app.py", "--share", "--host", "0.0.0.0"], env=ENV_UI,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in p.stdout:
        print(line, end="", flush=True)
except KeyboardInterrupt:
    p.terminate(); print("stopped")'''),
])

# ---------------------------------------------------------------- tool-using fine-tuned parser (#13)
NOTEBOOKS["finetune_tools_kaggle.ipynb"] = nb([
    ("markdown", """# Tool-using fine-tuned parser (ft-t1): train + parse, one size per session (GPU)

Trains a LoRA adapter on data **v3-tools** (`finetune/data/v3_tools/`: data v2 with the fill-with-median and
fill-with-most-common examples as multi-turn tool trajectories; built by `finetune/data_gen/build_tools_data.py`)
with `finetune/train_qlora_tools.py` (loss on every assistant turn), then parses the 120 tasks with tools
(`eval/parse_contracts_tools_ft.py`). Same settings as the current fine-tuning (1 epoch, batch 1 x 16).

Set `SIZE = "1_5b"` or `"3b"` (one per session: ~4 h / ~8 h, estimates). Run `SMOKE = True` first (20 steps).
Outputs: `ft_tools_<size>.zip` (adapter, log, run info) and `parse_ft_tools_<size>.zip`. Unzip the parse zip to
`results/stage1/extensions/intent_ft_tools_<size>/` and run `.venv/bin/python eval/ext_score.py`.
Settings: GPU T4 x1, Internet on."""),
    ("code", REPO + '''
SIZE = "1_5b"                   # "1_5b" or "3b"
SMOKE = True                    # True: 20 steps (check memory, loss, time per step); False: full training + parsing
BACKEND = "hf"
BASE = {"1_5b": "Qwen/Qwen2.5-1.5B-Instruct", "3b": "Qwen/Qwen2.5-3B-Instruct"}[SIZE]
CONFIG = f"finetune/configs/qwen25_{SIZE}_tools.yaml"
SYSTEM = f"intent_ft_tools_{SIZE}"
NEEDED = ["finetune/train_qlora_tools.py", "finetune/train_qlora.py", "eval/parse_contracts_tools_ft.py", CONFIG,
          "finetune/data_gen/build_tools_data.py"]'''),
    ("code", CLONE + '''
if not os.path.exists("finetune/data/v3_tools/sft_train.jsonl"):        # build it (CPU, ~1 min) if not pushed
    subprocess.run(["python", "finetune/data_gen/build_tools_data.py"], check=True)
print(open("finetune/data/v3_tools/sft_stats.json").read()[:600])'''),
    ("code", '''# ---------------- Libraries: training stack (as the current fine-tuning notebook) + pinned pandas for parsing ----------------
subprocess.run(["pip", "install", "-q", "-r", "finetune/requirements-train.txt"], check=True)
subprocess.run(["pip", "install", "-q", "--target", f"{WORK}/evalpkgs", "pandas==2.3.3", "numpy==2.1.3",
                "pyyaml==6.0.2", "pydantic==2.14.1"], check=True)
ENV = dict(os.environ, PYTHONPATH=f"{WORK}/evalpkgs")
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available()'''),
    ("code", SH + '''

out = f"finetune/runs/qwen25_{SIZE}_tools" + ("_smoke" if SMOKE else "")
sets = [f"output_dir={out}"] + (["max_steps=20", "eval_steps=10", "logging_steps=2", "limit_val=64"] if SMOKE
                                else ["limit_val=200"])
TRAIN_ENV = dict(os.environ)          # training uses Kaggle's stack, like finetune_qlora_kaggle.ipynb
p = subprocess.Popen(["python", "finetune/train_qlora_tools.py", "--config", CONFIG, "--set", *sets], env=TRAIN_ENV,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
t0 = time.time()
for line in p.stdout:
    print(line, end="", flush=True)
assert p.wait() == 0, "training failed"
mins = (time.time() - t0) / 60
info = json.load(open(f"{out}/run_info.json"))
print(f"trained in {mins:.1f} min | loss {info['first_logged_loss']} -> {info['last_logged_loss']} | prompt {info['parser_prompt_version']}")
if SMOKE:
    print(f"estimated full training: ~{mins * ((info['n_train'] + 15) // 16) / info['steps'] / 60:.1f} h (+ evaluations and parsing)")
open(f"{out}/git_commit.txt", "w").write(COMMIT)
print("saved:", shutil.make_archive(f"{WORK}/ft_tools_{SIZE}" + ("_smoke" if SMOKE else ""), "zip", out))'''),
    ("code", '''# ---------------- Parse the 120 tasks with tools (skipped in SMOKE mode) ----------------
if not SMOKE:
    pout = f"{WORK}/parse_ft_tools_{SIZE}"
    mins = sh(["python", "eval/parse_contracts_tools_ft.py", "--model", BASE, "--load-in-4bit", "--adapter",
               f"{out}/adapter", "--backend", BACKEND, "--system", SYSTEM, "--out", pout, "--cache-dir", f"{WORK}/llm_cache"])
    R = [json.loads(l) for l in open(f"{pout}/parses.jsonl")]
    print(f"parsed in {mins:.1f} min | status {dict(collections.Counter(r['parse_status'] for r in R))} | "
          f"tasks with a tool call {sum(r['n_tool_calls'] > 0 for r in R)}")
    open(f"{pout}/git_commit.txt", "w").write(COMMIT)
    print("saved:", shutil.make_archive(pout, "zip", pout))'''),
])

# ---------------------------------------------------------------- vLLM check (#7)
NOTEBOOKS["vllm_check_kaggle.ipynb"] = nb([
    ("markdown", """# vLLM check (GPU): same 5 prompts with transformers and vLLM

Runs the Baseline 3 parser prompt (p1-v3) for 5 test tasks with each backend (each in its own process) and compares
the replies, the parsed contracts and the time per prompt (`eval/ext_vllm_compare.py`). Use this before running any
system on vLLM: greedy outputs can differ, which is why vLLM runs are their own version.
Installing vLLM may replace Kaggle's torch build; this notebook is separate for that reason."""),
    ("code", REPO + '''
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"; LOAD_IN_4BIT = True
N = 5
BACKEND = "vllm"
NEEDED = ["eval/ext_vllm_compare.py", "src/igt/llm/vllm_backend.py", "requirements-vllm.txt"]'''),
    ("code", CLONE), ("code", LIBS),
    ("code", SH + '''

for be in ("hf", "vllm"):
    cmd = ["python", "eval/ext_vllm_compare.py", "--backend", be, "--model", MODEL_ID, "--n", str(N),
           "--out", f"{WORK}/compare_{be}.json"] + (["--load-in-4bit"] if LOAD_IN_4BIT else [])
    print(f"== {be}: {sh(cmd):.1f} min")
sh(["python", "eval/ext_vllm_compare.py", "--compare", f"{WORK}/compare_hf.json", f"{WORK}/compare_vllm.json"])'''),
])


if __name__ == "__main__":
    for name, d in NOTEBOOKS.items():
        p = HERE / name
        p.write_text(json.dumps(d, indent=1) + "\n")
        print("wrote", p.name)
