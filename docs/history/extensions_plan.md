# Plan: optional extensions as separate versions (v2, confirmed choices; nothing implemented yet)

Changes from v1 of this plan (your answers): #10 (unsafe-request check) dropped; #11 stops after **3** no-progress
attempts by default, changeable by a flag / config value; #6 uses **Gradio** (it runs inside a Kaggle notebook, where
the Live tab gets its GPU; Streamlit doesn't); #13 gets a notebook for **both** sizes (1.5B and 3B, one session each).

Ten extensions, each a **separate, opt-in version** next to the systems in the proposal. You can run, review or ignore
each one independently; none of them changes an existing system, prompt, result or default.

## 0. Rules that keep the extensions from colliding with the proposal

1. **Off by default.** Every extension is enabled only by an explicit flag, config file or system name. Running any
   existing command without the new flag behaves exactly as today.
2. **Own version id and own name.** Each extension has a short id (e.g. `rq1`). Systems that use it carry the id in
   their name (`intent_prompted_7b_p1v3_rq1`, `agent_full_np1`), and runs record it in `run_info.json` / `env.json`.
   The runners already refuse to mix versions in one results folder; the same check covers the new ids.
3. **Own output folders.** Stage 1 variants go to `results/stage1/extensions/<system>/`; agent variants to
   `results/stage2_agent/<system>/`; Stage 2 variants to `archive/stage2/early_evaluation/<system>/`. Existing folders are never written.
4. **Existing notebooks and prompts are frozen.** No existing notebook is edited; extensions get their own notebooks. The current prompt hashes (`b1-v1`, `b2-v1`, `p1-v1..v3`, `a1-v4`, `t1-v1`) stay
   pinned in tests. An extension that needs a different prompt gets a new prompt version.
5. **Proof of no collision, per extension:** the full test suite passes and `./reproduce_all.sh` still reports
   `results/stage1/` identical.
6. **One registry:** `docs/history/extensions.md` (written during implementation) lists every extension, its id, how to
   switch it on, what it needs (CPU / GPU), and where its results go.

## 1. Overview

| # | Extension | Id | Kind | Evaluate on | GPU needed to evaluate? | Effort |
|---|---|---|---|---|---|---|
| 8 | Re-ask once on an invalid contract | `rq1` | parser variant | Stage 1 (120 tasks) | yes, only for the invalid parses | small |
| 9 | Contract pre-check against the data | `pc1` | parser variant | Stage 1 | **no** (offline on existing parses); yes if combined with re-ask | small |
| 11 | Agent stops after *k* attempts that make no progress (default *k* = 3) | `np1` | agent variant | agent scenarios | **no** for a replay from existing traces; yes for a real run | small |
| 12 | Agent "can this be done?" step before repair | `fs1` | agent variant | agent scenarios | yes | small |
| 1 | Config file per system | `cfg` | infrastructure | — | no | small–medium |
| 7 | vLLM backend | `vllm` | infrastructure | any GPU run | yes (it is a GPU backend) | small |
| 3 | LLM-as-verifier baseline | `lv1` | new baseline | Stage 1 pool (803 candidates) | yes | medium |
| 6 | Demo UI (Gradio) | `ui` | presentation | — | no (the Live tab needs a GPU) | small–medium |
| 5 | Docker sandbox | `docker` | infrastructure | red-team set, pool | no (needs Docker installed) | medium |
| 13 | Tool-using fine-tuned parser | `ft-t1` | parser variant + training data | Stage 1 | yes (training + parsing) | medium–large |

## 2. Each extension

### 8. Re-ask once on an invalid contract (`rq1`)

- **What:** when a parser's contract fails validation, send the validation error back once ("Your contract is invalid:
  <error>. Reply with a corrected contract.") and use the second answer. Counted as 2 model calls.
- **Why:** half of Baseline 3's false rejections come from invalid contracts (51 of 106 correct candidates rejected);
  13% (7B) to 34% (1.5B) of contracts are invalid.
- **How:** new script `eval/reask_parses.py` reads an **existing** `parses.jsonl`, re-asks only the invalid tasks with
  the same model and the original conversation, and writes a new folder with `raw_output_first`, `raw_output`,
  `reasked`. The re-ask prompt is versioned (`rq1`). Works for any parser: prompted, fine-tuned, tool-using.
- **Systems:** `<parser>_rq1`, e.g. `intent_prompted_7b_p1v3_rq1`; scored by `eval/score_parsers.py` like any parser.
- **Kaggle:** one notebook cell list of parse folders to re-ask; only invalid tasks are sent (18 of 120 for 7B p1-v3).
- **Tests:** scripted model returns an invalid then a valid contract; valid parses are copied unchanged.

### 9. Contract pre-check against the data (`pc1`)

- **What:** deterministic checks of a *valid* contract against the input data, before it is used:
  - a `must_change` condition (`rows_removed_where`, a `where`) that matches **0 rows** (the requested change can't happen);
  - a fill or `const` value that is the **name of a statistic** (`"median"`, `"mean"`, `"mode"`, `"average"`,
    `"most common"`) instead of a value;
  - a literal whose type doesn't fit the column (text compared with a numeric column, a number with a text column);
  - a category value or `in` list item that **doesn't occur** in the column;
  - a `duplicate_count` key column that doesn't exist.
- **How:** `src/igt/contracts/precheck.py` → list of issues. Two modes: **pc1** = a contract with issues is treated as
  invalid (escalated), computed **offline** from the existing parses, no GPU; **pc1+rq1** = the issues are sent back
  with the re-ask of #8 (GPU).
- **Systems:** `<parser>_pc1`, `<parser>_pc1rq1`.
- **Tests:** each check on a hand-written contract; every gold contract passes the pre-check (no false alarms on gold).
- **Report:** how many parsed contracts each check flags, and how many of those were wrong vs the gold contract.

### 11. Agent stops after attempts that make no progress (`np1`)

- **What:** after *k* consecutive repair attempts with no progress (same definition as the metric: not passing and
  failures not a strict subset of the previous attempt's), the agent stops with ESCALATE ("no progress").
  **Default *k* = 3**, changeable without code changes: runner flag `--stop-after-no-progress K`, the same key in the
  system's config file (#1), and the `NO_PROGRESS_K` setting in the new extension notebooks. The value used is
  recorded in each run and in the system name (`agent_full_np1_k3`), so runs with different *k* never mix.
- **How:** new `AgentConfig` field `stop_after_no_progress` (default `None` = today's behaviour) and the runner flag.
  The agent prompt is unchanged, so the version id is recorded as `a1-v4+np1`.
- **Note:** the agent has at most 3 repair attempts (`max_attempts`), so with *k* = 3 the rule only fires when every
  attempt made no progress and stops the run as ESCALATE instead of REJECT; a smaller *k* (or a larger
  `max_attempts`) also saves calls. The offline replay reports *k* = 1, 2, 3 side by side so you can pick.
- **Offline replay (no GPU):** with greedy decoding the agent's replies up to the stop point are the recorded ones, so
  the outcome with the stop rule can be computed exactly from `traces.jsonl` (cut the run at the stop point). A real
  run on Kaggle confirms it.
- **Report:** escalations, repair success, tokens saved; the trade-off between giving up early and repairing late.
- **Tests:** scripted runs with 0, 1, 2 no-progress attempts.

### 12. Agent "can this be done?" step before repair (`fs1`)

- **What:** before the first repair attempt, one model call: given the request, the input schema, the contract and the
  failure report, answer `{"feasible": true|false, "reason": ...}`. If false, ESCALATE with the reason; otherwise
  repair as today.
- **Why:** the agent chose `escalate` itself once in 222 runs; 7 of its 8 escalations on impossible requests came from
  the verifier.
- **How:** new prompt version `a1-v4-fs1` (the a1-v4 prompts stay unchanged); `AgentConfig.feasibility_check`.
- **Report:** escalation recall on E (decided by the agent), false escalations on R, extra tokens.
- **Tests:** scripted feasible / infeasible replies; malformed reply counts as feasible (repair proceeds) and is logged.

### 1. Config file per system (`cfg`)

- **What:** one YAML per system in `configs/systems/` (Stage 1 verifiers and parsers, agent systems, Stage 2 presets,
  extension variants): which runner, which model and prompt version, which flags, which output folder.
- **How:** `src/igt/config.py` loads a file; every runner gets `--config <file>`; explicit flags still win. A script
  `eval/run_config.py configs/systems/<x>.yaml` prints or runs the exact command.
- **Tests:** each existing system's config produces the same arguments the current commands use (e.g. the Stage 2
  presets equal `PRESETS`); `reproduce_all.sh` can use them.
- **Effect on results:** none (the same commands, written down).

### 7. vLLM backend (`vllm`)

- **What:** a faster GPU backend with the same `complete(messages, ...)` interface as `HFChatBackend`, with LoRA
  adapters (`LoRARequest`) for the fine-tuned parsers.
- **How:** `src/igt/llm/vllm_backend.py`; `--backend vllm` in the agent, Stage 2, tool-parser and demo runners;
  `requirements-vllm.txt`.
- **Caveat:** greedy outputs can differ from transformers (different kernels), so a vLLM run is its own version: the
  backend is recorded in `run_info.json`, and the folder check refuses to mix it with transformers runs. Existing
  results stay on transformers.
- **Tests:** interface test with a stub (no GPU here); a Kaggle cell compares 5 prompts between the two backends.

### 3. LLM-as-verifier baseline (`lv1`)

- **What:** an LLM judges each candidate directly: it sees the request, the input schema, and a summary of what the
  code changed (rows removed / added, changed cells per column with examples, dtype changes, new and dropped columns;
  from the existing diff tool), and answers `{"verdict": "ACCEPT"|"REJECT", "reason": ...}`. No contract.
- **Why:** the comparison reviewers usually ask for: does a contract add anything over asking a model "is this right?"
- **How:** `src/igt/baselines/llm_verifier.py` (versioned prompt `lv1`), `eval/run_llm_verifier.py` over the frozen
  pool (803 candidates), writing verdicts in the standard format, so `eval/metrics.py` scores it with recall, FRR,
  McNemar against Baselines 2 and 3; `archive/notebooks/extensions/llm_verifier_kaggle.ipynb`. Model: the Baseline 3 7B.
- **Validation (H20):** the answer key is the ground truth for the verdicts, so no human labels are needed for this
  baseline; its reasons are logged for the failure analysis.
- **Tests:** scripted model; diff summary on a known candidate; malformed reply = ESCALATE.

### 6. Demo UI (`ui`)

- **Choice: Gradio.** The Live tab needs a GPU, which in this project means Kaggle; Gradio runs inside a notebook
  cell and gives a temporary public link (`share=True`), while Streamlit needs a separate server process. Locally it
  runs with `python app/demo_app.py`.
- **What:** a small web app on top of `eval/demo.py`: **Task** tab (pick a task and a candidate, see the contract, the
  code, the intent verifier vs conventional checks, the answer key), **Replay** tab (pick a recorded run, see its
  steps), **Live** tab (upload a CSV and type a request; needs a GPU, disabled otherwise).
- **How:** `app/demo_app.py` with Gradio; `requirements-demo.txt` (not a core dependency); a Kaggle notebook
  `archive/notebooks/extensions/demo_ui_kaggle.ipynb` that loads the models and launches the app with a share link. The demo
  functions return structured data as well as printing, so the CLI and the UI share them.
- **Tests:** the shared functions; the app module imports without a GPU.

### 5. Docker sandbox (`docker`)

- **What:** a second sandbox backend that runs untrusted code in a container with `--network none`, a read-only root
  filesystem, the data mounted read-only, memory / CPU / process limits; same `ExecResult` as today. Closes OI-6
  (today network isolation on the dev machine is a socket stub, and the filesystem isn't isolated).
- **How:** `src/igt/sandbox/docker_executor.py`, `sandbox/docker/Dockerfile` (Python + the pinned pandas / numpy);
  selected with `IGT_SANDBOX=docker` or a parameter; default stays the subprocess sandbox.
- **Tests (skipped when Docker isn't installed):** the red-team set blocked or contained; the reference solutions and a
  sample of pool candidates give the same status and output with both backends.

### 13. Tool-using fine-tuned parser (`ft-t1`)

- **What:** teach the small parsers (1.5B, 3B) to call the data tools of the tool-using parser, so option A works
  without a 7B model.
- **How:**
  - **Data:** extend `finetune/data_gen/generate_examples.py` so that patterns whose contract uses a value from the data
    (fill with median / mode / most common, thresholds, category lists) also produce a multi-turn example: assistant
    tool call → tool result (computed with the real tools on the training dataset) → final contract. Other patterns
    stay single-turn (the model must also learn when *not* to call tools). New data version `v3-tools` in
    `finetune/data/v3_tools/`; data v2 unchanged.
  - **Prompt:** `p1-v2` + the tool instructions = new version `ft-t1`, recorded in the adapter as today.
  - **Training:** `finetune/train_qlora.py` learns from every assistant turn in multi-turn examples (today only the last
    message); single-turn behaviour stays identical (tested).
  - **Parsing:** `eval/parse_contracts_tools.py --adapter` (zero-shot, the adapter's prompt version).
  - **Checks:** the existing 7 data checks plus "the tool results in each example match the data"; a hand-review sample.
- **Systems:** `intent_ft_tools_1_5b`, `intent_ft_tools_3b` in `results/stage1/extensions/`.
- **Notebooks: both sizes.** `archive/notebooks/extensions/finetune_tools_kaggle.ipynb` with a `SIZE = "1_5b" | "3b"` switch (one
  size per session: training then parsing the 120 tasks with tools), and a smoke mode as in the current fine-tuning
  notebook. Configs `finetune/configs/qwen25_1_5b_tools.yaml`, `qwen25_3b_tools.yaml` (same settings as the current
  ones: 1 epoch, batch 1 x 16). You can run 1.5B only, 3B only, or both.
- **Cost:** a new training run per size on Kaggle, a little longer than the current ones (~4 h / ~8 h), because the
  multi-turn examples are longer.

## 3. Order of implementation (one go, after your confirmation)

1. Shared pieces: `docs/history/extensions.md` registry, output folder rule, version ids, the "no collision" test.
2. CPU-evaluable first: **9** (pre-check), **11** (no-progress stop, with offline replay for *k* = 1, 2, 3), then
   **8** (re-ask), **12** (feasibility step).
3. Infrastructure: **1** (configs, including every extension), **7** (vLLM), **5** (Docker).
4. New baseline and UI: **3** (LLM-as-verifier), **6** (demo UI).
5. Largest last: **13** (tool-using fine-tuning data and training changes).

Each step ends with the full test suite and the `reproduce_all.sh` identity check. Offline results (9, 11) are
computed right away from the existing runs; the GPU ones (7, 8, 12, 3, 13) come with a Kaggle notebook or notebook cell
and are not run by me.

## 4. Notebooks that will exist after implementation (none run by me)

| Notebook | Extensions | What you set |
|---|---|---|
| `archive/notebooks/extensions/reask_kaggle.ipynb` | 8, 9 (pc1+rq1) | which parse folders to re-ask; pre-check on/off |
| `archive/notebooks/extensions/agent_extensions_kaggle.ipynb` (new; same layout as the agent notebook) | 11, 12, 7 | `NO_PROGRESS_K` (default 3); the `np1` / `fs1` systems; backend |
| `archive/notebooks/extensions/stage2_extensions_kaggle.ipynb` (new; same layout as the Stage 2 notebook) | 11, 12, 7, 8, 9 | `NO_PROGRESS_K`, feasibility on/off, re-ask / pre-check on/off, backend |
| `archive/notebooks/extensions/llm_verifier_kaggle.ipynb` | 3 | model |
| `archive/notebooks/extensions/demo_ui_kaggle.ipynb` | 6 | models, adapter |
| `archive/notebooks/extensions/finetune_tools_kaggle.ipynb` | 13 | `SIZE = "1_5b"` or `"3b"`, smoke mode |
| `archive/notebooks/extensions/vllm_check_kaggle.ipynb` | 7 | compares 5 prompts between vLLM and transformers |

**Existing notebooks are not edited:** `agent_eval_kaggle.ipynb`, `stage2_kaggle.ipynb`, `finetune_qlora_kaggle.ipynb`,
`tool_parser_kaggle.ipynb` and the others stay exactly as they are; every extension gets its own notebook.

#5 (Docker) and #1 (configs) need no notebook: Docker runs on a machine with Docker installed (not Kaggle), and the
configs are files.
