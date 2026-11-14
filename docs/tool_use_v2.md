# Two independent tool-use implementations

The new code is in `finetune/tool_use/`. It does not change the completed `stage2-tools-v1` experiment, the original adapters, or the selective-facts training currently running.

1. **Base + tools:** Qwen2.5-1.5B-Instruct or Qwen2.5-3B-Instruct, with no adapter. The model receives the contract vocabulary, tool instructions, illustrative examples, schema and three input rows. It chooses whether to request a tool.
2. **New tool adapter:** fresh QLoRA training on the selected base model. Training conversations teach tool requests, reading observations, returning contracts, answering without tools, correcting failed queries, and escalating unresolved modes/all-missing statistics. It does not continue either old adapter.

Both use `parser-tools-v2`, NF4 with FP16 computation, greedy decoding, identical prompts and the same read-only tools and limits. The protocol is custom JSON text (`{"tool":"profile_column","args":...}`), not native Hugging Face function calling. Tool observations arrive as user messages. Calls are executed by Python against the input DataFrame only. No gold contract, reference code or transformed output is available to the parser. There is no forced retrieval or automatic statistics injection.

## What to run now

All three notebooks are in `notebooks/kaggle/tool_use_v2/`. They contain normal Python code cells and are small; their source/data is packaged separately to avoid the earlier oversized Kaggle commit payload.

1. Create a **private Kaggle dataset** from `tool_use_source.zip` in that directory. Kaggle may unpack it; both packed and unpacked forms are supported. No GitHub push is required.
2. Upload **`tool_use_base.ipynb`** as a new Kaggle notebook. Attach that source dataset, choose **T4 x2**, and turn **Internet on**. Do not attach an old adapter.
3. Set `MODEL_SIZE = "1_5b"` or `"3b"` in the first code cell. Start with `"1_5b"` for continuity with the earlier experiments. Save Version → Run All. The notebook runs infrastructure checks, a two-task smoke evaluation, then a fresh 100-task parser development evaluation.
4. Download `tool_use_base_1_5b.zip` (or `tool_use_base_3b.zip`) and the executed notebook.
5. For training, upload **`tool_use_train.ipynb`** in a separate session, attach the same source dataset, and choose the **same MODEL_SIZE**. No old adapter or previous output bundle is needed. It independently evaluates the base, trains a fresh adapter for two epochs and scores both checkpoints. Download `tool_use_train_1_5b.zip` or `tool_use_train_3b.zip` plus the executed notebook.
6. Inspect `selection.json`. `eligible: true` means the candidate passed the finite development screening rule; it does not mean it is universally best. If false, the candidate remains available for investigation, but should not replace the main parser on that evidence.
7. To use that new adapter later, upload its **complete `adapter/` folder**, including `tool_protocol.json` and `selection.json`, as a new private Kaggle dataset. Upload **`tool_use_adapter.ipynb`**, attach it and the source dataset, set `MODEL_SIZE` to the trained size, and set `ADAPTER` to that folder. This optional notebook evaluates the trained parser with the same tool loop.

The training notebook already includes base-versus-checkpoint scores; the separate base notebook lets you inspect tool use immediately without waiting for training. Run each size in a separate session. A 1.5B adapter cannot be loaded onto the 3B model. The old contract adapter and the selective-facts adapter are rejected by this workflow.

Place downloaded ZIPs under `kaggle_outputs/tool_use_v2/` and executed notebooks under `notebooks/executed/tool_use_v2/`. Include the size in saved notebook names, for example `tool_use_base_3b_executed.ipynb`.

## Training and selection

The checked training corpus is converted to matching conversations; it is not made from previous model outputs. Original source splits are retained. There are **4,743 training conversations** and **914 validation conversations**. Of those, 913 training and 51 validation conversations include tools. Training includes 173 error-recovery and 48 explicit escalation examples; validation adds 12 escalation examples. The training-only `tool-repeat=4` weighting repeats the original 173 information-dependent conversations; validation conversations are not repeated.

Each observation is produced by the actual read-only `DataTools` implementation. Training masks system/user/observation/padding tokens and supervises **every assistant turn**, including calls and final contracts. Sequences above 8,192 tokens fail explicitly instead of being truncated. The pinned tokenizer checks conversation boundaries before GPU work. Training validation uses all designated validation conversations. Automatically constructed additions still need independent review.

The interface exposes `profile_column`, `count_rows`, `inspect_rows` and `group_profile`. Supervised tool examples currently cover **profile_column**, including full-column and single equality-group scopes. The other three tools are described in the prompt but do not yet have dedicated training coverage. This is not a claim of arbitrary tool-use competence.

The model sees at most six tool attempts, ten replies, two invalid-response corrections and 24,000 cumulative parser tokens. The context limit is 12,288 tokens. A failed tool call is explicit; the parser can correct it or escalate. A schema-valid contract can still be returned without a call: this experiment measures actual model behavior rather than forcing a successful-looking tool count.

Training uses physical GPU 0; semantic evaluation uses physical GPU 1 in a fresh process. Each child sees one local GPU. The notebook includes the fixed warmup API and single-visible-GPU attention checks used to address the earlier failures. A separate smoke job runs before full training. It does not replace full checkpoint evaluation.

Candidate selection uses 80 synthetic open-development cases and 20 original development cases. Within each cohort, correct-reference acceptance must not decrease and wrong-output acceptance must not increase compared with the same-size base. Total joint successes must strictly increase. Otherwise `eligible` is false. Synthetic cases include targeted wrong outputs; original cases have only a wrong-no-op probe. These checks are diagnostic, not a safety guarantee or an independent benchmark.

## Results and scope

`summary.json` reports correct-reference acceptance, wrong-output acceptance, joint success, invalid outputs, tool attempts, tool failures, tasks using tools and total tokens. `tasks/*.json` contains the complete model/tool transcript. Training also includes `baseline/`, `semantic/`, `selection.json`, training logs, data/source hashes and the selected `adapter/`. Model and protocol mismatches and modified adapter files are rejected.

**These notebooks evaluate/train the intent parser; they do not run the six end-to-end Stage 2 systems.** Establish the chosen size/adapter on this development comparison before adding it to a new Stage 2 experiment. Existing Stage 2 notebooks expect different adapters and must not be pointed at these weights. The reusable inference entry point is:

```bash
PYTHONPATH=src:. python -m finetune.tool_use.infer \
  --size 1_5b --csv input.csv --request 'Fill missing Age with its full-column median.' \
  --out parser_result.json
```

Add `--adapter /path/to/new/tool/adapter` for the trained arm. GPU/model dependencies are required. Both paths return the same parser record and extracted contract structure, ready for a separately versioned Stage 2 integration.

Local validation covers the tool loop, masking, precision, recovery, split isolation, adapter seals and notebook/source packaging. Actual model training/inference cannot be verified locally because this workspace has no Torch/CUDA environment. Kaggle preflights and the smoke job check the real environment before the full job; they cannot guarantee a long run will encounter no runtime or quota issues.

Validation on implementation: 28 targeted CPU tests passed (`test_tool_use_v2.py`, `test_stage2_tools.py`, `test_selective_attention.py`). The source ZIP passed CRC and per-file SHA checks, extracted-package CLI imports passed, all three notebooks passed schema validation and code-cell compilation, and the original frozen-source verifier passed.

Completed direct-base evaluations are now analyzed in `results/tool_use_v2_analysis/base_comparison/ANALYSIS.md`. The 1.5B model passes 14/100 correct references and 10/100 joint positive/negative checks; 3B passes 6/100 and 4/100 respectively. These are weak unadapted parser baselines, not the outcomes of the new training jobs. The executed uploads were named with swapped sizes; organized filenames follow their verified `MODEL_SIZE` settings. See `docs/uploaded_artifacts.md` for locations.

The completed 1.5B training run is analyzed in `results/tool_use_v2_analysis/train_1_5b/ANALYSIS.md`: selected epoch 2 passes 90/100 correct references and 89/100 joint checks, with 16/16 statistic cases solved through tool observations. It also repeats a supervised recovery placeholder on all 16 statistic cases and admits one targeted wrong datetime conversion. Preserve and inspect these failures rather than interpreting `eligible: true` alone as deployment readiness. The source/data used by running or completed jobs have not been changed.
