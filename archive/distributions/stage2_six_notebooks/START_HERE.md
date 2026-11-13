# Stage 2: six standalone evaluations

Workflow version: **stage2-clean-v1**. This is a new six-system experiment. Each evaluation starts from the frozen research source and produces its own 100-task result bundle. Evaluation notebooks need no previous outputs. The shared runtime is embedded identically in all seven notebooks, so uploading the notebooks is sufficient; no GitHub push or adapter upload is required.

## 1. Download and extract the notebook pack

Get [stage2_six_notebooks.zip](../notebooks/kaggle/stage2_six_notebooks.zip) and extract it on your computer. Upload the `.ipynb` files to Kaggle individually. Keep this guide for the run order.

## 2. Run notebook 06 first

Upload **`stage2_06_no_contract_to_generator.ipynb`** as a fresh Kaggle notebook.

- Set **Internet: On**.
- Set **Accelerator: GPU T4 ×2**.
- Leave the notebook configuration as supplied. No input datasets or prior ZIPs are needed.
- Choose **Save Version → Save & Run All**.

The notebook verifies 218 frozen source/input hashes and the inference environment, tests the actual efficient attention kernel on both GPUs, and verifies model placement and four-bit repair weights. Expect six `kernel_self_test_passed` events. The first real task is **`taxi_dedup_003`**, the earlier memory failure. It is evaluated normally and counted exactly once. The remaining 99 tasks follow automatically.

Success means **`COMPLETE AND VALIDATED: no_contract_to_generator 100 / 100`**. A small kernel self-test alone is not proof that the full task fits. Wait for the entire notebook to finish before running the other systems.

Download **`stage2_06_no_contract_to_generator_bundle.zip`** and the executed notebook from that saved version's outputs. If it stops, retain the ZIP and the full `.log` file from the output. A failed infrastructure operation does not become a REJECT or a successful task record. Its latest completed tasks remain saved automatically. Do not spend GPU time on the other five systems until notebook 06 completes.

## 3. Run notebooks 01–05

Use a **fresh notebook/session for each**, with Internet and GPU T4 ×2. Run one at a time, using Save Version → Save & Run All. Each reports its own **100/100** completion; it does not expect the other five systems' outputs.

1. `stage2_01_baseline1.ipynb` — direct generation. Download `stage2_01_baseline1_bundle.zip`.
2. `stage2_02_baseline2_repair.ipynb` — conventional checks with repair. Download `stage2_02_baseline2_repair_bundle.zip`.
3. `stage2_03_baseline3_repair.ipynb` — prompted 7B parser contracts with repair. Download `stage2_03_baseline3_repair_bundle.zip`.
4. `stage2_04_gold_contract.ipynb` — gold-contract oracle comparison. Download `stage2_04_gold_contract_bundle.zip`.
5. `stage2_05_proposed.ipynb` — fine-tuned 1.5B contracts with contract-aware generation and repair. Download `stage2_05_proposed_bundle.zip`.

Keep one completed bundle per system, plus its executed notebook. Model downloads and cold caches add overhead, so the older 10–26 minute arm timings are not guaranteed durations for these independent runs.

## 4. Collect the six results

Upload **`stage2_collect_results.ipynb`** as a fresh notebook.

- Set **Accelerator: None**. Internet is not required.
- Attach the **six completed evaluation notebook outputs** as inputs, or upload the six output ZIPs as Kaggle inputs.
- Include one completed version of each system. ZIPs and automatically unpacked uploads are supported.
- Run All.

The collector checks configuration and environment equality, source/runtime hashes, exact task coverage, task/trace agreement, model and GPU self-test evidence, and stored metrics. It accepts an identical ZIP and extracted-folder view of the same output, but rejects conflicting versions, missing systems, incomplete runs and earlier experiments.

Success means **`COMPLETE: all six systems, 600 / 600 verified task records`**.

Download **`stage2_complete_results.zip`**. It contains every system's results and traces, the common experiment plan, GPU diagnostics, the combined `stage2_metrics_test.json`, paired comparisons, confidence intervals, and a spreadsheet-friendly `stage2_summary.csv`.

## What changed and why

- All six systems use the same memory-efficient attention implementation, with grouped key/value heads expanded explicitly. The previous failures were inside attention even after the models were placed on separate GPUs. This policy is uniform from the start of this new experiment.
- Generator: Qwen2.5-3B FP16 on GPU 1. Repair: Qwen2.5-7B NF4 on GPU 0. The direct-generation baseline loads only its generator. Model and tokenizer revisions are pinned to the same immutable Hugging Face commits across notebooks.
- Prompts remain the development-selected b2-v2 / a1-v4; request-only generation uses b1-v1. The 100 test tasks, parser inputs, seed 0, greedy decoding, three repair attempts, eight tool calls per attempt and 60,000-token budget remain fixed. The ablation measures the whole contract-aware generator prompt, not just contract text.
- Task results and traces are committed atomically together. Each completed task triggers a new output ZIP. Rerunning cells in the same session automatically retains matching work; a fresh session starts fresh. No separate resume notebook is required.
- Infrastructure failures are logged with the task ID, prompt length, GPU memory and traceback. They leave the arm explicitly incomplete. Previous failed attempts can remain in the diagnostics after a successful same-session rerun; only successfully completed task records enter the metrics.
- Existing experiments and their 562 records are preserved separately. The collector accepts only this new workflow. Attention kernels can change numerical outputs, so the new results should be reported with their runtime version.

## Validation and limits

Local tests exercise all six evaluator paths with scripted backends, compare their task outcomes and token accounting with the frozen evaluator, check atomic failure/retry handling, test result collection and rejection of mismatches, validate generated notebook syntax and shared payload identity, and test GQA expansion semantics. Scripted outputs are explicitly marked invalid as model-quality results and rejected by the normal collector.

The laptop has no usable local PyTorch/GPU setup for the real Qwen runs. Full T4 completion therefore remains to be verified by notebook 06. The workflow cannot guarantee that Kaggle will never encounter a dependency, network, quota or hardware failure. It detects these failures early where possible and preserves finished work.

Implementation: `eval/stage2_clean/`. Rebuild the notebook pack after any shared-runtime change with:

```bash
.venv/bin/python notebooks/kaggle/make_stage2_six_notebooks.py
```

Runtime changes produce a different fingerprint. Never mix bundles from two different builds. Kernel behavior follows [PyTorch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html); model revisions were verified against the [3B model metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct) and [7B model metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct).
