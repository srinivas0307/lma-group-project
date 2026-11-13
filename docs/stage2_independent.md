# Independent Stage 2 workflow

Status: implementation and CPU pipeline checks ready. Actual model execution on Kaggle has not been tested for this new workflow.

This workflow starts from raw CSVs, task requests, executable source and model weights. You do **not** need Stage 1, parser pilot runs, previous parses, tuned threshold files, generated-code pools, previous Stage 2 outputs or a GitHub push. Every system produces its own fresh results in its own notebook job. The collector only combines finished results; no system consumes the collector's outputs.

The proposed system still needs your original trained 1.5B adapter. This is the trained parser model, not a saved contract or evaluation result. Its weights are not included in the notebook pack. Base model weights are downloaded from Hugging Face.

## What each run does

1. Unpack the relevant source and raw benchmark inputs embedded in the notebook.
2. Read the CSV and compute schema/sample information directly from it.
3. If a parser is needed, load the model and generate contracts **inside this job** from request + schema + input-data profile. Save the exact messages and fresh raw replies under `contracts/` for inspection.
4. Unload the parser before loading the generator and repair models. This avoids keeping all three models in GPU memory together.
5. Generate code, execute it in the safety-checked sandbox, verify and repair it.
6. After the runtime finishes each task, execute the reference in the evaluator and label the outcome. References never enter proposed-system prompts, tool observations or repair decisions.
7. Save traces, parser costs, generation/repair costs, final correctness and the reference-output contract check. Export an atomic ZIP after each task.

The parsing phase and transformation phase belong to the same job. Saving the contracts there makes the run auditable; the runner never imports another job's contracts, and requires a fresh output directory. No disk cache of LLM responses is used.

## Run on Kaggle

Download `notebooks/kaggle/stage2_independent/stage2_independent_notebooks.zip` and extract it. Use **these seven new notebooks**, not the earlier parser-pilot notebooks or frozen Stage 2 notebooks.

For the proposed system first:

1. Create a private Kaggle dataset containing **all files** from `finetune/runs/qwen25_1_5b/adapter/`, for example under `qwen25_1_5b/adapter/`. You do not need the 3B adapter.
2. Upload `stage2_05_proposed.ipynb` to Kaggle. Attach that adapter dataset. Select **GPU T4 ×2**, turn **Internet on**, then **Save Version → Save & Run All** in a fresh session.
3. The notebook first runs two tasks as an infrastructure check, then creates a separate full 100-task run from scratch. It generates all 100 contracts itself before running generation/repair. Do not treat the smoke output as a model-quality result.
4. Wait for **COMPLETE AND VALIDATED: 100 / 100**. Download `stage2_independent_proposed.zip` and the executed notebook. An exception or incomplete job is not a valid result; keep its log and partial ZIP for diagnosis.

For the full comparison, run the other five GPU notebooks in separate fresh sessions:

- `stage2_01_baseline1.ipynb`: direct generation; no adapter.
- `stage2_02_baseline2_repair.ipynb`: conventional checks + repair; no adapter.
- `stage2_03_baseline3_repair.ipynb`: freshly prompted 7B contracts + repair; no adapter.
- `stage2_04_gold_contract.ipynb`: gold contracts from task annotations; no adapter. This is an oracle comparison, not a deployable system.
- `stage2_06_no_contract_to_generator.ipynb`: fresh fine-tuned contracts used by the verifier/repair, but not shown to the initial generator; attach the same 1.5B adapter.

Every GPU notebook uses T4 ×2 and Internet on. None needs another notebook's output. You can run only the proposed notebook first to inspect it; run all six for a full comparison.

Then upload `stage2_independent_collect.ipynb`, select **None** for accelerator, enable Internet, and attach the six full ZIP outputs. It requires exactly one distinct complete ZIP per system from the same notebook pack. Wait for **COMPLETE: 600 fresh task records verified.** Download `stage2_independent_summary.zip` and the executed collector.

## Where to put outputs

- All full GPU ZIPs and the summary ZIP: `kaggle_outputs/stage2/independent_v1/`.
- Executed notebooks: `notebooks/executed/stage2/independent_v1/`.
- Keep the template notebooks in `notebooks/kaggle/stage2_independent/` unchanged.

Inside a GPU ZIP, `contracts/` contains fresh parser messages/contracts for parser-based systems; `tasks/` contains results and traces; `plan.json` records method, model revisions, source/input hashes and settings. `failure.json` appears if a job failed. The summary ZIP contains `SUMMARY.md` and `summary.json`. Parser token costs are included in total costs.

## Experiment differences and limits

This is a new method, not an exact reproduction of the earlier six-system experiment:

- Proposed and ablation use the existing 1.5B adapter with a new input profile. Its weights have not been retrained for that format. The historical base revision was not recorded, so a revision is resolved and recorded per job.
- Prompted 7B uses zero-shot p1-v3 plus the profile. It does not reuse previous few-shot parser results.
- Conventional checks use the project's hand-set default configuration, explicitly recorded as `hand-set-untuned-v1`; they do not load the earlier tuned thresholds. Do not compare this baseline to the old tuned baseline as if only the parser changed.
- The default profile includes full-column statistics, bounded category information and selected single-column group profiles. Omitted information is explicitly marked. Arbitrary conditional and multi-column queries still need the later tool/hybrid design. Context overflow raises an error instead of silently truncating evidence.
- References and gold annotations are evaluation inputs, except in the clearly labelled gold-contract oracle arm. The proposed runtime does not receive them. The post-run reference check diagnoses false rejection but does not prove that a passing contract rejects every wrong implementation.
- The original 100 test tasks were already inspected during failure analysis. These new runs are diagnostic improvements, not fresh confirmatory evidence. Later use a reviewed independent benchmark.
- CPU/scripted tests check wiring, output independence, bundling and model-facing boundaries. They do not establish GPU compatibility or quality improvements. The notebooks check the pinned Torch/Kaggle configuration before real inference; if setup fails, share its log rather than silently changing method settings.

## Local commands and maintenance

To regenerate the pack after source changes:

```bash
.venv/bin/python notebooks/kaggle/stage2_independent/make_notebooks.py
.venv/bin/python -m pytest tests/test_stage2_independent.py -q
```

Local collection of downloaded full runs:

```bash
.venv/bin/python -m eval.stage2_independent.collect \
  kaggle_outputs/stage2/independent_v1/stage2_independent_baseline1.zip \
  kaggle_outputs/stage2/independent_v1/stage2_independent_baseline2_repair.zip \
  kaggle_outputs/stage2/independent_v1/stage2_independent_baseline3_repair.zip \
  kaggle_outputs/stage2/independent_v1/stage2_independent_gold_contract.zip \
  kaggle_outputs/stage2/independent_v1/stage2_independent_proposed.zip \
  kaggle_outputs/stage2/independent_v1/stage2_independent_no_contract_to_generator.zip \
  --out results/stage2_independent_v1
```

The old frozen experiment and parser pilot remain available separately. Neither is a prerequisite for this workflow. Reusing tested source modules is code reuse; this new workflow does not depend on their experiment outputs.
