# End-to-end Stage 2 with combined 3B repair

Both parser methods are integrated:

- **Selective facts:** existing selected 1.5B selective-facts adapter and automatic facts input → 3B FP16 code generation → combined 3B FP16 repair.
- **Trained tool use:** existing selected 1.5B tool adapter chooses dataset queries → 3B FP16 code generation → the same combined 3B FP16 repair.

Combined repair is the frozen version that achieved 16 correct, zero incorrect accepted and one withheld on the 17-case repair-development suite, compared with 11/1/5 for control. It uses compact failure evidence, one bounded fresh-code recovery and the declared-output-column guard. That small development result does not predict full Stage 2 accuracy.

Experiment ID: `stage2-combined-3b-fp16-v1`. Parser input protocols, adapter weights, parser NF4 loading, generator prompt, models, repair proposal/tool/token limits and benchmark inputs match the completed 3B control workflow. The new repair prompt and schema guard are the method changes. Parser correction, grouped-statistic routing and the ordinary-word “mean” fix are not part of this integration.

## Files for Kaggle

All files are under `notebooks/kaggle/stage2_combined_3b/`:

- `stage2_combined_3b_source.zip`: the shared, small source/data package for both methods.
- `stage2_combined_3b_selective_proposed_kaggle.ipynb`: selective-facts end-to-end run.
- `stage2_combined_3b_tools_proposed_kaggle.ipynb`: trained-tool end-to-end run.
- `stage2_combined_3b_collect_kaggle.ipynb`: optional CPU comparison after both full runs.

Each run has its own summary, so a separate collector is not required to inspect either result. The collector is useful for a strict paired comparison and refuses older, partial, scripted or smoke bundles.

## Run instructions

1. Upload the new `stage2_combined_3b_source.zip` as one Kaggle dataset.
2. Import the selective-facts notebook and attach that source dataset plus **the selected selective-facts adapter dataset from your completed 3B selective run**.
3. Enable Internet and T4 GPU (one T4 or T4 x2). Run the setup and development cell first. It completes a two-task smoke and the 20 development tasks, displaying the development summary.
4. Review the development result. The final cell runs the 100 diagnostic test tasks freshly. Save Version → Run All runs smoke, development and full test sequentially in one session; it does not pause for a quality review or impose an accuracy threshold.
5. Download the full ZIP, development ZIP, and executed notebook. Run the tool notebook in a separate fresh session with the same source dataset and **the selected trained tool-adapter dataset from your completed 3B tool run**.
6. Optionally run the CPU collector with the new source dataset and both full result bundles, uploaded as separate datasets. It also discovers automatically extracted result folders.

No GitHub push, previous parser prediction file, old Stage 2 output, training run or output from the other notebook is required. The two methods can run independently or in separate GPU sessions in parallel. Each stage starts fresh; smoke/development predictions are not reused in the test run.

Use the **new combined source ZIP** with the new notebooks. An older source ZIP will not pass the manifest check. Existing adapter datasets can contain older source files; discovery selects only the exact new source manifest and validates the required adapter files by hash.

The selected adapters are the later selective-facts and trained-tool adapters, **not merely any original Qwen 1.5B adapter**. Adapter files must include selection metadata as in the dataset used by the completed runs. There is no 3B parser option in this integration: the parser is the selected 1.5B method-specific adapter; generation and repair are 3B FP16.

If you need a new single-upload adapter/source dataset, use the optional builder:

```bash
.venv/bin/python kaggle_uploads/stage2_combined_3b/make_kaggle_datasets.py
```

It creates `stage2_combined_3b_selective_kaggle_dataset.zip` and `stage2_combined_3b_tools_kaggle_dataset.zip`. Each includes only the new source/data and the exact selected adapter files, avoiding nested training/checkpoint archives. These large duplicate upload ZIPs are not generated or retained by default after the repository cleanup.

## Expected outputs

Selective-facts full run: `stage2_combined_3b_selective_proposed.zip`.
Tool full run: `stage2_combined_3b_tools_proposed.zip`.

Each notebook also produces a `_proposed_dev.zip` containing 20 development records and a `_proposed_smoke.zip` containing two smoke records. Download the `_dev.zip` as well as the full ZIP so development behavior can be inspected. Run directories have timestamps, and partial bundles are exported as work progresses. If an infrastructure error occurs, download the log and partial ZIP; a partial run is not a completed result.

Both full bundles contain fresh contracts, initial generations, per-task repair traces, `summary.json`, completion metadata and SHA-256 seals. Expected full coverage is 100 records per method. The optional collector produces `stage2_combined_3b_comparison.zip` with the two-method summary.

Paste future uploads under:

- `kaggle_outputs/stage2/combined_3b_v1/` for original result ZIPs.
- `notebooks/executed/stage2/combined_3b_v1/` for executed notebooks.

Do not rename old 3B outputs to the new names. Their source and experiment metadata identify a different repair version.

## What the new measurements tell us

Compare against the corresponding earlier 3B runs (73 correct, two incorrect accepted, 25 withheld for each method). Check actual parser outputs and initial generated code before attributing differences solely to repair. The configuration preserves those stages, but saved outputs still need comparison after the GPU runs.

The summary records correct acceptance, incorrect acceptance, withholding, lost initially correct outputs, repairs, parser costs, parser tool failures, repair recovery requests, automatic diagnostic time and total recorded model tokens. It retains both the original parser/reference verifier result and the combined guard/reference result for audit. Neither reference verdict is supplied to the model or used to select live repair actions.

The schema guard may reveal contract problems and increase withholding. A change is a successful repair only when the accepted result matches the reference; merely preventing a false acceptance is reported separately. Wrong intent columns and contradictory contracts can still block correct code.

The 100 test tasks have repeatedly informed development; label follow-up results as diagnostic. This integration currently supports the two proposed methods, not a new completed six-baseline experiment. A wider architecture claim still requires matching baselines and fresh held-out evaluation.

## Verification and source boundaries

Code is under `eval/stage2_combined_3b/`. It invokes the frozen combined repair implementation in `eval/stage2_repair_dev/`; the existing 3B parser and backend are reused without changes. Earlier notebooks, source bundles and runtime hashes remain valid.

The local tests cover both complete staged pipelines with scripted models, helper-column repair, parser stops, version isolation, collector accounting, one resident model at a time, no reference fields in live messages, and intact/extracted Kaggle inputs. The exact source pack also passes an isolated CPU parser/repair preflight and two independent scripted pipeline runs through its collector.

These are infrastructure checks, not a GPU quality result. The actual 3B model and Kaggle environment are tested by the supplied notebooks. Full-run metadata records actual model revisions, FP16 weight precision, environment and source/input hashes.

Rebuild/check from the repository root:

```bash
PYTHONPATH=.:src .venv/bin/python notebooks/kaggle/stage2_combined_3b/make_notebooks.py
.venv/bin/python -m pytest -q tests/test_stage2_combined_3b.py tests/test_stage2_repair_dev.py
.venv/bin/python tools/validate_stage2_combined_3b.py
```

Always upload the notebook and source ZIP from the same build. The notebook pins the source manifest hash.
