# Stage 2 prompt development: b2-v2 and a1-v5

The first real Stage 2 development run used commit `8ea400a`, generator b2-v1 and agent a1-v4. Direct generation was correct on 16/20 tasks; the gold-contract system was initially correct on 11/20 and finally correct on 14/20. Preserve that run unchanged. Its 20-task final-accuracy comparison is not statistically significant (paired McNemar p = 0.625).

Local replay of 58 recorded code snippets reproduced the initial accuracies under pandas 2.3.3. Failures included invented column identifiers, unnecessary dtype conversions, incorrect complements of compound filters, incompatible pandas APIs, repeated ineffective repairs, and a correct tool-tested candidate that was never submitted. The accepted-wrong Titanic repair retained a helper column: the existing contract vocabulary does not forbid extra columns.

## New, opt-in versions

- `src/igt/agents/code_generator_v2.py`: **b2-v2**, the b2-v1 prompt plus general implementation rules. Requests and contract formatting are unchanged. It contains no benchmark task examples, reference solutions or task-specific constants. Calls without a contract still use the original b1-v1 prompt.
- `src/igt/agents/repair_agent_v5.py`: **a1-v5**, additional full-agent repair instructions. The existing repair loop, tools, budget and acceptance policy remain unchanged. A successful tool test still requires `propose_fix`, followed by execution and verification.
- `eval/runner_stage2_prompts.py`: selects versions and delegates to the existing Stage 2 runner. Original modules are restored after success or failure. System names include the actual generator and agent versions and `sb1`.
- `src/igt/sandbox/environment.py`: **sb1**, explicitly places the pinned package target in the actual child runner before pandas/numpy imports. This is necessary because the child uses `python -I`, which ignores PYTHONPATH. A trusted probe executes inside that same sandbox and rejects mismatched library versions before loading models.

The old runner, agent, generator prompts, sandbox executor and existing notebook remain byte-for-byte unchanged. The verifier, vocabulary and benchmark are not altered. Existing extension features (np1, fs1, re-ask and vLLM) are not enabled.

## Kaggle development workflow

Use `notebooks/kaggle/stage2_prompt_dev_kaggle.ipynb`. Push these new files to the repository first; its defaults clone branch `extra`. It runs only the 20 development tasks and uses real HF inference by default.

Phase A runs three systems under the same verified environment and budgets:

1. Direct baseline, b1-v1 / a1-v4 (no repair).
2. Gold contract, b2-v1 / a1-v4.
3. Gold contract, b2-v2 / a1-v4.

This isolates generator-prompt development. Compare initial code accuracy first, then accepted-correct final accuracy, silent errors and repair cost. Select the generator using development evidence and record the choice; do not change selection criteria after viewing test results.

Phase B is off by default (`RUN_REPAIR_COMPARISON = False`). After Phase A, choose `GENERATOR_FOR_REPAIR_COMPARISON` and enable it. It adds a1-v5 with that same generator and compares against its a1-v4 arm, reusing completed matching runs. This isolates repair-prompt development. It does not claim that prompt instructions guarantee better model behavior.

The notebook installs the repository's pinned data requirements into a separate package target. The runner records actual sandbox Python/pandas/numpy paths and versions, inference-package versions, generator hash, agent hash and budgets in its resume signature. It refuses to resume if these differ. Parent pandas/numpy are also checked against the pinned versions. CUDA packages are not automatically upgraded; the actual installed inference versions are recorded.

Output: `/kaggle/working/results/stage2_prompt_dev/`, bundled as `stage2_prompt_dev_bundle.zip`. Every system has `runs.jsonl`, `traces.jsonl`, `run_info.json`, `prompt_run_info.json`, and `git_commit.txt`; the bundle also contains metrics, console logs, a complete input/prompt-source checksum manifest and environment metadata. Download the bundle before ending the session.

The new comparison has a different verified execution environment version from the original uploaded run. Rerun all Phase A arms together; do not compare an old unverified arm against only the new arm as if environment were held fixed. Use a fresh output/cache location for a different inference environment.

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_stage2_prompt_versions.py tests/test_stage2.py
.venv/bin/python eval/runner_stage2_prompts.py \
  --preset gold_contract --splits dev --backend pipeline_check \
  --generator-prompt b2-v2 --agent-prompt a1-v4 \
  --limit 2 --out /tmp/stage2_prompt_dev
```

`pipeline_check` uses scripted responses containing reference solutions and is explicitly recorded as **not a valid model-quality result**. It checks wiring, metadata, sandbox execution and isolation only. No new training or GPU evaluation has been performed locally.

After completing development, freeze the selected prompt hashes, environment and budgets before running the held-out test comparison. Keep the extra-column verifier gap disclosed. Any future schema or verifier change needs a separately versioned comparison across systems.
