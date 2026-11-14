# Normalization and one correction: implementation status

The separate inference-only implementation is in `finetune/selective_facts_correction/`. Use the [run instructions](../../../docs/selective_facts_correction.md). It reuses the exact selected epoch-2 adapter and corrected facts input; no retraining is performed.

## Verified offline result

On the saved scope-fix predictions, trimming only recognized operator whitespace recovers the three invalid numeric-filter contracts: **85/100 → 88/100** correct-reference passes with zero accepted tested wrong outputs. The three modified predictions were replayed on CPU against their reference/negative transformations. The 85 existing passes were unchanged. This is postprocessing of saved outputs, not a fresh GPU score.

The input-only diagnostics flag ten other failures and none of the 85 saved passing contracts. They flag none of the 80 synthetic gold development contracts. Two semantic errors remain unflagged: the schema-valid wrong arithmetic/new-column contract in `warehouse_edit_2` and the wrong real column in `airbnb_filt_006`. No task-specific correction was added.

One-correction quality is **pending GPU evaluation**. The evaluation freshly generates a first answer for every task, compares raw control/normalized/corrected arms, and allows at most one additional model reply when diagnostics trigger. It cannot use reference scores to decide whether to retry or which output to retain. A failed revised contract is withheld and recorded; the verifier is unchanged.

## Validation

25 targeted tests passed, covering normalization boundaries, literal/unknown-field preservation, input-consistency diagnostics, retained-row scopes, no reference leakage, correction bounds, context failures, real-verifier integration with scripted replies, attention helpers and both notebook platforms' packed/unpacked source loading. The isolated source ZIP passed checksums, CLI imports and CPU preflight. Discovery/validation of the actual supplied epoch-2 adapter passed from that package. Original frozen and scope-fix runtime hashes are unchanged.

The notebooks also record actual Python/Torch/CUDA/GPU versions inside the new result ZIP. No new GPU scores are claimed locally.

Reproduce the saved-output analysis with `.venv/bin/python tools/analyze_selective_correction_offline.py`. Detailed changes/triggers are in `offline_audit.json`.

The GPU experiment is now complete. Fresh paired results are 85/100 control, 88/100 normalized and 88/100 corrected, with zero accepted tested wrong outputs. Correction adds ten replies and no recovered cases. The completed audit and recommendation are in `completed/ANALYSIS.md` and `completed/audit.json`; the pending status above describes the earlier implementation stage only.
