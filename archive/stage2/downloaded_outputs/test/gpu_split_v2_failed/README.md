# Failed GPU-split recovery checkpoint

`archive/stage2/test/root_checkpoint/stage2_test_bundle.zip` is the checkpoint linked in the user's executed
`archive/notebooks/executed/stage2_test/stage2_test_resume_uploaded.ipynb` log. It retains 562/600 records: five complete systems
and 62 records for `no_contract_to_generator`.

The v2 intervention records verify repair-model parameters on `cuda:0` and
generator parameters on `cuda:1`. Repair generation still failed with CUDA OOM
inside SDPA. No new task completed. All 22 existing run, trace and per-system
metadata files match the original root `archive/stage2/test/root_checkpoint/stage2_test_bundle.zip` byte for byte.

Preserve this archive. The current recovery notebook consumes the same frozen
plan and adds a documented attention-kernel intervention; see
`docs/history/stage2_test_run.md`.
