"""Baselines and the generator ablation re-run on the current 3B stack (contract-check-v1 + repair-hints-v1).

Systems follow the six-system plan (eval/runner_stage2.py PRESETS): only proposed and gold give the contract to the
generator; baseline 3 and no_contract_to_generator use their contract for checking, correction and repair only;
baseline 2 uses conventional data-quality checks and no contract at all.
"""
