"""Load the notebook cells into one namespace (CPU-only parts), exactly as the notebook runs them."""
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
CPU_CELLS = ["c02_setup.py", "c03_tasks.py", "c04_contract.py", "c05_verifier.py", "c06_sandbox.py",
             "c08_parser_common.py", "c13_evaluate.py"]

def load(extra=(), scratch="/tmp/claude-1000/-home-srinivas-billuri-4-1-LMA-intent-parser/7fa0dbd3-d71d-47bf-8585-883ceb58c1ab/scratchpad/work"):
    ns = {"RUN_MODELS": False, "WORK_DIR": scratch, "INPUT_SEARCH_ROOTS": [], "DATA_ROOT": str(HERE.parent),
          "EXEC_TIMEOUT_S": 60, "__name__": "nb"}
    for c in CPU_CELLS + list(extra):
        p = HERE / "cells" / c
        if p.exists():
            exec(compile(p.read_text(), str(p), "exec"), ns)
    return ns
