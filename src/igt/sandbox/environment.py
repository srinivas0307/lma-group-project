"""Explicit package roots and a trusted probe inside the actual subprocess sandbox.

Opt-in only. python -I ignores PYTHONPATH, so the package root is inserted in
the child runner itself, before pandas/numpy import. Model-generated code
still passes the original AST guard and uses the original limits/unpickler.
"""
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import executor

EXPECTED = {"pandas": "2.3.3", "numpy": "2.1.3"}
PROBE_CODE = '''
import sys
def transform(df):
    return pd.DataFrame({
        "pandas": [pd.__version__], "numpy": [np.__version__],
        "python": [sys.version.split()[0]], "executable": [sys.executable],
        "pandas_file": [pd.__file__], "numpy_file": [np.__file__],
    })
'''


@contextmanager
def sandbox_environment(package_root=None):
    """Yield verified child versions, restoring the original runner even on failure."""
    original_runner = executor._RUNNER
    root = Path(package_root).resolve() if package_root is not None else None
    if root is not None:
        for package in EXPECTED:
            if not (root / package / "__init__.py").is_file():
                raise ValueError(f"{root}: missing {package}; install requirements.txt into this target directory")
    try:
        if root is not None:
            executor._RUNNER = f"import sys\nsys.path.insert(0, {str(root)!r})\n" + original_runner
        # This fixed, trusted diagnostic deliberately uses run(), since __version__
        # attributes are prohibited in untrusted candidate code by the AST guard.
        result = executor.run(PROBE_CODE, pd.DataFrame({"probe": [0]}), timeout_s=30)
        if not result.ran_ok:
            raise RuntimeError(f"sandbox environment probe failed: {result.status}: {result.error}")
        info = result.df_out.iloc[0].to_dict()
        for package, version in EXPECTED.items():
            if info[package] != version:
                raise RuntimeError(f"sandbox {package} is {info[package]}, expected {version}; "
                                   "install requirements.txt and pass --sandbox-packages")
            if root is not None and not Path(info[f"{package}_file"]).resolve().is_relative_to(root):
                raise RuntimeError(f"sandbox loaded {package} outside {root}")
        info["package_root"] = str(root) if root is not None else None
        info["version"] = "sb1"
        yield info
    finally:
        executor._RUNNER = original_runner
