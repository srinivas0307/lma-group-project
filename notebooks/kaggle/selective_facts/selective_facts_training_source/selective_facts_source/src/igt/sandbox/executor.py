"""Run generated `transform(df)` code in a separate, limited process (README §4.5, Phase 3).

Layers (the AST guard in ast_guard.py runs before this, via `safe_run`):
1. separate process, started in its own session; the whole process group is killed on timeout
2. resource limits in the child: CPU seconds, address space (memory), max file size, open files
3. network: a new network namespace (`unshare -rn`) when the kernel allows it; otherwise the child
   replaces the socket API with stubs that raise (weaker, reported as network_isolated=False)
4. isolated interpreter (`python -I`), minimal environment, empty temporary working directory
5. the result DataFrame comes back as a pickle read with a restricted unpickler that only
   accepts pandas/numpy data classes, so a hostile child can't run code in the parent

Not covered: filesystem isolation. The child can still read files its user can read; the AST guard
is what blocks file access. Use Docker (`--network none`, read-only mount) if stronger isolation is needed.
"""
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .ast_guard import check

DEFAULTS = dict(timeout_s=60.0, cpu_s=30, mem_gb=4.0, max_file_mb=256, max_output_chars=2000)

_RUNNER = r'''
import contextlib, io, json, pickle, resource, sys, time, traceback
import numpy as np
import pandas as pd

in_pkl, out_pkl, res_path, code_path, func_name, cpu_s, mem_gb, fsize_mb, max_chars, net_ns = sys.argv[1:11]

def _no_network(*a, **k):
    raise PermissionError("network access is disabled in the sandbox")

import socket
class _NoNetSocket(socket.socket):          # stays a class so modules that subclass socket still import
    def __init__(self, *a, **k):
        _no_network()
socket.socket = _NoNetSocket
for _n in ("create_connection", "create_server", "socketpair", "fromfd", "getaddrinfo",
           "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
    if hasattr(socket, _n):
        setattr(socket, _n, _no_network)

df = pickle.load(open(in_pkl, "rb"))
code = open(code_path).read()
res = dict(status="error", error=None, stdout="")

def limit(kind, soft, hard=None):
    try:
        resource.setrlimit(kind, (soft, soft if hard is None else hard))
    except (ValueError, OSError):
        pass

cpu = int(cpu_s)
limit(resource.RLIMIT_CPU, cpu, cpu + 1)
limit(resource.RLIMIT_AS, int(float(mem_gb) * 1024 ** 3))
limit(resource.RLIMIT_FSIZE, int(fsize_mb) * 1024 ** 2)
limit(resource.RLIMIT_CORE, 0)

sink = io.StringIO()
t0 = time.time()
try:
    ns = {"pd": pd, "np": np, "__name__": "candidate"}
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(code, "<candidate>", "exec"), ns)
        fn = ns.get(func_name)
        if not callable(fn):
            raise NameError(f"function '{func_name}' is not defined")
        out = fn(df.copy())
    if isinstance(out, pd.DataFrame):
        with open(out_pkl, "wb") as fh:
            pickle.dump(out, fh, protocol=4)
        res["status"] = "ok"
    else:
        res.update(status="bad_return", error=f"returned {type(out).__name__}, not a DataFrame")
except MemoryError:
    res.update(status="memory", error="MemoryError")
except BaseException as e:
    res.update(status="error", error=f"{type(e).__name__}: {str(e)[:500]}")
res["duration_s"] = round(time.time() - t0, 4)
res["stdout"] = sink.getvalue()[: int(max_chars)]
with open(res_path, "w") as fh:
    json.dump(res, fh)
'''


@dataclass
class ExecResult:
    status: str                         # ok | error | bad_return | timeout | killed | memory | blocked
    df_out: pd.DataFrame | None = None
    error: str | None = None
    stdout: str = ""
    duration_s: float | None = None
    network_isolated: bool = False
    returncode: int | None = None
    limits: dict = field(default_factory=dict)

    @property
    def ran_ok(self) -> bool:
        return self.status == "ok" and isinstance(self.df_out, pd.DataFrame)


# ---------------- restricted unpickling of the child's result ----------------
_ALLOWED_PREFIXES = ("pandas.core.", "pandas._libs.", "numpy._core.", "numpy.core.")
_DENIED_PREFIXES = ("pandas.core.computation", "pandas.io")
_ALLOWED_EXACT = {("numpy", "dtype"), ("numpy", "ndarray"), ("builtins", "slice"), ("builtins", "complex"),
                  ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "range"), ("builtins", "bytearray"),
                  ("datetime", "datetime"), ("datetime", "date"), ("datetime", "timedelta"), ("datetime", "time"),
                  ("decimal", "Decimal")}
_DENIED_NAME_PARTS = ("read", "load", "eval", "exec", "query", "system", "open", "compile")


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        ok = (module, name) in _ALLOWED_EXACT or (
            module.startswith(_ALLOWED_PREFIXES) and not module.startswith(_DENIED_PREFIXES)
            and "." not in name and not any(p in name.lower() for p in _DENIED_NAME_PARTS))
        if not ok:
            raise pickle.UnpicklingError(f"sandbox result refers to disallowed global {module}.{name}")
        return super().find_class(module, name)


def load_result_pickle(path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        obj = _SafeUnpickler(fh).load()
    if not isinstance(obj, pd.DataFrame):
        raise pickle.UnpicklingError(f"sandbox result is {type(obj).__name__}, not a DataFrame")
    return obj


# ---------------- network namespace probe ----------------
@lru_cache(maxsize=1)
def network_namespace_available() -> bool:
    """True if `unshare -rn` works for this user (unprivileged user + network namespace)."""
    try:
        p = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
        return p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run(code: str, df: pd.DataFrame, func_name: str = "transform", *, python: str | None = None,
        timeout_s: float = DEFAULTS["timeout_s"], cpu_s: int = DEFAULTS["cpu_s"], mem_gb: float = DEFAULTS["mem_gb"],
        max_file_mb: int = DEFAULTS["max_file_mb"], max_output_chars: int = DEFAULTS["max_output_chars"],
        require_network_isolation: bool = False) -> ExecResult:
    """Execute `code` (which must define `func_name(df)`) on a copy of `df` in a limited child process.

    Does NOT run the AST guard; use `safe_run` for untrusted code.
    """
    limits = dict(timeout_s=timeout_s, cpu_s=cpu_s, mem_gb=mem_gb, max_file_mb=max_file_mb)
    net_ns = network_namespace_available()
    if require_network_isolation and not net_ns:
        return ExecResult(status="error", error="network namespace not available (unshare -rn failed)", limits=limits)
    with tempfile.TemporaryDirectory(prefix="igt_sandbox_") as tmp:
        tmp = Path(tmp)
        work = tmp / "work"
        work.mkdir()
        in_pkl, out_pkl, res_path, code_path, runner = (tmp / n for n in
                                                        ("in.pkl", "out.pkl", "result.json", "code.py", "runner.py"))
        with open(in_pkl, "wb") as fh:
            pickle.dump(df, fh, protocol=4)
        code_path.write_text(code)
        runner.write_text(_RUNNER)
        cmd = [python or sys.executable, "-I", str(runner), str(in_pkl), str(out_pkl), str(res_path), str(code_path),
               func_name, str(int(cpu_s)), str(mem_gb), str(int(max_file_mb)), str(int(max_output_chars)),
               str(int(net_ns))]
        if net_ns:
            cmd = ["unshare", "-rn", *cmd]
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "LANG": "C.UTF-8", "TMPDIR": str(work),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        try:
            _, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return ExecResult(status="timeout", error=f"no result after {timeout_s}s",
                              duration_s=round(time.time() - t0, 4), network_isolated=net_ns,
                              returncode=proc.returncode, limits=limits)
        finally:
            _kill_group(proc)        # also reaps anything the candidate forked
        if not res_path.exists():
            rc = proc.returncode
            sig = f" ({signal.Signals(-rc).name})" if rc is not None and rc < 0 else ""
            return ExecResult(status="killed", error=f"sandbox exited with code {rc}{sig} (CPU/memory limit?) "
                                                     f"{err.decode(errors='replace')[-300:]}",
                              duration_s=round(time.time() - t0, 4), network_isolated=net_ns, returncode=rc,
                              limits=limits)
        try:
            res = json.loads(res_path.read_text())
        except json.JSONDecodeError as e:
            return ExecResult(status="error", error=f"unreadable sandbox result: {e}", network_isolated=net_ns,
                              returncode=proc.returncode, limits=limits)
        df_out = None
        if res["status"] == "ok":
            try:
                df_out = load_result_pickle(out_pkl)
            except Exception as e:
                res.update(status="error", error=f"result rejected: {type(e).__name__}: {e}")
        return ExecResult(status=res["status"], df_out=df_out, error=res.get("error"), stdout=res.get("stdout", ""),
                          duration_s=res.get("duration_s"), network_isolated=net_ns, returncode=proc.returncode,
                          limits=limits)


def safe_run(code: str, df: pd.DataFrame, func_name: str = "transform", **kw) -> ExecResult:
    """AST guard, then sandboxed execution. The only entry point for untrusted code."""
    g = check(code)
    if not g.ok:
        return ExecResult(status="blocked", error=g.reason)
    return run(code, df, func_name, **kw)
