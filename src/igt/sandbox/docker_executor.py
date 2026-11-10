"""Docker sandbox backend (extension #5, id `docker`; docs/extensions.md). Off by default.

Same contract as igt.sandbox.executor.run / safe_run (same child script, same ExecResult, same restricted unpickler
for the result), but the child runs in a container:
    --network none                 no network at all (closes OI-6: the subprocess sandbox uses a socket stub when
                                   `unshare -rn` isn't available)
    --read-only, --tmpfs /tmp      read-only root filesystem; the project and home folders are not visible
    inputs mounted read-only       /in: input pickle, code, child script
    /out (the only writable mount) result.json + out.pkl, read back with executor.load_result_pickle
    --memory/--memory-swap, --cpus 1, --pids-limit 64, --cap-drop ALL, no-new-privileges, host uid:gid (not root)
plus the same rlimits inside (CPU seconds, address space, file size) and the timeout (docker kill).

    from igt.sandbox.docker_executor import docker_safe_run, docker_available
    r = docker_safe_run(code, df)                         # AST guard, then the container
    # or select it for every safe_run call of a process that uses igt.sandbox.select_sandbox: IGT_SANDBOX=docker

Image: sandbox/docker/Dockerfile (python 3.12 + pandas 2.3.3 / numpy 2.1.3); build it once with
    docker build -t igt-sandbox:docker1 sandbox/docker
"""
import json
import os
import pickle
import subprocess
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .ast_guard import check
from .executor import _RUNNER, DEFAULTS, ExecResult, load_result_pickle

IMAGE = os.environ.get("IGT_DOCKER_IMAGE", "igt-sandbox:docker1")
DOCKER_SANDBOX_VERSION = "docker1"


def _tmp_root():
    """Folder for the per-run bind mounts. Not /tmp: a snap-installed Docker daemon has its own private /tmp and
    can't see the host's. Default <project>/.docker_tmp (git-ignored); override with IGT_DOCKER_TMP."""
    d = Path(os.environ.get("IGT_DOCKER_TMP", Path(__file__).resolve().parents[3] / ".docker_tmp"))
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@lru_cache(maxsize=1)
def docker_available(image=IMAGE) -> bool:
    """True if the docker CLI works and the sandbox image exists locally."""
    try:
        p = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=30)
        return p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def no_new_privileges_supported(image=IMAGE) -> bool:
    """--security-opt no-new-privileges fails on some hosts ("exec ... operation not permitted", e.g. snap Docker with
    AppArmor); probe once and leave it out there (recorded in ExecResult.limits). The other limits still apply."""
    try:
        p = subprocess.run(["docker", "run", "--rm", "--security-opt", "no-new-privileges", image, "python", "-c", "0"],
                           capture_output=True, timeout=60)
        return p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def docker_run(code: str, df: pd.DataFrame, func_name: str = "transform", *, image: str = IMAGE,
               timeout_s: float = DEFAULTS["timeout_s"], cpu_s: int = DEFAULTS["cpu_s"], mem_gb: float = DEFAULTS["mem_gb"],
               max_file_mb: int = DEFAULTS["max_file_mb"], max_output_chars: int = DEFAULTS["max_output_chars"],
               startup_s: float = 30.0, **_ignored) -> ExecResult:
    """Execute code in a container. Does NOT run the AST guard; use docker_safe_run for untrusted code.
    startup_s is added to the timeout for container start-up (the candidate's own time is limited by cpu_s)."""
    limits = dict(timeout_s=timeout_s, cpu_s=cpu_s, mem_gb=mem_gb, max_file_mb=max_file_mb, backend="docker", image=image)
    with tempfile.TemporaryDirectory(prefix="igt_docker_", dir=_tmp_root()) as tmp:
        tmp = Path(tmp)
        ind, outd = tmp / "in", tmp / "out"
        ind.mkdir()
        outd.mkdir()
        with open(ind / "in.pkl", "wb") as fh:
            pickle.dump(df, fh, protocol=4)
        (ind / "code.py").write_text(code)
        (ind / "runner.py").write_text(_RUNNER)
        for p in (ind, *ind.iterdir()):
            p.chmod(0o755 if p.is_dir() else 0o644)
        name = f"igt-sbx-{uuid.uuid4().hex[:12]}"
        mem = f"{int(float(mem_gb) * 1024 + 256)}m"          # a little above RLIMIT_AS, so the rlimit fires first
        nnp = no_new_privileges_supported(image)
        limits["no_new_privileges"] = nnp
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,size=64m,mode=1777", "--memory", mem, "--memory-swap", mem, "--cpus", "1",
               "--pids-limit", "64", "--cap-drop", "ALL", *(["--security-opt", "no-new-privileges"] if nnp else []),
               "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=1",
               "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
               "-v", f"{ind}:/in:ro", "-v", f"{outd}:/out:rw", "-w", "/tmp", image,
               "python", "-I", "/in/runner.py", "/in/in.pkl", "/out/out.pkl", "/out/result.json", "/in/code.py",
               func_name, str(int(cpu_s)), str(mem_gb), str(int(max_file_mb)), str(int(max_output_chars)), "1"]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout_s + startup_s)
            rc, err = p.returncode, p.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return ExecResult(status="timeout", error=f"no result after {timeout_s}s", duration_s=round(time.time() - t0, 4),
                              network_isolated=True, limits=limits)
        res_path = outd / "result.json"
        if not res_path.exists():
            return ExecResult(status="killed", error=f"container exited with code {rc} (CPU/memory limit?) {err[-300:]}",
                              duration_s=round(time.time() - t0, 4), network_isolated=True, returncode=rc, limits=limits)
        try:
            res = json.loads(res_path.read_text())
        except json.JSONDecodeError as e:
            return ExecResult(status="error", error=f"unreadable sandbox result: {e}", network_isolated=True,
                              returncode=rc, limits=limits)
        df_out = None
        if res["status"] == "ok":
            try:
                df_out = load_result_pickle(outd / "out.pkl")       # restricted unpickler, as in the subprocess sandbox
            except Exception as e:
                res.update(status="error", error=f"result rejected: {type(e).__name__}: {e}")
        return ExecResult(status=res["status"], df_out=df_out, error=res.get("error"), stdout=res.get("stdout", ""),
                          duration_s=res.get("duration_s"), network_isolated=True, returncode=rc, limits=limits)


def docker_safe_run(code: str, df: pd.DataFrame, func_name: str = "transform", **kw) -> ExecResult:
    """AST guard, then the container."""
    g = check(code)
    if not g.ok:
        return ExecResult(status="blocked", error=g.reason)
    return docker_run(code, df, func_name, **kw)


def select_safe_run(backend=None):
    """'subprocess' (default) or 'docker'; default from IGT_SANDBOX. Returns a safe_run-compatible function."""
    backend = backend or os.environ.get("IGT_SANDBOX", "subprocess")
    if backend == "docker":
        if not docker_available():
            raise RuntimeError(f"IGT_SANDBOX=docker but image {IMAGE} is missing: docker build -t {IMAGE} sandbox/docker")
        return docker_safe_run
    if backend == "subprocess":
        from .executor import safe_run
        return safe_run
    raise ValueError(f"unknown sandbox backend {backend}")
