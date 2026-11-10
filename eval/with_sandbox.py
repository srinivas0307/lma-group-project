"""Run any project script with the Docker sandbox instead of the subprocess sandbox (extension #5), without editing it.

    IGT_SANDBOX=docker .venv/bin/python eval/with_sandbox.py eval/runner_agent.py --system agent_full_docker ...
    .venv/bin/python eval/with_sandbox.py --sandbox docker eval/safety_table.py

igt.sandbox.safe_run is replaced by docker_safe_run before the script is loaded, so every module that imports it
afterwards (runners, the repair agent, the tools) runs untrusted code in the container. Default sandbox: subprocess
(then this wrapper changes nothing). Write results to a new folder: Docker runs are their own version.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def install(backend):
    import igt.sandbox as sb
    import igt.sandbox.executor as ex
    from igt.sandbox.docker_executor import select_safe_run
    fn = select_safe_run(backend)
    sb.safe_run = fn
    ex.safe_run = fn
    return fn


def main():
    argv = sys.argv[1:]
    backend = os.environ.get("IGT_SANDBOX", "subprocess")
    if argv[:1] == ["--sandbox"]:
        backend, argv = argv[1], argv[2:]
    if not argv:
        sys.exit(__doc__)
    install(backend)
    print(f"[with_sandbox] sandbox = {backend}", flush=True)
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
