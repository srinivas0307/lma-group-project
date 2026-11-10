"""Functions shared by the demo UI (extension #6, app/demo_app.py) and the CLI demo (eval/demo.py, unchanged).

Each function runs one eval/demo.py mode and returns (printed text, result dict), so the UI shows exactly what the
CLI prints. No GPU needed except for live().
"""
import contextlib
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _capture(argv):
    from eval import demo
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = demo.main(argv)
    except SystemExit as e:                       # argparse / demo errors: show them instead of closing the app
        return buf.getvalue() + f"\n[error] {e}", None
    except Exception as e:
        return buf.getvalue() + f"\n[error] {type(e).__name__}: {e}", None
    return buf.getvalue(), res


@lru_cache(maxsize=1)
def task_ids():
    import yaml
    return [yaml.safe_load(f.read_text())["id"] for f in sorted((ROOT / "benchmark" / "tasks").glob("*/*.yaml"))]


@lru_cache(maxsize=1)
def _pool():
    from igt.paths import POOL
    return [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()]


def candidates(task_id):
    """Pool candidates of a task, labelled for a dropdown: (label, candidate_id)."""
    out = []
    for c in _pool():
        if c["task_id"] == task_id:
            label = f"{c['candidate_id']}  ({'ran' if c['ran_ok'] else c['status']}, {'correct' if c['is_correct'] else 'wrong'})"
            out.append((label, c["candidate_id"]))
    return out


def contract_sources():
    from igt.paths import PARSING
    return ["gold"] + [f"parsed:{p.relative_to(ROOT)}" for p in sorted(PARSING.glob("*/parses.jsonl"))]


def run_folders():
    folders = []
    for base in ("results/stage2_agent", "results/stage2"):
        folders += [str(p.parent.relative_to(ROOT)) for p in sorted((ROOT / base).glob("*/runs.jsonl"))]
    return folders


def run_ids(folder):
    p = ROOT / folder / "runs.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r.get("scenario_id") or r.get("task_id") for r in rows]


def task(task_id, candidate=None, contract="gold"):
    argv = ["task", task_id, "--contract", contract] + (["--candidate", candidate] if candidate else [])
    return _capture(argv)


def replay(folder, run_id):
    return _capture(["replay", folder, run_id])


_BACKENDS = {}


def _install_backend_cache():
    """Keep loaded models between Live calls (eval/demo.py loads them per call)."""
    import igt.llm.hf as hf
    if getattr(hf.HFChatBackend, "_igt_cached", False):
        return
    real = hf.HFChatBackend

    class Cached:
        _igt_cached = True

        def __new__(cls, model_id, dtype="float16", load_in_4bit=False, adapter=None):
            k = (model_id, dtype, load_in_4bit, str(adapter))
            if k not in _BACKENDS:
                _BACKENDS[k] = real(model_id, dtype=dtype, load_in_4bit=load_in_4bit, adapter=adapter)
            return _BACKENDS[k]
    hf.HFChatBackend = Cached


def live(data_path, request, parser_model, generator_model, agent_model, adapter=None, load_in_4bit=True,
         generator_contract=True, repair=True, backend="hf", scripted=None):
    """scripted: dict(contract=path, code=path, agent=path) for tests (backend='scripted')."""
    argv = ["live", "--data", str(data_path), "--request", request, "--backend", backend,
            "--parser-model", parser_model, "--generator-model", generator_model, "--agent-model", agent_model,
            "--generator-contract" if generator_contract else "--no-generator-contract",
            "--repair" if repair else "--no-repair"]
    argv += ["--load-in-4bit"] if load_in_4bit else []
    argv += ["--adapter", str(adapter)] if adapter else []
    for k, v in (scripted or {}).items():
        argv += [f"--scripted-{k}", str(v)]
    if backend == "hf":
        _install_backend_cache()
    return _capture(argv)


def gpu_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
