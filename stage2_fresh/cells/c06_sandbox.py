# ===== Safety checker + sandboxed execution =====
ALLOWED_IMPORTS = {"pandas", "numpy", "re", "math", "datetime", "string"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "input", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview", "os", "sys",
                   "subprocess", "shutil", "socket", "pickle", "importlib", "builtins"}
FORBIDDEN_ATTRS = {"to_csv", "to_pickle", "to_parquet", "to_excel", "to_sql", "to_hdf", "to_feather", "to_json",
                   "to_clipboard", "read_csv", "read_pickle", "read_parquet", "read_excel", "read_sql", "read_json",
                   "read_html", "read_clipboard", "read_table", "read_fwf", "read_hdf", "system", "popen",
                   "unlink", "rmtree", "save", "savetxt", "savez", "savez_compressed", "load", "loadtxt", "genfromtxt",
                   "fromfile", "tofile", "memmap", "read_feather", "read_xml", "read_stata", "read_sas", "read_spss",
                   "read_orc", "read_sql_query", "read_sql_table", "to_html", "to_latex", "to_markdown", "to_xml",
                   "to_stata", "to_orc", "to_gbq"}


def extract_code(text):
    """The last fenced block that defines transform(df). Prose/CSV blocks are ignored; a stray first line
    'python' inside the fence is dropped; an unterminated block (cut-off reply) is accepted."""
    cands = []
    for body in re.findall(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(.*?)(?:```|$)", text, flags=re.S):
        lines = body.strip("\n").splitlines()
        while lines and lines[0].strip().lower() in ("python", "py", "python3"):
            lines = lines[1:]
        body = "\n".join(lines).strip()
        if "def transform" in body:
            cands.append(body)
    if cands:
        return cands[-1]
    if "def transform" in text:
        rest = text[text.index("def transform"):]
        return rest.split("```")[0].strip()
    return None


def format_problem(code):
    """Problems that mean the reply was not usable code at all (asked again once, not counted as a repair)."""
    if code is None:
        return "the reply has no ```python block that defines def transform(df)"
    probs = [p for p in safety_check(code) if p.startswith("SyntaxError") or "top-level function" in p]
    return "; ".join(probs) or None


def safety_check(code):
    """Static AST check. Returns a list of problems (empty = safe to execute)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of {a.name!r} is not allowed (only {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"dunder attribute access {node.attr!r} is not allowed")
            elif node.attr in FORBIDDEN_ATTRS:
                problems.append(f"file/system access via .{node.attr} is not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "transform" for n in tree.body):
        problems.append("the code must define a top-level function transform(df)")
    return sorted(set(problems))


_RUNNER = r'''
import sys, pickle, traceback, json, resource
import numpy as np, pandas as pd
try:
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3, 8 * 1024 ** 3))
except Exception:
    pass
work = sys.argv[1]
df = pd.read_pickle(work + "/input.pkl")
code = open(work + "/code.py").read()
ns = {"pd": pd, "np": np, "__name__": "candidate"}
try:
    exec(compile(code, "<transform>", "exec"), ns)
    out = ns["transform"](df.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"transform(df) must return a pandas DataFrame, got {type(out).__name__}")
    out.to_pickle(work + "/output.pkl")
    json.dump({"ok": True}, open(work + "/status.json", "w"))
except Exception as e:
    tb = traceback.format_exc().strip().splitlines()
    json.dump({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": "\n".join(tb[-8:])}, open(work + "/status.json", "w"))
'''
_RUNNER_PATH = WORK_DIR / "_sandbox_runner.py"
_RUNNER_PATH.write_text(_RUNNER)


def run_in_sandbox(code, before, timeout=EXEC_TIMEOUT_S):
    """Execute transform(df) in a separate Python process (own memory limit + timeout).
    Returns {"ok", "output" (DataFrame|None), "error", "traceback", "seconds"}."""
    work = Path(WORK_DIR) / "_sandbox" / hashlib.sha1(f"{time.time_ns()}{os.getpid()}".encode()).hexdigest()[:12]
    work.mkdir(parents=True, exist_ok=True)
    before.to_pickle(work / "input.pkl")
    (work / "code.py").write_text(code)
    t0 = time.time()
    try:
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run([sys.executable, str(_RUNNER_PATH), str(work)], timeout=timeout, capture_output=True, cwd=str(work), env=env)
        st = json.loads((work / "status.json").read_text()) if (work / "status.json").exists() else {"ok": False, "error": "sandbox crashed (no status)", "traceback": ""}
    except subprocess.TimeoutExpired:
        st = {"ok": False, "error": f"TimeoutError: execution exceeded {timeout}s", "traceback": ""}
    out = pd.read_pickle(work / "output.pkl") if st.get("ok") else None
    shutil.rmtree(work, ignore_errors=True)
    return {"ok": bool(st.get("ok")), "output": out, "error": st.get("error"), "traceback": st.get("traceback", ""),
            "seconds": round(time.time() - t0, 3)}


def execute_and_verify(code, before, contract):
    """Safety check -> sandbox -> verify. Returns a stage record (without the DataFrame) and the output."""
    rec = {"code": code}
    if code is None:
        rec.update(status="no_code", feedback="Your reply did not contain a ```python block defining transform(df).")
        return rec, None
    problems = safety_check(code)
    if problems:
        rec.update(status="unsafe", safety=problems, feedback="Safety checker rejected the code:\n- " + "\n- ".join(problems))
        return rec, None
    ex = run_in_sandbox(code, before)
    rec["exec_seconds"] = ex["seconds"]
    if not ex["ok"]:
        rec.update(status="exec_error", error=ex["error"], feedback=f"Execution failed: {ex['error']}\n{ex['traceback']}")
        return rec, None
    rep = verify(before, ex["output"], contract)
    rec.update(status=rep["verdict"], report=rep, feedback=render_report(rep))
    return rec, ex["output"]
