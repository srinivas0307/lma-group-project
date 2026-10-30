"""AST allow-list checker for generated pandas code (README §4.5).

Static check only; it is the first layer. The executor (process isolation, CPU/memory/time limits,
network cut) is the second layer and must contain anything this check misses.

Policy:
- imports: only modules in ALLOWED_IMPORTS
- names: no builtins that reach files, the interpreter or other code (open, eval, exec, getattr, ...)
- attributes: no dunder attributes, no frame/code introspection, no pandas/numpy file or OS I/O
"""
import ast
import warnings
from dataclasses import dataclass

ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "re", "math", "datetime", "string", "collections", "itertools",
    "functools", "statistics", "typing",
})
# submodules of allowed packages that still reach files, the OS or native code
BLOCKED_MODULE_PREFIXES = ("pandas.io", "pandas.core.computation", "pandas.util", "pandas.testing",
                           "numpy.ctypeslib", "numpy.lib.format", "numpy.lib.npyio", "numpy.f2py",
                           "numpy.distutils", "numpy.testing")
ALLOWED_DUNDER_DEFS = frozenset({"__init__"})

BLOCKED_NAMES = frozenset({
    # builtins that run code, reach the interpreter, or bypass attribute checks
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
    "breakpoint", "exit", "quit", "getattr", "setattr", "delattr", "memoryview", "help",
    "__builtins__", "__loader__", "__spec__",
    # module names (in case they are reachable without an import)
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "io", "pickle", "requests",
    "urllib", "http", "builtins", "importlib", "ctypes", "multiprocessing", "threading",
    "signal", "glob", "tempfile", "resource", "gc", "inspect", "marshal", "code", "pty",
})

BLOCKED_ATTRS = frozenset({
    # pandas readers / writers
    "read_csv", "read_excel", "read_parquet", "read_json", "read_sql", "read_sql_query",
    "read_sql_table", "read_pickle", "read_table", "read_html", "read_feather", "read_hdf",
    "read_clipboard", "read_fwf", "read_xml", "read_orc", "read_sas", "read_spss", "read_stata",
    "to_csv", "to_parquet", "to_pickle", "to_excel", "to_json", "to_sql", "to_hdf", "to_feather",
    "to_html", "to_xml", "to_clipboard", "to_orc", "to_stata", "to_latex", "to_markdown",
    "HDFStore", "ExcelWriter", "ExcelFile",
    # numpy file I/O and raw memory
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "open_memmap", "ctypeslib", "ctypes",
    # submodules that reach I/O or the OS
    "io", "os", "sys", "subprocess", "system", "popen", "spawn", "fork",
    # string-evaluating pandas APIs (can evaluate expressions outside the AST check)
    "eval",
    # frame / code / generator introspection (escape routes that are not dunders)
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "gi_frame", "gi_code", "cr_frame",
    "ag_frame", "tb_frame", "tb_next", "co_code", "func_globals",
})


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""
    lineno: int | None = None


def _blocked(node, reason):
    return GuardResult(False, reason, getattr(node, "lineno", None))


def _module_allowed(name):
    return name.split(".")[0] in ALLOWED_IMPORTS and not any(
        name == p or name.startswith(p + ".") for p in BLOCKED_MODULE_PREFIXES)


def check(code: str) -> GuardResult:
    """Return GuardResult(ok=True) if the code passes the allow-list, else the first violation."""
    try:
        with warnings.catch_warnings():                 # model code may contain '\W' in non-raw strings
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError as e:
        return GuardResult(False, f"syntax error: {e.msg}", e.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not _module_allowed(a.name):
                    return _blocked(node, f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not _module_allowed(node.module or ""):
                return _blocked(node, f"from {'.' * node.level}{node.module or ''} import ...")
            for a in node.names:
                if (a.name in BLOCKED_ATTRS or a.name in BLOCKED_NAMES or a.name == "*"
                        or not _module_allowed(f"{node.module}.{a.name}")):
                    return _blocked(node, f"from {node.module} import {a.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name.endswith("__") and node.name not in ALLOWED_DUNDER_DEFS:
                return _blocked(node, f"defines dunder method '{node.name}'")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return _blocked(node, f"name '{node.id}'")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return _blocked(node, f"dunder attribute '{node.attr}'")
            if node.attr in BLOCKED_ATTRS:
                return _blocked(node, f"attribute '{node.attr}'")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return _blocked(node, "global/nonlocal statement")
    return GuardResult(True)
