# ===== Benchmark tasks =====
# Each task: request (what the system sees), data_file (the "before" data),
# reference_code and gold_contract (used ONLY by the evaluator, never by the system).

def load_tasks(data_root):
    tasks = []
    for f in sorted((Path(data_root) / "benchmark" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        t["request"] = " ".join(str(t["request"]).split())
        tasks.append(t)
    return tasks


_DF_CACHE = {}

def load_before(task):
    """Loading rule of the benchmark: pd.read_csv(path) with no options; index 0..n-1 is the row id."""
    path = Path(DATA_ROOT) / task["data_file"]
    if path not in _DF_CACHE:
        _DF_CACHE[path] = pd.read_csv(path)
    return _DF_CACHE[path].copy()


ALL_TASKS = load_tasks(DATA_ROOT)
print(len(ALL_TASKS), "tasks:", {s: sum(t["split"] == s for t in ALL_TASKS) for s in ("dev", "test")})
