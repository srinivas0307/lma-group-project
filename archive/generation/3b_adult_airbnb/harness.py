
import contextlib, glob, io, json, os, pickle, resource, sys, time
import numpy as np
import pandas as pd
import yaml


def load_tasks(root):
    tasks = []
    for f in sorted(glob.glob(f"{root}/eval/tasks/*/*.yaml")):
        with open(f) as fh:
            tasks.append(yaml.safe_load(fh))
    return tasks


def load_data(root, data_file):
    return pd.read_csv(f"{root}/{data_file}")


def family(s):
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_object_dtype(dt):
        return "string" if all(isinstance(v, str) for v in s.dropna()) else "object"
    if pd.api.types.is_string_dtype(dt):
        return "string"
    return str(dt)


def norm(v):
    if v is None:
        return ("N",)
    try:
        if pd.isna(v):
            return ("N",)
    except (TypeError, ValueError):
        pass
    if isinstance(v, (bool, np.bool_)):
        return ("B", bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return ("X", round(float(v), 9))
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return ("T", pd.Timestamp(v).isoformat())
    return ("S", str(v))


def outputs_equal(ref, cand):
    """Answer key: content comparison. Row order, index and column order are ignored."""
    if not isinstance(cand, pd.DataFrame):
        return False, "not a DataFrame"
    if cand.columns.duplicated().any():
        return False, "duplicate column names"
    if set(ref.columns) != set(cand.columns):
        extra = sorted(map(str, set(cand.columns) - set(ref.columns)))
        missing = sorted(map(str, set(ref.columns) - set(cand.columns)))
        return False, f"columns differ (extra={extra}, missing={missing})"
    if len(ref) != len(cand):
        return False, f"row count differs ({len(cand)} vs {len(ref)})"
    cols = list(ref.columns)
    for c in cols:
        fr, fc = family(ref[c]), family(cand[c])
        if fr != fc:
            return False, f"dtype family differs in {c} ({fc} vs {fr})"
        if fr == "category" and ref[c].cat.ordered:
            if not cand[c].cat.ordered or list(ref[c].cat.categories) != list(cand[c].cat.categories):
                return False, f"category order differs in {c}"

    def rows(df):
        return sorted(zip(*[[norm(v) for v in df[c].astype(object).tolist()] for c in cols]))

    if rows(ref) != rows(cand):
        return False, "values differ"
    return True, "equal"


def index_kept(before, out):
    """Informational: is the output index still usable as the row id?"""
    try:
        if not out.index.is_unique or not out.index.isin(before.index).all():
            return False
        common = [c for c in before.columns if c in out.columns]
        if not common or len(out) == 0:
            return True
        bv = before.loc[out.index, common].astype(object).values
        ov = out[common].astype(object).values
        agree = sum(2 * sum(norm(x) == norm(y) for x, y in zip(b, o)) >= len(common) for b, o in zip(bv, ov))
        return agree / len(out) >= 0.95
    except Exception:
        return False


def exec_transform(code, func_name, df):
    ns = {"pd": pd, "np": np, "__name__": "candidate"}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(code, "<candidate>", "exec"), ns)
        fn = ns.get(func_name)
        if not callable(fn):
            raise NameError(f"function '{func_name}' is not defined")
        return fn(df.copy())


def set_limits(cpu_s, mem_gb):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    lim = int(mem_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def cmd_schema(root, out_path):
    tasks = load_tasks(root)
    datasets = {}
    for t in tasks:
        f = t["data_file"]
        if f in datasets:
            continue
        df = load_data(root, f)
        datasets[f] = dict(
            n_rows=len(df),
            columns=[dict(name=str(c), dtype=str(df[c].dtype), nulls=int(df[c].isna().sum())) for c in df.columns],
            head_csv=df.head(5).to_csv(index=False),
        )
    slim = [dict(id=t["id"], split=t["split"], dataset=t["dataset"], transformation_type=t["transformation_type"],
                 difficulty=t.get("difficulty"), request=" ".join(str(t["request"]).split()), data_file=t["data_file"])
            for t in tasks]
    with open(out_path, "w") as fh:
        json.dump(dict(tasks=slim, datasets=datasets), fh)


def cmd_refs(root, ref_dir, out_path):
    os.makedirs(ref_dir, exist_ok=True)
    report = []
    for t in load_tasks(root):
        df = load_data(root, t["data_file"])
        rec = dict(id=t["id"])
        try:
            out = exec_transform(t["reference_code"], "transform", df)
            with open(f"{ref_dir}/{t['id']}.pkl", "wb") as fh:
                pickle.dump(out, fh)
            rec.update(ok=True, rows_after=len(out), expected_rows_after=t["expected_effect"]["rows_after"],
                       self_equal=outputs_equal(out, out)[0], noop_equal=outputs_equal(out, df)[0])
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        report.append(rec)
    with open(out_path, "w") as fh:
        json.dump(report, fh)


def cmd_run(root, ref_dir, job_path, out_path, cpu_s, mem_gb):
    with open(job_path) as fh:
        job = json.load(fh)
    df = load_data(root, job["data_file"])
    with open(f"{ref_dir}/{job['task_id']}.pkl", "rb") as fh:
        ref = pickle.load(fh)
    res = dict(status="error", ran_ok=False, correct=False, rows_before=len(df))
    try:
        set_limits(int(cpu_s), float(mem_gb))
    except Exception:
        pass
    t0 = time.time()
    try:
        out = exec_transform(job["code"], job["func_name"], df)
        res["runtime_s"] = round(time.time() - t0, 3)
        if not isinstance(out, pd.DataFrame):
            res.update(status="bad_return", error=f"returned {type(out).__name__}, not a DataFrame")
        else:
            ok, reason = outputs_equal(ref, out)
            res.update(status="ok", ran_ok=True, correct=bool(ok), reason=reason, rows_after=len(out),
                       cols_added=sorted(map(str, set(out.columns) - set(df.columns))),
                       cols_removed=sorted(map(str, set(df.columns) - set(out.columns))),
                       index_kept=bool(index_kept(df, out)))
    except MemoryError:
        res.update(status="memory", error="MemoryError")
    except BaseException as e:
        res.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    with open(out_path, "w") as fh:
        json.dump(res, fh, default=str)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"schema": cmd_schema, "refs": cmd_refs, "run": cmd_run}[cmd](*args)
