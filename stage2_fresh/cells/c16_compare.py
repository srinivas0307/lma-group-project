import json, zipfile
from pathlib import Path
import pandas as pd

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 250)
SEARCH = [Path("/kaggle/input"), Path("."), Path("/kaggle/working")]
RUNS = {"selective_facts": "stage2_fresh_selective_facts", "tool_use": "stage2_fresh_tool_use"}


def load_run(run_name):
    """Task records from an extracted folder or a ZIP named <run_name>.zip anywhere under SEARCH."""
    for root in SEARCH:
        if not root.exists():
            continue
        for d in root.rglob(run_name):
            if d.is_dir() and (d / "tasks").exists():
                return {f.stem: json.loads(f.read_text()) for f in (d / "tasks").glob("*.json")}
        for z in root.rglob(f"{run_name}*.zip"):
            with zipfile.ZipFile(z) as zf:
                return {Path(n).stem: json.loads(zf.read(n)) for n in zf.namelist()
                        if n.startswith(f"{run_name}/tasks/") and n.endswith(".json")}
    raise FileNotFoundError(run_name)


recs = {k: load_run(v) for k, v in RUNS.items()}
ids = sorted(set(recs["selective_facts"]) & set(recs["tool_use"]))
print({k: len(v) for k, v in recs.items()}, "common tasks:", len(ids))


def row(r):
    p = r["parser"]
    return {"split": r["split"], "type": r["transformation_type"], "parser": p["status"], "fb": len(p.get("feedback", [])),
            "tools": p.get("tool_attempts", 0), "facts": len(p.get("facts") or []),
            "exact": (r.get("contract_scores") or {}).get("exact_match"),
            "mc_f1": (r.get("contract_scores") or {}).get("must_change", {}).get("f1"),
            "mp_f1": (r.get("contract_scores") or {}).get("must_preserve", {}).get("f1"),
            "initial": r["stages"][0]["status"] if r.get("stages") else None, "repairs": max(0, len(r.get("stages", [])) - 1),
            "outcome": r["outcome"], "reason": r.get("withhold_reason"), "tokens": r["tokens"]["total"]}


frames = {k: pd.DataFrame({i: row(recs[k][i]) for i in ids}).T for k in RUNS}
both = pd.concat(frames, axis=1)

# ---- headline table per split
def headline(df):
    return pd.Series({
        "tasks": len(df), "correct accepted": (df.outcome == "correct_accepted").sum(),
        "incorrect accepted": (df.outcome == "incorrect_accepted").sum(), "withheld": (df.outcome == "withheld").sum(),
        "parser ok": (df.parser == "ok").sum(), "parser used feedback": (df.fb > 0).sum(),
        "recovered by feedback": ((df.fb > 0) & (df.parser == "ok")).sum(),
        "contract exact match": (df.exact == True).sum(), "must_change F1 (mean)": round(pd.to_numeric(df.mc_f1).mean(), 3),
        "must_preserve F1 (mean)": round(pd.to_numeric(df.mp_f1).mean(), 3),
        "initial PASS": (df.initial == "PASS").sum(), "entered repair": (df.repairs > 0).sum(),
        "parser tool attempts": df.tools.sum(), "tasks with facts": (df.facts > 0).sum(), "total tokens": df.tokens.sum()})

for split in ("dev", "test"):
    print(f"\n=========== {split} ===========")
    display(pd.DataFrame({k: headline(f[f.split == split]) for k, f in frames.items()}))

# ---- paired outcome table and the tasks that differ
test_ids = [i for i in ids if recs["selective_facts"][i]["split"] == "test"]
print("\nPaired outcomes on test (rows: selective facts, columns: tool use)")
display(pd.crosstab(frames["selective_facts"].loc[test_ids, "outcome"], frames["tool_use"].loc[test_ids, "outcome"]))

diff = both[both[("selective_facts", "outcome")] != both[("tool_use", "outcome")]]
print(f"\n{len(diff)} tasks with different outcomes:")
cols = [(k, c) for k in RUNS for c in ("parser", "fb", "tools", "facts", "exact", "initial", "repairs", "outcome", "reason")]
display(diff[cols])

# ---- per transformation type (test)
print("\nCorrect accepted by transformation type (test)")
display(pd.DataFrame({k: f.loc[test_ids].groupby("type").outcome.apply(lambda s: (s == "correct_accepted").sum())
                      for k, f in frames.items()}))


def show(task_id):
    """Print both parsers' contracts / feedback / outcome for one task."""
    for k in RUNS:
        r = recs[k][task_id]
        p = r["parser"]
        print(f"===== {k}: {task_id} — {r['request']}")
        print("parser:", p["status"], "| facts:", p.get("facts"), "| tools:", [(c['tool'], c['args'], c['observation'].get('ok')) for c in p.get("tool_calls", [])])
        for fb in p.get("feedback", []):
            print("  feedback round", fb["round"], ":", fb["problems"])
        print("contract:", json.dumps(p.get("contract")))
        print("stages:", [(s["status"], s.get("attempt")) for s in r.get("stages", [])], "->", r["outcome"], r.get("withhold_reason"))
        print()

for i in list(diff.index)[:5]:
    show(i)
