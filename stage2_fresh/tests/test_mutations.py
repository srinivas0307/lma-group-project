"""Wrong outputs derived from the reference must be rejected by the gold contract."""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness
import numpy as np
ns = harness.load()
acc = collections.Counter(); tot = collections.Counter()
for t in ns["ALL_TASKS"]:
    b = ns["load_before"](t); ref = ns["reference_output"](t)
    c, _ = ns["normalize_contract"](t["gold_contract"])
    muts = {}
    if len(ref) > 1:
        muts["drop_extra_row"] = ref.drop(ref.index[len(ref) // 2])
    if len(ref) < len(b):
        muts["reset_index"] = ref.reset_index(drop=True)
    changed = set(ns["profile_change"](b, ref)["cells_changed_by_column"]) | set(ref.columns) - set(b.columns)
    untouched = [col for col in b.columns if col in ref.columns and col not in changed and ref[col].dtype.kind in "if"]
    if untouched:
        m = ref.copy(); col = untouched[0]; m.loc[m.index[0], col] = (m.loc[m.index[0], col] if m.loc[m.index[0], col] == m.loc[m.index[0], col] else 0) + 1
        muts["corrupt_untouched_cell"] = m
    m = ref.copy(); m["helper_tmp"] = 1; muts["extra_column"] = m
    for k, out in muts.items():
        tot[k] += 1
        if ns["verify"](b, out, c)["verdict"] == "PASS":
            acc[k] += 1; print("ACCEPTED", k, t["id"])
print({k: f"{acc[k]}/{tot[k]} wrongly accepted" for k in tot})
