import sys, json, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import harness
ns = harness.load()
tasks = ns["ALL_TASKS"]
bad = []
t0 = time.time()
for t in tasks:
    before = ns["load_before"](t)
    c, errs = ns["normalize_contract"](t["gold_contract"])
    if errs: bad.append((t["id"], "schema", errs)); continue
    if ns["safety_check"](t["reference_code"]): bad.append((t["id"], "safety", ns["safety_check"](t["reference_code"])))
    ref = ns["reference_output"](t)
    rep = ns["verify"](before, ref, c)
    if rep["verdict"] != "PASS": bad.append((t["id"], "ref-not-pass", ns["render_report"](rep)))
    noop = ns["verify"](before, before.copy(), c)
    if noop["verdict"] == "PASS": bad.append((t["id"], "noop-pass", ""))
    ok, why = ns["outputs_match"](ref, ref)
    if not ok: bad.append((t["id"], "self-match", why))
    sc = ns["contract_scores"](c, c, list(before.columns))
    if not sc["exact_match"]: bad.append((t["id"], "self-f1", sc))
    if not ns["outputs_match"](before, ref)[0] is False: bad.append((t["id"], "noop-matches-ref", ""))
print(f"{len(tasks)} tasks checked in {time.time()-t0:.0f}s; problems: {len(bad)}")
for b in bad: print(b[0], b[1]); print("   ", str(b[2])[:1500])
