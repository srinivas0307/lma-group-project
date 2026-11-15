# ===== CPU self-test of the verifier on this machine's pandas (no models involved) =====
# Every gold contract must PASS on its reference output and must NOT pass on the unchanged input.
_bad = []
for _t in ALL_TASKS:
    _b = load_before(_t)
    _c, _e = normalize_contract(_t["gold_contract"])
    if _e:
        _bad.append((_t["id"], "gold schema", _e)); continue
    if verify(_b, reference_output(_t), _c)["verdict"] != "PASS":
        _bad.append((_t["id"], "reference rejected", render_report(verify(_b, reference_output(_t), _c))))
    if verify(_b, _b.copy(), _c)["verdict"] == "PASS":
        _bad.append((_t["id"], "no-op accepted", ""))
print(f"verifier self-test: {len(ALL_TASKS) - len(_bad)}/{len(ALL_TASKS)} tasks OK")
for _x in _bad:
    print("  WARNING", _x[0], _x[1], str(_x[2])[:500])
