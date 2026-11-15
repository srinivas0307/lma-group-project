# ===== Stage 2 proposed pipeline: parse -> plan+generate -> safety -> sandbox -> verify -> accept | repair | flag =====

def _strip(stage):
    """Stage record without bulky fields (DataFrames are never stored)."""
    s = {k: v for k, v in stage.items() if k != "report"}
    if "report" in stage:
        r = stage["report"]
        s["report"] = {"verdict": r["verdict"], "structural": r.get("structural", []), "profile": r.get("profile"),
                       "failed": [{k: x[k] for k in ("section", "index", "type", "status", "detail", "examples")}
                                  for x in r.get("predicates", []) if x["status"] != "PASS"]}
    return s


def run_task(task):
    t0 = time.time()
    before = load_before(task)
    rec = {k: task[k] for k in ("id", "split", "dataset", "transformation_type", "difficulty", "request")}
    rec["system"] = SYSTEM_NAME
    # 1) intent parser (variant-specific)
    parser = run_parser(task, before)
    rec["parser"] = parser
    rec["tokens"] = {"parser": tokens_of(parser["calls"]), "generator": 0, "repair": 0}
    contract = parser.get("contract") if parser["status"] == "ok" else None
    final_output, rec["stages"] = None, []
    if contract is None:
        rec["decision"], rec["withhold_reason"] = "WITHHOLD", parser["status"]
    else:
        # 2) planner + generator, 3) safety, 4) sandbox, 5) profile + verify
        gen = generate_code(task, before, contract)
        rec["tokens"]["generator"] = tokens_of(gen["calls"])
        rec["generation"] = {"messages": gen["messages"], "calls": gen["calls"]}
        stage, out = execute_and_verify(gen["code"], before, contract)
        stage["attempt"] = 0
        rec["stages"].append(_strip(stage))
        initial_output, final_output = out, out
        # 6) repair loop on VIOLATION / crash / unsafe / no code; UNCERTAIN is flagged, not repaired
        if stage["status"] not in ("PASS", "UNCERTAIN") and MAX_REPAIR_ATTEMPTS > 0:
            attempts, calls, msgs, out = repair(task, before, contract, gen["code"], stage, out)
            rec["stages"] += [_strip(a) for a in attempts]
            rec["tokens"]["repair"] = tokens_of(calls)
            rec["repair"] = {"messages": msgs, "calls": calls}
            final_output = out
        last = rec["stages"][-1]["status"]
        rec["decision"] = "ACCEPT" if last == "PASS" else "WITHHOLD"
        rec["withhold_reason"] = None if last == "PASS" else ("verify_uncertain" if last == "UNCERTAIN" else
                                                              ("repair_exhausted" if len(rec["stages"]) > 1 else f"initial_{last}"))
        # evaluator only: was the initial / final candidate actually correct?
        ref = reference_output(task)
        rec["initial_correct"], rec["initial_match_detail"] = outputs_match(initial_output, ref) if initial_output is not None else (False, "no output")
        rec["contract_scores"] = contract_scores(contract, normalize_contract(task["gold_contract"])[0], list(before.columns))
    ref = reference_output(task)
    rec["final_correct"], rec["final_match_detail"] = outputs_match(final_output, ref) if final_output is not None else (False, "no output")
    if rec["decision"] == "ACCEPT":
        rec["outcome"] = "correct_accepted" if rec["final_correct"] else "incorrect_accepted"
    else:
        rec["outcome"] = "withheld"
    rec["tokens"]["total"] = sum(rec["tokens"].values())
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def summarize(records):
    n = len(records)
    if not n:
        return {}
    cnt = lambda f: sum(1 for r in records if f(r))
    with_contract = [r for r in records if r.get("contract_scores")]
    mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    s = {
        "tasks": n,
        "correct_accepted": cnt(lambda r: r["outcome"] == "correct_accepted"),
        "incorrect_accepted": cnt(lambda r: r["outcome"] == "incorrect_accepted"),
        "withheld": cnt(lambda r: r["outcome"] == "withheld"),
        "withheld_but_final_candidate_correct": cnt(lambda r: r["outcome"] == "withheld" and r["final_correct"]),
        "withhold_reasons": dict(pd.Series([r["withhold_reason"] for r in records if r["withhold_reason"]]).value_counts().to_dict()),
        "parser": {
            "ok": cnt(lambda r: r["parser"]["status"] == "ok"),
            "used_feedback": cnt(lambda r: len(r["parser"]["feedback"]) > 0),
            "recovered_by_feedback": cnt(lambda r: r["parser"]["status"] == "ok" and len(r["parser"]["feedback"]) > 0),
            "feedback_rounds_total": sum(len(r["parser"]["feedback"]) for r in records),
            "tool_attempts": sum(r["parser"].get("tool_attempts", 0) for r in records),
            "tool_failures": sum(r["parser"].get("tool_failures", 0) for r in records),
            "tasks_with_facts": cnt(lambda r: bool(r["parser"].get("facts"))),
            "contract_exact_match": cnt(lambda r: r.get("contract_scores", {}).get("exact_match")),
            "must_change_f1_mean": mean([r["contract_scores"]["must_change"]["f1"] for r in with_contract]),
            "must_preserve_f1_mean": mean([r["contract_scores"]["must_preserve"]["f1"] for r in with_contract]),
        },
        "initial_pass": cnt(lambda r: r["stages"] and r["stages"][0]["status"] == "PASS"),
        "initial_correct": cnt(lambda r: r.get("initial_correct")),
        "entered_repair": cnt(lambda r: len(r["stages"]) > 1),
        "fixed_by_repair": cnt(lambda r: len(r["stages"]) > 1 and r["decision"] == "ACCEPT"),
        "repair_attempts_total": sum(max(0, len(r["stages"]) - 1) for r in records),
        "tokens": {k: int(sum(r["tokens"][k] for r in records)) for k in ("parser", "generator", "repair", "total")},
        "seconds": round(sum(r["seconds"] for r in records), 1),
    }
    s["silent_error_rate_among_accepted"] = round(s["incorrect_accepted"] / max(1, s["correct_accepted"] + s["incorrect_accepted"]), 4)
    return s


def task_row(r):
    p = r["parser"]
    return {"id": r["id"], "split": r["split"], "type": r["transformation_type"], "parser_status": p["status"],
            "parser_feedback_rounds": len(p["feedback"]), "parser_tools": p.get("tool_attempts", 0),
            "facts": len(p.get("facts") or []),
            "contract_exact": (r.get("contract_scores") or {}).get("exact_match"),
            "initial_verdict": r["stages"][0]["status"] if r["stages"] else None,
            "repair_attempts": max(0, len(r["stages"]) - 1), "decision": r["decision"],
            "withhold_reason": r["withhold_reason"], "final_correct": r["final_correct"], "outcome": r["outcome"],
            "tokens": r["tokens"]["total"], "seconds": r["seconds"]}
