# ===== Run: smoke -> dev -> test (resumable; one JSON per task) =====
OUT_DIR = WORK_DIR / RUN_NAME
(OUT_DIR / "tasks").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "config.json").write_text(json.dumps({
    "system": SYSTEM_NAME, "parser_protocol": PARSER_PROTOCOL, "parser_base": PARSER_BASE, "parser_revision": PARSER_REVISION,
    "parser_adapter": str(PARSER_ADAPTER), "coder": CODER_BASE, "max_parser_feedback": MAX_PARSER_FEEDBACK,
    "max_repair_attempts": MAX_REPAIR_ATTEMPTS, "max_repair_tool_calls": MAX_REPAIR_TOOL_CALLS,
    "repetition_penalty": REPETITION_PENALTY, "torch": torch.__version__, "transformers": transformers.__version__,
    "pandas": pd.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(N_GPU)]}, indent=1))


def run_split(tasks, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, t in enumerate(tasks, 1):
        f = out_dir / f"{t['id']}.json"
        if f.exists():
            records.append(json.loads(f.read_text()))
            continue
        try:
            r = run_task(t)
        except Exception as e:  # keep going; the failure is recorded as a withheld task
            r = {"id": t["id"], "split": t["split"], "transformation_type": t["transformation_type"], "request": t["request"],
                 "parser": {"status": "pipeline_error", "feedback": [], "calls": []}, "stages": [], "decision": "WITHHOLD",
                 "withhold_reason": "pipeline_error", "error": traceback.format_exc(), "final_correct": False,
                 "outcome": "withheld", "tokens": {"parser": 0, "generator": 0, "repair": 0, "total": 0}, "seconds": 0}
            print(traceback.format_exc())
        f.write_text(json.dumps(r, default=str, indent=1))
        records.append(r)
        row = task_row(r)
        print(f"[{label} {i:3d}/{len(tasks)}] {r['id']:<20} parser={row['parser_status']}(fb={row['parser_feedback_rounds']},"
              f"tools={row['parser_tools']},facts={row['facts']}) initial={row['initial_verdict']} repairs={row['repair_attempts']} "
              f"-> {r['decision']:<8} {r['outcome']:<18} {row['tokens']:>6} tok {row['seconds']:>5}s", flush=True)
    return records


all_records = {}
if RUN_SMOKE:
    smoke = [t for t in ALL_TASKS if t["split"] == "dev"][:SMOKE_TASKS]
    recs = run_split(smoke, OUT_DIR / "smoke", "smoke")
    assert all(r["withhold_reason"] != "pipeline_error" for r in recs), "smoke run hit a pipeline error - see the traceback above"
    print("smoke OK")
for split in RUN_SPLITS:
    sel = [t for t in ALL_TASKS if t["split"] == split][:MAX_TASKS_PER_SPLIT]
    all_records[split] = run_split(sel, OUT_DIR / "tasks", split)

summary = {split: summarize(recs) for split, recs in all_records.items()}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
results = pd.DataFrame([task_row(r) for recs in all_records.values() for r in recs])
results.to_csv(OUT_DIR / "results.csv", index=False)
for split, s in summary.items():
    print(f"\n===== {SYSTEM_NAME} | {split}: {s['tasks']} tasks =====")
    print(f"correct accepted   : {s['correct_accepted']}")
    print(f"incorrect accepted : {s['incorrect_accepted']}   (silent errors)")
    print(f"withheld           : {s['withheld']}   (of which final candidate was actually correct: {s['withheld_but_final_candidate_correct']})")
    print(f"withhold reasons   : {s['withhold_reasons']}")
    print(f"parser             : {s['parser']}")
    print(f"initial PASS {s['initial_pass']} | entered repair {s['entered_repair']} | fixed by repair {s['fixed_by_repair']}")
    print(f"tokens             : {s['tokens']}")
