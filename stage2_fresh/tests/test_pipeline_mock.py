"""End-to-end pipeline test with scripted fake LLMs (no GPU). Exercises parser feedback, parser tools,
generator, safety, sandbox, verifier, repair tools, evaluator and metrics for both parser variants."""
import sys, json, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness

CFG = dict(MAX_PARSER_FEEDBACK=2, MAX_REPAIR_ATTEMPTS=3, MAX_REPAIR_TOOL_CALLS=3, PARSER_MAX_NEW_TOKENS=1024,
           GEN_MAX_NEW_TOKENS=1024, REPAIR_MAX_NEW_TOKENS=1024, SYSTEM_NAME="mock")

class FakeLM:
    def __init__(self, script): self.script, self.seen = script, []
    def chat(self, messages, max_new_tokens=1024, tools=None):
        self.seen.append(copy.deepcopy(messages))
        reply = self.script(messages, tools)
        return reply, {"prompt_tokens": 100, "completion_tokens": 50, "truncated": False}

def load_variant(kind):
    ns = harness.load()
    ns.update(CFG)
    src = (harness.HERE / "cells" / f"c09_parser_{kind}.py").read_text()
    prompt = (harness.HERE / "prompts" / f"system_{kind}.txt").read_text()
    exec(src.replace("__SYSTEM_PROMPT__", repr(prompt)), ns)
    for c in ("c10_generator.py", "c11_repair.py", "c12_pipeline.py"):
        exec((harness.HERE / "cells" / c).read_text(), ns)
    return ns

def fence(obj): return "```json\n" + json.dumps(obj) + "\n```"

def make_coder(task, broken_code):
    ref = task["reference_code"]
    def script(messages, tools):
        if tools is None:                       # generator
            return "# plan: do it\n```python\n" + broken_code + "\n```"
        n_assist = sum(m["role"] == "assistant" for m in messages)
        if n_assist == 0:                        # repair: first ask a tool
            return '<tool_call>\n{"name": "column_summary", "arguments": {"column": "%s"}}\n</tool_call>' % task["_col"]
        return "The bug: wrong rows.\n```python\n" + ref + "\n```"
    return script

problems = []
def check(cond, msg):
    if not cond: problems.append(msg); print("FAIL:", msg)

# ---------- variant A: selective facts, with one broken contract first (feedback loop)
ns = load_variant("selective_facts")
tasks = {t["id"]: t for t in ns["ALL_TASKS"]}
t = tasks["titanic_miss_004"]; t["_col"] = "Age"
gold = {k: v for k, v in t["gold_contract"].items()}
bad = copy.deepcopy(gold); bad["must_change"][0]["fn"] = "median"          # schema error
def parser_script(messages, tools):
    return fence(bad) if len(messages) == 2 else fence(gold)
ns["PARSER_LM"] = FakeLM(parser_script)
ns["CODER_LM"] = FakeLM(make_coder(t, "def transform(df):\n    out = df.copy()\n    out['Age'] = out['Age'].fillna(0)\n    return out"))
r = ns["run_task"](t)
print("A:", r["parser"]["status"], r["parser"]["facts"], [s["status"] for s in r["stages"]], r["outcome"])
check(r["parser"]["status"] == "ok" and len(r["parser"]["feedback"]) == 1, "sf feedback loop")
check(r["parser"]["facts"][0]["value"] == 28.0 and r["parser"]["facts"][0]["scope"] == {"kind": "full_column"}, "sf facts")
check("Required input facts (sf1):" in ns["PARSER_LM"].seen[0][1]["content"], "facts in prompt")
check([s["status"] for s in r["stages"]] == ["VIOLATION", "PASS"], "repair path")
check(r["stages"][1]["tools"][0]["name"] == "column_summary", "repair tool used")
check(r["outcome"] == "correct_accepted" and r["initial_correct"] is False, "outcome")
check(r["contract_scores"]["exact_match"], "contract scores")

# vacuous contract -> rejected after feedback rounds
t2 = tasks["adult_filt_001"]
vac = {"contract_version": "1.0", "must_change": [{"type": "row_count", "target": None, "op": "unchanged"}],
       "must_preserve": [{"type": "columns_unchanged", "target": "*", "except": []}]}
ns["PARSER_LM"] = FakeLM(lambda m, tl: fence(vac))
r2 = ns["run_task"](t2)
print("A vacuous:", r2["parser"]["status"], r2["parser"]["feedback"][0]["problems"][0][:90], r2["outcome"])
check(r2["parser"]["status"] == "parser_rejected" and len(r2["parser"]["feedback"]) == 3, "vacuous rejected")

# wrong value spelling -> soft feedback -> corrected
t3 = tasks["adult_filt_006"]; t3["_col"] = "workclass"
g3 = t3["gold_contract"]; w3 = json.loads(json.dumps(g3).replace('"Private"', '"private"'))
ns["PARSER_LM"] = FakeLM(lambda m, tl: fence(w3) if len(m) == 2 else fence(g3))
ns["CODER_LM"] = FakeLM(make_coder(t3, t3["reference_code"]))
r3 = ns["run_task"](t3)
print("A value fb:", r3["parser"]["feedback"], r3["outcome"])
check(r3["outcome"] == "correct_accepted" and any("never occurs" in x for x in r3["parser"]["feedback"][0]["problems"]), "value feedback")

# facts for every statistic request
for tid in ("titanic_miss_005", "airbnb_miss_004", "adult_miss_004", "titanic_miss_002", "taxi_miss_001", "adult_filt_001"):
    try:
        f = ns["compute_facts"](tasks[tid]["request"], ns["load_before"](tasks[tid]))
    except ns["FactUnavailable"] as e:
        f = f"UNAVAILABLE: {e}"
    print("facts", tid, json.dumps(f)[:300])
f5 = ns["compute_facts"](tasks["titanic_miss_005"]["request"], ns["load_before"](tasks["titanic_miss_005"]))
check([x["value"] for x in f5] == [37.0, 29.0, 24.0], "group facts")
check(ns["compute_facts"](tasks["airbnb_miss_004"]["request"], ns["load_before"](tasks["airbnb_miss_004"]))[0]["value"] == 0.59, "airbnb median")

# ---------- variant B: tool use (tool call, bad tool, then contract with an error, then fixed)
ns = load_variant("tool_use")
tasks = {t["id"]: t for t in ns["ALL_TASKS"]}
t = tasks["titanic_miss_004"]; t["_col"] = "Age"
gold = t["gold_contract"]
bad = copy.deepcopy(gold); bad["must_change"][0]["args"] = [{"lit": 28.0, "col": "Age"}]
seq = ['{"tool": "profile_column", "args": {"column": "__unknown_column__"}}',
       '{"tool": "profile_column", "args": {"column": "Age"}}', fence(bad), fence(gold)]
ns["PARSER_LM"] = FakeLM(lambda m, tl: seq[sum(x["role"] == "assistant" for x in m)])
ns["CODER_LM"] = FakeLM(make_coder(t, "import os\ndef transform(df):\n    return df"))
r = ns["run_task"](t)
p = r["parser"]
print("B:", p["status"], p["tool_attempts"], p["tool_failures"], [s["status"] for s in r["stages"]], r["outcome"])
obs = json.loads(p["messages"][5]["content"].split("\n")[1])
print("   profile obs median:", obs["result"]["statistics"]["median"], "| correction msg:", p["messages"][7]["content"][:160])
check(p["status"] == "ok" and p["tool_attempts"] == 2 and p["tool_failures"] == 1, "tool loop")
check(obs["result"]["statistics"]["median"] == 28.0, "tool observation")
check(p["messages"][7]["content"].startswith("Invalid response. Send one tool-call JSON object"), "trained correction message")
check([s["status"] for s in r["stages"]] == ["unsafe", "PASS"], "unsafe -> repaired")
check(r["outcome"] == "correct_accepted", "B outcome")

# escalation + incorrect accepted path (contract too weak -> wrong code passes)
t4 = tasks["adult_dedup_002"]
ns["PARSER_LM"] = FakeLM(lambda m, tl: '{"action": "escalate", "reason": "ambiguous"}')
r4 = ns["run_task"](t4)
check(r4["outcome"] == "withheld" and r4["withhold_reason"] == "escalated", "escalation")

recs = [r, r4]
s = ns["summarize"](recs); print("summary:", json.dumps(s)[:400])
check(s["correct_accepted"] == 1 and s["withheld"] == 1, "summary")
other = ns["DataTools"](ns["load_before"](tasks["titanic_miss_005"]))
print("group_profile:", json.dumps(other.group_profile("Age", ["Pclass"]))[:250])
print("count_rows:", other.count_rows({"column": "Age", "op": "is_null"}))
print("inspect_rows:", other.inspect_rows(["Age", "Pclass"], 2, {"column": "Age", "op": "is_null"}))
print("\nPROBLEMS:", len(problems), problems)
