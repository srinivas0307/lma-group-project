"""Stage 2 pieces: contract-aware generator prompt (b2-v1), end-to-end runner, metrics, and the fine-tuning
review tools (tally + fixes). Runs with the scripted pipeline-check backend (no GPU)."""
import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "data_gen"))

from igt.agents.code_generator import (CONTRACT_PROMPT_HASH, CONTRACT_SYSTEM_PROMPT, PROMPT_HASH, SYSTEM_PROMPT,  # noqa: E402
                                       build_generator_messages, build_messages, contract_for_prompt,
                                       load_prompt_schema)
from igt.contracts import load_contract  # noqa: E402

SCHEMA = load_prompt_schema(ROOT)
TASK = yaml.safe_load((ROOT / "benchmark/tasks/titanic/titanic_filt_003.yaml").read_text())
GT = dict(request=" ".join(TASK["request"].split()), data_file=TASK["data_file"])


# ---------------- generator prompt ----------------
def test_b1_unchanged_and_default():
    assert PROMPT_HASH == "ebbe8bceae71"
    assert build_generator_messages(GT, SCHEMA) == build_messages(GT, SCHEMA)
    assert build_generator_messages(GT, SCHEMA, None) == build_messages(GT, SCHEMA)


def test_b2_only_adds_contract():
    b1 = build_messages(GT, SCHEMA)
    b2 = build_generator_messages(GT, SCHEMA, TASK["gold_contract"])
    assert CONTRACT_PROMPT_HASH == "243a463f9777"           # pinned: changing the prompt needs a new version
    assert b2[0]["content"].startswith(SYSTEM_PROMPT) and b2[0]["content"] == CONTRACT_SYSTEM_PROMPT
    assert b2[1]["content"].startswith(b1[1]["content"])
    assert "Intent contract" in b2[1]["content"] and '"rows_removed_where"' in b2[1]["content"]


def test_contract_text_has_no_ids_and_round_trips():
    txt = contract_for_prompt(TASK["gold_contract"])
    assert '"id"' not in txt
    obj = json.loads(txt)
    load_contract(obj)                                       # still a valid v1.0 contract
    assert len(obj["must_change"]) == len(TASK["gold_contract"]["must_change"])


# ---------------- runner + metrics ----------------
def _run(tmp, *extra):
    from eval.runner_stage2 import main
    return main(["--backend", "pipeline_check", "--tasks", "titanic_filt_003", "titanic_dedup_001",
                 "--out", str(tmp), *extra])


def _runs(d):
    return [json.loads(l) for l in (d / "runs.jsonl").read_text().splitlines()]


def test_baseline1_accepts_noop(tmp_path):
    runs = _runs(_run(tmp_path, "--preset", "baseline1"))
    assert len(runs) == 2
    assert all(r["outcome"] == "ACCEPT" and r["final_correct"] is False and r["attempts"] == 0 for r in runs)


def test_gold_contract_catches_and_repairs(tmp_path):
    d = _run(tmp_path, "--preset", "gold_contract")
    runs = _runs(d)
    assert all(r["first_check"] == "VIOLATION" and r["outcome"] == "ACCEPT" and r["final_correct"] for r in runs)
    assert all(r["generator_prompt"] == "b2-v1" and r["no_repair_correct"] is False for r in runs)
    info = json.loads((d / "run_info.json").read_text())
    assert info["valid_result"] is False                     # pipeline checks are never results


def test_invalid_parsed_contract_escalates_before_generation(tmp_path):
    p = tmp_path / "parses.jsonl"
    p.write_text(json.dumps(dict(task_id="titanic_filt_003", raw_output="no json here")) + "\n")
    runs = _runs(_run(tmp_path, "--preset", "proposed", "--parses", str(p)))
    by = {r["task_id"]: r for r in runs}
    assert by["titanic_filt_003"]["stage_ended"] == "parse" and by["titanic_filt_003"]["outcome"] == "ESCALATE"
    assert by["titanic_dedup_001"]["contract_status"] == "missing"


def test_resume_refuses_other_config(tmp_path):
    _run(tmp_path, "--preset", "baseline1", "--system", "x")
    with pytest.raises(SystemExit):
        _run(tmp_path, "--preset", "gold_contract", "--system", "x")


def test_bad_combinations_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, "--preset", "baseline1", "--verifier", "intent")
    with pytest.raises(SystemExit):
        _run(tmp_path, "--preset", "proposed")              # parsed contract without --parses


def test_metrics(tmp_path):
    from eval.stage2_metrics import main
    _run(tmp_path, "--preset", "baseline1")
    _run(tmp_path, "--preset", "gold_contract")
    rep = main([str(tmp_path), "test"])
    g, b = rep["systems"]["gold_contract"], rep["systems"]["baseline1"]
    assert g["final_accuracy"] == 1.0 and g["no_repair_accuracy"] == 0.0 and g["repair_gain"] == 1.0
    assert b["final_accuracy"] == 0.0 and b["silent_error_rate"] == 1.0
    assert rep["paired_final_correct"]["baseline1 vs gold_contract"]["only_second"] == 2
    assert sorted(rep["not_valid_results"]) == ["baseline1", "gold_contract"]


def test_named_comparisons_match_by_preset(tmp_path):
    """Folder names with generator / parser tags (as the Kaggle notebook writes them) still get the plan's comparisons."""
    from eval.stage2_metrics import main
    _run(tmp_path, "--preset", "gold_contract", "--system", "gold_contract__gen_3b")
    p = tmp_path / "parses.jsonl"
    p.write_text("".join(json.dumps(dict(task_id=t["id"], raw_output="```json\n" + json.dumps(t["gold_contract"]) + "\n```"))
                         + "\n" for t in (TASK, yaml.safe_load((ROOT / "benchmark/tasks/titanic/titanic_dedup_001.yaml").read_text()))))
    _run(tmp_path, "--preset", "proposed", "--parses", str(p), "--system", "proposed__gen_3b__ft_x")
    rep = main([str(tmp_path), "test"])
    assert any(k.startswith("proposed__gen_3b__ft_x vs gold_contract__gen_3b") for k in rep["comparisons"])


# ---------------- fine-tuning review tools ----------------
def test_tally_review(tmp_path):
    import tally_review
    f = tmp_path / "r.csv"
    rows = [("a", "OK"), ("b", "ok"), ("c", "Wrong contract"), ("d", "unnatural-request"), ("e", "")]
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "type", "pattern", "style", "verdict", "note"])
        for i, v in rows:
            w.writerow([i, "filtering", "p1", "plain", v, ""])
    s = tally_review.main(["--csv", str(f), "--out", str(tmp_path / "s.json")])
    assert s["n_reviewed"] == 4 and s["not_ok"] == 2 and s["decision"] == "incomplete"
    assert s["verdicts"] == {"OK": 2, "wrong_contract": 1, "unnatural_request": 1}


def test_review_fixes():
    import fix_review_issues as fx
    e = dict(id="x", transformation_type="deduplication", pattern="dedup_on_key", request="Remove duplicates on a (one copy is enough).",
             contract={"must_change": [{"type": "duplicate_count", "target": ["a"], "op": "eq", "value": 0}]},
             reference_code='def transform(df):\n    return df.drop_duplicates(subset=["a"], keep="first").copy()\n')
    assert fx.fix_dedup_keep(e)["fix"] == "F1"
    assert e["contract"]["must_change"][0]["keep"] == "first" and e["request"].endswith("(keep the first one).")
    assert fx.fix_dedup_keep(e) is None                      # idempotent
    e2 = dict(e, request="Each combination of `O3` should appear only once (keep the last one).")
    assert fx.fix_single_key_combination(e2)["new"].startswith("Each value of `O3`")
    e3 = dict(pattern="filter_with_nulls", request="Discard orders whose sales is less than 29. Orders without a sales stay in.")
    assert fx.fix_filter_nulls_wording(e3)["new"] == "Discard orders with sales below 29. Keep orders with missing sales values."
