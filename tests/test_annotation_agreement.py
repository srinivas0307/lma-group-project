"""Second-annotation sheets and the agreement script (guide §12, OI-8)."""
import json
import shutil
from pathlib import Path

import yaml

from eval import annotation_agreement as aa

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "benchmark" / "annotation" / "second_pass"


def _fill(tmp, task_ids, modify=None, scenario_outcomes=None):
    d = tmp / "second_pass"
    shutil.copytree(SHEETS, d)
    for f in sorted((d / "tasks").glob("*.yaml")):
        s = yaml.safe_load(f.read_text())
        if s["task_id"] in task_ids:
            gold = yaml.safe_load((ROOT / "benchmark/tasks" / s["dataset"] / f"{s['task_id']}.yaml").read_text())
            c = json.loads(json.dumps(gold["gold_contract"]))
            if modify:
                modify(c)
            s.update(status="done", annotator="test", contract=c)
            f.write_text(yaml.dump(s, sort_keys=False))
    for f in sorted((d / "scenarios").glob("*.yaml")):
        s = yaml.safe_load(f.read_text())
        if scenario_outcomes and s["scenario_id"] in scenario_outcomes:
            s.update(status="done", annotator="test", expected_outcome=scenario_outcomes[s["scenario_id"]])
            f.write_text(yaml.dump(s, sort_keys=False))
    return d


def test_sheets_are_blind():
    manifest = json.loads((SHEETS / "manifest.json").read_text())
    assert len(manifest["tasks"]) == 25 and all(len(v) == 5 for v in manifest["by_type"].values())
    for f in (SHEETS / "tasks").glob("*.yaml"):
        text = f.read_text()
        s = yaml.safe_load(text)
        assert s["contract"]["must_change"] == [] and s["assumptions"] == []
        assert "reference_code" not in s and "gold_contract" not in s and "def transform" not in text


def test_identical_contracts_agree(tmp_path):
    d = _fill(tmp_path, {"titanic_filt_003", "adult_col_003"}, scenario_outcomes={"E_002": "ESCALATE"})
    r = aa.main(["--sheets", str(d), "--out", str(tmp_path)])
    assert r["n_tasks"] == 2 and r["exact_match"] == 1.0
    assert r["contracts"]["must_change"]["f1"] == 1.0 and r["contracts"]["must_preserve"]["f1"] == 1.0
    assert r["accepts_reference"] == 2 and r["rejects_noop"] == 2
    assert r["candidate_decisions"]["agreement"] == 1.0 and r["scenarios"]["agreement"] == 1.0
    assert (tmp_path / "agreement.md").exists()


def test_disagreement_is_measured(tmp_path):
    def drop_retained(c):                       # second annotator forgets "rows with age >= 18 or unknown are kept"
        c["must_preserve"] = [p for p in c["must_preserve"] if p["type"] != "rows_retained_where"]
    d = _fill(tmp_path, {"titanic_filt_003"}, modify=drop_retained, scenario_outcomes={"E_002": "ACCEPT"})
    r = aa.main(["--sheets", str(d), "--out", str(tmp_path)])
    assert r["exact_match"] == 0.0 and r["contracts"]["must_preserve"]["recall"] < 1.0
    assert r["contracts"]["must_change"]["f1"] == 1.0
    assert r["scenarios"]["agreement"] == 0.0
    assert "titanic_filt_003" in (tmp_path / "agreement.md").read_text()


def test_check_reports_invalid_contract(tmp_path):
    d = _fill(tmp_path, {"titanic_filt_003"}, modify=lambda c: c["must_change"].append({"type": "no_such_predicate"}))
    r = aa.main(["--sheets", str(d), "--check"])
    assert r["ready_tasks"] == 0 and any("invalid contract" in p for ps in r["problems"].values() for p in ps)
