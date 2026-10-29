"""The fine-tuning data checker must keep good examples and drop each kind of bad one."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "finetune" / "data_gen" / "check_examples.py"
BUILD = ROOT / "finetune" / "data_gen" / "build_sft.py"


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    d = ROOT / "finetune" / "datasets" / "_pytest_toy"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.DataFrame({"species": rng.choice(["Adelie", "Gentoo"], 40), "mass_g": rng.integers(3000, 6000, 40),
                  "bill_mm": np.where(rng.random(40) < 0.2, np.nan, rng.normal(40, 5, 40).round(1))}).to_csv(d / "toy.csv", index=False)
    pd.DataFrame({"name": ["a", "b"], "v": [1, 2]}).to_csv(d / "toy2.csv", index=False)
    yield d
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


def ex(i, request, contract, code, data="finetune/datasets/_pytest_toy/toy.csv", ds="toy"):
    return dict(id=f"toy_{i}", source_dataset=ds, data_file=data, transformation_type="filtering",
                pattern="filter_numeric_threshold", style="plain", request=request, contract=contract, reference_code=code)


GOOD_C = {"must_change": [{"type": "rows_removed_where", "condition": {"column": "mass_g", "op": "<", "value": 4000}},
                          {"type": "row_count", "op": "decreased"}],
          "must_preserve": [{"type": "rows_retained_where", "condition": {"column": "mass_g", "op": ">=", "value": 4000}},
                            {"type": "columns_unchanged", "target": "*", "except": []}]}
GOOD_CODE = "def transform(df):\n    return df[df['mass_g'] >= 4000].copy()\n"


def test_checker_keeps_good_and_drops_bad(toy, tmp_path):
    rows = [
        ex(1, "Drop the penguins lighter than 4000 grams.", GOOD_C, GOOD_CODE),                                   # ok
        ex(2, "Get rid of birds under four kilos in body mass please.", GOOD_C,
           "def transform(df):\n    return df[df['mass_g'] >= 3500].copy()\n"),                                    # passes? no
        ex(3, "Keep heavy penguins only, cutoff 4000g.", {"must_change": [{"type": "nonsense"}]}, GOOD_CODE),      # schema
        ex(4, "Remove light penguins below 4000 g mass.", GOOD_C, "import os\ndef transform(df):\n    return df"),  # runs
        ex(5, "Remove the listings that are never available.", GOOD_C, GOOD_CODE),                                 # leakage
        ex(6, "Filter light ones out under the 4000 gram mark.", GOOD_C, GOOD_CODE, data="data/adult_dev.csv"),    # data
        ex(7, "Drop the penguins lighter than 4000 grams.", GOOD_C, GOOD_CODE),                                    # duplicate text
        ex(8, "Show me the penguins as they are.", {"must_preserve": [{"type": "row_count", "op": "unchanged"}],
                                                   "must_change": [{"type": "row_count", "op": "unchanged"}]},
           "def transform(df):\n    return df\n"),                                                                # nonvacuous
    ]
    src = tmp_path / "raw.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "checked.jsonl"
    p = subprocess.run([sys.executable, str(CHECK), str(src), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    rep = json.loads(out.with_suffix(".report.json").read_text())
    status = {d["id"]: d["status"] for d in rep["dropped"]}
    assert status == {"toy_2": "passes", "toy_3": "schema", "toy_4": "runs", "toy_5": "leakage", "toy_6": "data",
                      "toy_1": "unique", "toy_7": "unique", "toy_8": "nonvacuous"}, status


def test_build_sft_uses_eval_prompt_format(toy, tmp_path):
    rows = [ex(1, "Drop the penguins lighter than 4000 grams.", GOOD_C, GOOD_CODE),
            ex(2, "Remove all rows with a name of b.",
               {"must_change": [{"type": "rows_removed_where", "condition": {"column": "name", "op": "==", "value": "b"}}]},
               "def transform(df):\n    return df[df['name'] != 'b'].copy()\n", data="finetune/datasets/_pytest_toy/toy2.csv", ds="toy2")]
    src = tmp_path / "checked.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = subprocess.run([sys.executable, str(BUILD), str(src), "--val-datasets", "toy2", "--out-dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    tr = [json.loads(l) for l in (tmp_path / "sft_train.jsonl").read_text().splitlines()]
    va = [json.loads(l) for l in (tmp_path / "sft_val.jsonl").read_text().splitlines()]
    assert len(tr) == 1 and len(va) == 1
    sys.path.insert(0, str(ROOT / "src"))
    from igt.agents.intent_parser import SYSTEM_PROMPT, extract_contract
    m = tr[0]["messages"]
    assert [x["role"] for x in m] == ["system", "user", "assistant"] and m[0]["content"] == SYSTEM_PROMPT
    assert m[1]["content"].startswith("Dataset: 40 rows, 3 columns.") and m[1]["content"].endswith("4000 grams.")
    assert extract_contract(m[2]["content"])["status"] == "ok"
