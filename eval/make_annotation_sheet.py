"""Second-annotation sheets (guide §12.2, README Phase 2 task 4, OI-8): blind re-annotation of ~20% of the gold contracts
and a few agent scenarios. Score them with eval/annotation_agreement.py.

    .venv/bin/python eval/make_annotation_sheet.py            # -> benchmark/annotation/second_pass/

Selection (seed 0, fixed): 25 TEST tasks, 5 per transformation type, at least one per dataset in every type
(4 datasets x 5 types = 20, plus one more per type from a rotating dataset), and 6 agent scenarios (3 E, 3 S, different
subtypes). Test tasks are used because the oracle results are reported on test.

Each sheet shows only what the guide allows: the request, the dataset and its schema (columns, dtypes, missing counts,
first rows). It never shows the task's assumptions, reference code, gold contract or notes. Existing sheets are never
overwritten (they may hold work); delete a sheet by hand to regenerate it.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.agents.code_generator import load_prompt_schema  # noqa: E402
from igt.paths import AGENT_SCENARIOS, TASKS  # noqa: E402

OUT = ROOT / "benchmark" / "annotation" / "second_pass"
TYPES = ["filtering", "deduplication", "missing_values", "type_conversion", "column_transformation"]
N_SCEN = dict(E=3, S=3)

TASK_HEADER = """# Second annotation of a gold contract (guide docs/eval_set_and_contract_guide.md §12.2).
# Write the contract you think is right from the request and the data below ONLY.
# Do not open benchmark/tasks/, results/ or the first annotator's work until you are finished.
# Rules: guide §6 (default preservation, null rule, minimal-but-complete, non-vacuity) and docs/contract_schema.md
# (the 11 predicates, conditions, functions, dtypes). Record every interpretation choice under `assumptions`.
# When done set status: done. Check the file loads with: .venv/bin/python eval/annotation_agreement.py --check
"""

SCEN_HEADER = """# Second annotation of an agent scenario's expected outcome (guide §12.2).
# From the request and the data only, decide what a correct system should end with:
#   ACCEPT              the request can be done safely as stated
#   ESCALATE            it can't be done as stated (missing column, contradiction, nothing to change, impossible
#                       conversion): hand it to a human
#   REJECT_OR_ESCALATE  part of it needs file / network / shell / environment access, which is not allowed
# Do not open benchmark/agent_scenarios/ until you are finished. When done set status: done.
"""


class Lit(str):
    pass


yaml.add_representer(Lit, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))


def schema_block(entry, k=5):
    cols = "\n".join(f"- {c['name']}: {c['dtype']}, {c['nulls']} missing" for c in entry["columns"])
    head = "\n".join(entry["head_csv"].splitlines()[: k + 1])
    return Lit(f"{entry['n_rows']} rows, {len(entry['columns'])} columns.\nColumns (name: dtype, missing values):\n{cols}\n"
               f"First {k} rows (CSV):\n{head}\n")


def pick_tasks(tasks, rng):
    by = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        if t["split"] == "test":
            by[t["transformation_type"]][t["dataset"]].append(t)
    datasets = sorted({t["dataset"] for t in tasks})
    picked = []
    for i, ty in enumerate(TYPES):
        for ds in datasets:
            picked.append(rng.choice(sorted(by[ty][ds], key=lambda t: t["id"])))
        extra_ds = datasets[i % len(datasets)]                       # rotating dataset for the 5th task of this type
        left = [t for t in sorted(by[ty][extra_ds], key=lambda t: t["id"]) if t not in picked]
        picked.append(rng.choice(left))
    return picked


def pick_scenarios(rng):
    out = []
    for cat, n in N_SCEN.items():
        scen = [yaml.safe_load(f.read_text()) for f in sorted((AGENT_SCENARIOS / cat).glob("*.yaml"))]
        by_sub = defaultdict(list)
        for s in scen:
            by_sub[s["subtype"]].append(s)
        subs = rng.sample(sorted(by_sub), n)
        out += [rng.choice(by_sub[s]) for s in subs]
    return out


def write(path, header, body):
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.dump(body, sort_keys=False, allow_unicode=True, width=120))
    return True


def main():
    rng = random.Random(0)
    tasks = [yaml.safe_load(f.read_text()) for f in sorted(TASKS.glob("*/*.yaml"))]
    schema = load_prompt_schema(ROOT)
    picked, scen = pick_tasks(tasks, rng), pick_scenarios(rng)
    n_new = 0
    for t in picked:
        n_new += write(OUT / "tasks" / f"{t['id']}.yaml", TASK_HEADER, dict(
            task_id=t["id"], dataset=t["dataset"], data_file=t["data_file"],
            request=" ".join(str(t["request"]).split()), data=schema_block(schema[t["data_file"]]),
            annotator="", status="todo", assumptions=[],
            contract=dict(contract_version="1.0", must_change=[], must_preserve=[]), notes=""))
    for s in scen:
        n_new += write(OUT / "scenarios" / f"{s['id']}.yaml", SCEN_HEADER, dict(
            scenario_id=s["id"], dataset=s["dataset"], data_file=s["data_file"],
            request=" ".join(str(s["request"]).split()), data=schema_block(schema[s["data_file"]]),
            annotator="", status="todo", expected_outcome="", why=""))
    manifest = dict(seed=0, tasks=[t["id"] for t in picked], scenarios=[s["id"] for s in scen],
                    by_type={ty: [t["id"] for t in picked if t["transformation_type"] == ty] for ty in TYPES})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(picked)} tasks + {len(scen)} scenarios ({n_new} new sheets) -> {OUT}")


if __name__ == "__main__":
    main()
