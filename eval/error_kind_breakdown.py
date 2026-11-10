"""Stage 1 recall by error kind (README Phase 5 task 3, Phase 12 failure analysis). CPU only.

    .venv/bin/python eval/error_kind_breakdown.py        # -> results/stage1/summary/error_kinds.{json,md}

For every runnable wrong TEST candidate, its primary error kind comes from benchmark/annotation/error_kinds_pool_v4.csv
(eval/draft_error_kinds.py; drafts until reviewed). For each system the share of those candidates it does not accept
(REJECT or ESCALATE) is reported per kind: which kinds of mistake each verifier catches. Systems: every verdicts file
under results/stage1/ (Baseline 2, Baseline 3, the oracle and its direction ablations, the size ablation, and the
fine-tuned parsers once scored). The table states how many labels are reviewed.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.paths import STAGE1, STAGE1_SUMMARY  # noqa: E402

LABELS = ROOT / "benchmark" / "annotation" / "error_kinds_pool_v4.csv"
SYSTEMS = [("Baseline 2 (conventional)", STAGE1 / "baseline2" / "verdicts_baseline2.jsonl"),
           ("Baseline 3 (prompted 7B)", STAGE1 / "baseline3" / "verdicts_intent_prompted_7b_p1v3.jsonl"),
           ("prompted 3B", STAGE1 / "size_ablation" / "intent_prompted_3b_p1v2" / "verdicts_intent_prompted_3b_p1v2.jsonl"),
           ("prompted 1.5B", STAGE1 / "size_ablation" / "intent_prompted_1_5b_p1v3" / "verdicts_intent_prompted_1_5b_p1v3.jsonl"),
           ("fine-tuned 1.5B", STAGE1 / "finetuned" / "intent_ft_1_5b" / "verdicts_intent_ft_1_5b.jsonl"),
           ("fine-tuned 3B", STAGE1 / "finetuned" / "intent_ft_3b" / "verdicts_intent_ft_3b.jsonl"),
           ("7B + data tools", STAGE1 / "tool_parser" / "intent_tools_7b_t1v1" / "verdicts_intent_tools_7b_t1v1.jsonl"),
           ("Oracle (gold contract)", STAGE1 / "oracle" / "verdicts_intent_oracle.jsonl"),
           ("Oracle, must_change only", STAGE1 / "oracle" / "verdicts_intent_oracle_change_only.jsonl"),
           ("Oracle, must_preserve only", STAGE1 / "oracle" / "verdicts_intent_oracle_preserve_only.jsonl")]


def main(argv=None):
    with open(LABELS, newline="") as fh:
        labels = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    kinds = Counter(r["primary_kind"] for r in labels)
    order = [k for k, _ in kinds.most_common()]
    reviewed = sum(bool((r.get("reviewed_by") or "").strip()) for r in labels)
    report = dict(n=len(labels), reviewed=reviewed, kinds=dict(kinds), systems={})
    for name, path in SYSTEMS:
        if not path.exists():
            continue
        v = {}
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                v[r["candidate_id"]] = r["outcome"]
        caught = defaultdict(int)
        for r in labels:
            caught[r["primary_kind"]] += v.get(r["candidate_id"]) != "ACCEPT"
        report["systems"][name] = dict(overall=round(sum(caught.values()) / len(labels), 3),
                                       by_kind={k: dict(caught=caught[k], n=kinds[k]) for k in order})
    STAGE1_SUMMARY.mkdir(parents=True, exist_ok=True)
    (STAGE1_SUMMARY / "error_kinds.json").write_text(json.dumps(report, indent=2))
    names = list(report["systems"])
    md = ["# Stage 1: wrong candidates caught, by error kind (test)", "",
          f"{len(labels)} runnable wrong test candidates of pool v4. Kinds from `benchmark/annotation/error_kinds_pool_v4.csv` "
          f"(**{reviewed} of {len(labels)} labels reviewed by a person; the others are drafts**, see `docs/error_taxonomy.md`). "
          "Each cell: candidates of that kind the system did not accept / candidates of that kind.", "",
          "| Kind | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for k in order:
        cells = [f"{report['systems'][s]['by_kind'][k]['caught']}" for s in names]
        md.append(f"| {k} | {kinds[k]} | " + " | ".join(cells) + " |")
    md.append("| **all (recall)** | " + f"{len(labels)} | " + " | ".join(f"**{report['systems'][s]['overall']:.3f}**" for s in names) + " |")
    (STAGE1_SUMMARY / "error_kinds.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return report


if __name__ == "__main__":
    main()
