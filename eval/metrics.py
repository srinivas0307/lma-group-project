"""Stage 1 metrics for any verdicts file over the frozen pool (README §9.4, G7, G8).

Definitions (one row = one unique candidate in the pool):
    runnable          candidate ran and returned a DataFrame
    recall            runnable wrong candidates not accepted / runnable wrong candidates
                      (crashes are excluded: every system rejects them, so they can't separate systems)
    recall_all        all wrong candidates not accepted / all wrong candidates (crashes included)
    false_rejection   correct candidates not accepted / correct candidates
    silent_error_rate wrong and accepted / accepted (G8)
    escalate_rate     ESCALATE outcomes / candidates (counted as "not accepted" everywhere else, G7)
Confidence intervals: Wilson (candidates treated as independent) and a task-level percentile
bootstrap (resamples tasks, which is the honest one because candidates of a task are correlated).
Paired comparison of two systems: exact McNemar test on per-candidate "decision was right".

Usage:
    python eval/metrics.py --verdicts results/stage1/baseline1/verdicts_baseline1.jsonl results/stage1/baseline2/verdicts_baseline2.jsonl
"""
import argparse, json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.types import validate_record  # noqa: E402

TYPES = ["column_transformation", "deduplication", "filtering", "missing_values", "type_conversion"]


def load_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - m) / d), 3), round(min(1.0, (c + m) / d), 3)]


def counts(rows):
    """rows: list of (candidate, outcome)."""
    c = dict(n=0, runnable=0, correct=0, wrong_runnable=0, wrong_all=0, accepted=0, escalated=0,
             wrong_runnable_rejected=0, wrong_all_rejected=0, correct_rejected=0, wrong_accepted=0)
    for cand, outcome in rows:
        acc = outcome == "ACCEPT"
        c["n"] += 1
        c["runnable"] += cand["ran_ok"]
        c["accepted"] += acc
        c["escalated"] += outcome == "ESCALATE"
        if cand["is_correct"]:
            c["correct"] += 1
            c["correct_rejected"] += not acc
        else:
            c["wrong_all"] += 1
            c["wrong_all_rejected"] += not acc
            c["wrong_accepted"] += acc
            if cand["ran_ok"]:
                c["wrong_runnable"] += 1
                c["wrong_runnable_rejected"] += not acc
    return c


RATES = {  # name -> (numerator, denominator)
    "recall": ("wrong_runnable_rejected", "wrong_runnable"),
    "recall_all": ("wrong_all_rejected", "wrong_all"),
    "false_rejection": ("correct_rejected", "correct"),
    "silent_error_rate": ("wrong_accepted", "accepted"),
    "escalate_rate": ("escalated", "n"),
}


def rates(c):
    return {k: (round(c[a] / c[b], 3) if c[b] else None) for k, (a, b) in RATES.items()}


def bootstrap(rows, n_boot=2000, seed=0):
    by_task = defaultdict(list)
    for cand, out in rows:
        by_task[cand["task_id"]].append((cand, out))
    tasks = sorted(by_task)
    if not tasks:
        return {}
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        c = counts([r for t in rng.choices(tasks, k=len(tasks)) for r in by_task[t]])
        for k, (a, b) in RATES.items():
            if c[b]:
                samples[k].append(c[a] / c[b])
    out = {}
    for k, v in samples.items():
        v.sort()
        out[k] = [round(v[int(0.025 * (len(v) - 1))], 3), round(v[int(0.975 * (len(v) - 1))], 3)]
    return out


def summarize(rows, n_boot=2000):
    c = counts(rows)
    r = rates(c)
    return dict(counts=c, rates=r,
                wilson={k: wilson(c[a], c[b]) for k, (a, b) in RATES.items()},
                bootstrap=bootstrap(rows, n_boot))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(pool, va, vb):
    """Per-candidate 'right decision' = accept correct / reject wrong. Paired over the same candidates."""
    b = c = 0
    for cid, cand in pool.items():
        ra = (va[cid] == "ACCEPT") == cand["is_correct"]
        rb = (vb[cid] == "ACCEPT") == cand["is_correct"]
        b += ra and not rb
        c += rb and not ra
    return dict(only_a_right=b, only_b_right=c, p_value=round(mcnemar_exact(b, c), 4))


def load_verdicts(path, pool, strict=True):
    """Load one verdicts file and check it against the shared format (src/igt/types.py, docs/verdict_format.md)."""
    V = load_jsonl(path)
    if not V:
        raise SystemExit(f"{path}: empty")
    names = {v.get("system") for v in V}
    if len(names) != 1:
        raise SystemExit(f"{path}: one system per file, found {names}")
    dup = [k for k, n in Counter(v.get("candidate_id") for v in V).items() if n > 1]
    unknown = sorted({v.get("candidate_id") for v in V} - set(pool))
    missing = sorted(set(pool) - {v.get("candidate_id") for v in V})
    problems = [f"duplicate candidate_id {k}" for k in dup[:3]] + [f"not in pool: {k}" for k in unknown[:3]]
    problems += [f"{len(missing)} pool candidates have no verdict (e.g. {missing[:3]})"] if missing else []
    if strict:
        for v in V:
            problems += [f"{v.get('candidate_id')}: {p}" for p in validate_record(v, pool.get(v.get("candidate_id")))]
    if problems:
        raise SystemExit(f"{path}: {len(problems)} format problem(s):\n  " + "\n  ".join(problems[:10]))
    return names.pop(), {v["candidate_id"]: v["outcome"] for v in V}


def evaluate(pool_path, verdict_paths, n_boot=2000, strict=True):
    pool = {c["candidate_id"]: c for c in load_jsonl(pool_path)}
    systems = {}
    for p in verdict_paths:
        name, out = load_verdicts(p, pool, strict)
        if name in systems:
            raise SystemExit(f"{p}: system name '{name}' used by two files")
        systems[name] = out
    report = {}
    for split in sorted({c["split"] for c in pool.values()}):
        ids = [cid for cid, c in pool.items() if c["split"] == split]
        rep = {}
        for name, out in systems.items():
            rows = [(pool[i], out[i]) for i in ids]
            rep[name] = dict(overall=summarize(rows, n_boot),
                             by_type={t: summarize([r for r in rows if r[0]["transformation_type"] == t], n_boot)
                                      for t in TYPES if any(r[0]["transformation_type"] == t for r in rows)})
        names = list(systems)
        rep["paired"] = {f"{a} vs {b}": compare({i: pool[i] for i in ids}, systems[a], systems[b])
                         for x, a in enumerate(names) for b in names[x + 1:]}
        report[split] = rep
    return report


def fmt_ci(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "-"


def print_table(report):
    for split, rep in report.items():
        print(f"\n=== {split} ===")
        print(f"{'system':<20}{'scope':<23}{'n':>4}{'run':>5}{'wrong':>6}{'acc':>5}"
              f"{'recall':>8} {'boot CI':<14}{'FRR':>6} {'boot CI':<14}{'silent':>7} {'boot CI':<14}")
        for name, s in rep.items():
            if name == "paired":
                continue
            for scope, x in [("overall", s["overall"])] + list(s["by_type"].items()):
                c, r, b = x["counts"], x["rates"], x["bootstrap"]
                f = lambda v: f"{v:.3f}" if v is not None else "  -  "
                print(f"{name:<20}{scope:<23}{c['n']:>4}{c['runnable']:>5}{c['wrong_runnable']:>6}{c['accepted']:>5}"
                      f"{f(r['recall']):>8} {fmt_ci(b.get('recall')):<14}{f(r['false_rejection']):>6} "
                      f"{fmt_ci(b.get('false_rejection')):<14}{f(r['silent_error_rate']):>7} "
                      f"{fmt_ci(b.get('silent_error_rate')):<14}")
        for pair, m in rep["paired"].items():
            print(f"McNemar {pair}: only first right={m['only_a_right']}, only second right={m['only_b_right']}, "
                  f"p={m['p_value']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "benchmark" / "pool" / "pool_v4.jsonl")
    ap.add_argument("--verdicts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, help="write the full report as JSON")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    report = evaluate(args.pool, args.verdicts, args.n_boot)
    print_table(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
