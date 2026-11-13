"""Metric definitions copied from the frozen evaluator; equivalence is regression-tested."""
import math
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - m) / d), 3), round(min(1.0, (c + m) / d), 3)]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def rate(k, n):
    return round(k / n, 3) if n else None


def summarize(runs):
    n = len(runs)
    acc = [r for r in runs if r["outcome"] == "ACCEPT"]
    wrong0 = [r for r in runs if r.get("initial_correct") is False]
    right0 = [r for r in runs if r.get("initial_correct") is True]
    entered = [r for r in runs if r.get("attempts", 0) > 0]
    tools = [x for r in runs for x in r.get("tool_trace", [])]
    bad = [x for x in tools if x["error_type"] in ("invalid_args", "unknown_tool")]
    fin = sum(r["final_correct"] for r in runs)
    nor = sum(bool(r.get("no_repair_correct")) for r in runs)
    out = dict(
        n_runs=n, n_tasks=len({r["task_id"] for r in runs}),
        final_accuracy=rate(fin, n), final_accuracy_ci=wilson(fin, n),
        no_repair_accuracy=rate(nor, n), repair_gain=round((fin - nor) / n, 3) if n else None,
        initial_code_accuracy=rate(len(right0), n),
        repair_success=rate(sum(r["final_correct"] for r in wrong0), len(wrong0)),
        unnecessary_retries=rate(sum(r.get("attempts", 0) > 0 for r in right0), len(right0)),
        accept=rate(len(acc), n), reject=rate(sum(r["outcome"] == "REJECT" for r in runs), n),
        escalate=rate(sum(r["outcome"] == "ESCALATE" for r in runs), n),
        parse_escalations=sum(r.get("stage_ended") == "parse" for r in runs),
        silent_error_rate=rate(sum(not r["final_correct"] for r in acc), len(acc)),
        mean_attempts=round(sum(r["attempts"] for r in entered) / len(entered), 2) if entered else None,
        tools=dict(calls=len(tools), valid_call_rate=rate(len(tools) - len(bad), len(tools)),
                   unknown_tool_calls=sum(r.get("unknown_tool_calls", 0) for r in runs)),
        safety=dict(runs_with_unsafe_attempt=sum(r.get("unsafe_attempts", 0) > 0 for r in runs),
                    initial_code_blocked=sum(bool(r.get("initial_blocked")) for r in runs)),
        mean_tokens=round(sum(r.get("tokens", 0) for r in runs) / n) if n else None,
    )
    by = defaultdict(list)
    for r in runs:
        by[r["transformation_type"]].append(r)
    out["by_type"] = {k: dict(n=len(v), final_accuracy=rate(sum(r["final_correct"] for r in v), len(v)),
                              no_repair_accuracy=rate(sum(bool(r.get("no_repair_correct")) for r in v), len(v)))
                      for k, v in sorted(by.items())}
    return out


def pair(a, b):
    ka = {(r["task_id"], r["seed"]): r["final_correct"] for r in a}
    kb = {(r["task_id"], r["seed"]): r["final_correct"] for r in b}
    common = sorted(set(ka) & set(kb))
    x = sum(ka[k] and not kb[k] for k in common)
    y = sum(kb[k] and not ka[k] for k in common)
    return dict(n=len(common), only_first=x, only_second=y, p_value=round(mcnemar_exact(x, y), 4)) if common else None
