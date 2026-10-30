"""Fault-injection hooks for agent evaluation (README Phase 3, T scenarios).

    reg = FaultInjector(build_registry(ctx), [{"tool": "profile_column", "on_call": 1, "kind": "error"}])

Each rule fires exactly on the n-th call (1-based) of that tool:
    error      -> ToolResult(ok=False, error.type = "injected_fault")
    timeout    -> ToolResult(ok=False, error.type = "timeout")
    malformed  -> ToolResult(ok=True) whose output is garbled (agent should notice and retry or move on)
The injector has the same interface as ToolRegistry (call, specs, log, tools) and records which faults fired.
"""
from collections import Counter

from .registry import ToolCall, ToolResult

KINDS = ("error", "timeout", "malformed")


class FaultInjector:
    def __init__(self, registry, rules):
        for r in rules:
            if r["kind"] not in KINDS:
                raise ValueError(f"fault kind must be one of {KINDS}")
            if int(r["on_call"]) < 1:
                raise ValueError("on_call is 1-based")
        self.registry = registry
        self.rules = [dict(r) for r in rules]
        self.counts = Counter()
        self.fired = []

    @property
    def tools(self):
        return self.registry.tools

    @property
    def log(self):
        return self.registry.log

    def specs(self):
        return self.registry.specs()

    def call(self, call: ToolCall) -> ToolResult:
        self.counts[call.name] += 1
        n = self.counts[call.name]
        rule = next((r for r in self.rules if r["tool"] == call.name and int(r["on_call"]) == n), None)
        if rule is None:
            return self.registry.call(call)
        self.fired.append(dict(tool=call.name, on_call=n, kind=rule["kind"]))
        if rule["kind"] == "error":
            res = ToolResult(call.name, False, error=dict(type="injected_fault", message="internal tool error, try again"))
        elif rule["kind"] == "timeout":
            res = ToolResult(call.name, False, error=dict(type="timeout", message="tool timed out"))
        else:
            res = ToolResult(call.name, True, output=dict(garbled="\x00\x00#%&<<<truncated"))
        self.registry.log.append(dict(name=call.name, args=call.args, ok=res.ok,
                                      error_type=(res.error or {}).get("type"), injected=rule["kind"], duration_s=0.0))
        return res
