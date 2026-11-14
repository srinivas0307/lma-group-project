"""Tool registry (README Phase 3): allow-listed tools with JSON argument schemas and structured errors.

    reg = ToolRegistry([Tool(...), ...])
    res = reg.call(ToolCall("profile_column", {"column": "price"}))   # never raises
    res.ok, res.output, res.error  # error = {"type": ..., "message": ...}

Error types (returned to the agent as observations, counted in the tool-use metrics):
    unknown_tool      name not in the registry (also counted as an unauthorized-tool attempt)
    invalid_args      missing / extra / wrongly typed arguments (message says which)
    tool_error        the handler raised (bad column, nothing to diff, ...)
    injected_fault    a fault hook fired (tests and T scenarios only)
    timeout           a (real or injected) timeout
Every call is appended to `reg.log` (name, args, ok, error type, duration) for traces and metrics.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Callable

MAX_OUTPUT_CHARS = 6000

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


@dataclass
class Tool:
    name: str
    description: str
    args_schema: dict              # {"type": "object", "properties": {...}, "required": [...]}
    handler: Callable              # handler(**args) -> JSON-serialisable dict

    def spec(self) -> dict:
        """What the agent prompt shows: name, description, argument schema."""
        return dict(name=self.name, description=self.description, args=self.args_schema)


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: dict | None = None
    error: dict | None = None
    duration_s: float = 0.0

    def to_observation(self) -> dict:
        d = dict(tool=self.name, ok=self.ok)
        d.update(output=self.output) if self.ok else d.update(error=self.error)
        return d


def _type_ok(value, schema):
    t = schema.get("type")
    if t is None:
        return True
    py = _TYPES[t]
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    if not isinstance(value, py):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if t == "array" and "items" in schema:
        return all(_type_ok(v, schema["items"]) for v in value)
    if t in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    return True


def validate_args(args, schema) -> list[str]:
    """Return a list of problems (empty = valid). Supports the JSON-schema subset the tools use."""
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    props, req = schema.get("properties", {}), schema.get("required", [])
    p = [f"missing required argument '{k}'" for k in req if k not in args]
    p += [f"unknown argument '{k}' (allowed: {sorted(props)})" for k in args if k not in props]
    for k, v in args.items():
        if k in props and not _type_ok(v, props[k]):
            want = props[k].get("type") + (f" in {props[k]['enum']}" if "enum" in props[k] else "")
            p.append(f"argument '{k}' must be {want}, got {json.dumps(v, default=str)[:60]}")
    return p


def _clip(output):
    s = json.dumps(output, default=str)
    if len(s) <= MAX_OUTPUT_CHARS:
        return json.loads(s)
    return dict(truncated=True, preview=s[:MAX_OUTPUT_CHARS])


class ToolRegistry:
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}
        self.log = []

    def specs(self):
        return [t.spec() for t in self.tools.values()]

    def call(self, call: ToolCall) -> ToolResult:
        t0 = time.time()
        res = self._call(call)
        res.duration_s = round(time.time() - t0, 4)
        self.log.append(dict(name=call.name, args=call.args, ok=res.ok,
                             error_type=(res.error or {}).get("type"), duration_s=res.duration_s))
        return res

    def _call(self, call):
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(call.name, False, error=dict(type="unknown_tool",
                              message=f"no tool named '{call.name}'. Available: {sorted(self.tools)}"))
        problems = validate_args(call.args, tool.args_schema)
        if problems:
            return ToolResult(call.name, False, error=dict(type="invalid_args", message="; ".join(problems)))
        try:
            return ToolResult(call.name, True, output=_clip(tool.handler(**call.args)))
        except TimeoutError as e:
            return ToolResult(call.name, False, error=dict(type="timeout", message=str(e)))
        except Exception as e:
            return ToolResult(call.name, False, error=dict(type="tool_error", message=f"{type(e).__name__}: {e}"[:500]))
