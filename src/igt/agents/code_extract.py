"""Extract the `transform` function from a raw LLM reply (same rules as the Baseline 1 Kaggle run).

- Takes the first fenced ```python block that parses and defines a function; falls back to an
  unterminated block (truncated output) and then to the whole reply.
- Keeps imports, function/class definitions and literal constants; drops top-level example usage
  (e.g. `df = pd.read_csv(...)`, `print(transform(df))`).
- If there is no `transform` but exactly one function, uses it (renamed=True).
- code_hash = sha256 of the normalized code (ast.unparse), used to deduplicate candidates.
"""
import ast
import hashlib
import re

FENCE_RE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def candidate_blocks(text):
    blocks = FENCE_RE.findall(text)
    if "```" in text:                                   # unterminated block (e.g. output truncated)
        tail = text.split("```", 1)[1]
        tail = tail.split("\n", 1)[1] if "\n" in tail else ""
        if "```" not in tail:
            blocks.append(tail)
    blocks.append(text)                                 # unfenced reply
    return blocks


def keep_definitions(tree):
    kept, dropped = [], 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            kept.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            try:
                ast.literal_eval(node.value)
                kept.append(node)
            except Exception:
                dropped += 1
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue                                    # docstring
        else:
            dropped += 1
    return ast.Module(body=kept, type_ignores=[]), dropped


def code_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def extract_code(text):
    reason = "no code block with a function"
    for block in candidate_blocks(text):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            reason = "syntax error"
            continue
        funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not funcs:
            continue
        if "transform" in funcs:
            name, renamed = "transform", False
        elif len(funcs) == 1:
            name, renamed = funcs[0], True
        else:
            return dict(parse_ok=False, parse_reason=f"several functions, none named transform: {funcs}")
        module, dropped = keep_definitions(tree)
        code = ast.unparse(module)
        return dict(parse_ok=True, code=code, code_hash=code_hash(code), func_name=name,
                    renamed=renamed, top_level_dropped=dropped)
    return dict(parse_ok=False, parse_reason=reason)
