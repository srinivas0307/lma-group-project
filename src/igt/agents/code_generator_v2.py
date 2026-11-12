"""Opt-in b2-v2 prompt, designed from Stage 2 development failures.

The original b1-v1/b2-v1 prompts remain unchanged. No benchmark examples or
reference solutions are included in these general implementation rules.
"""
import hashlib

from . import code_generator as original

PROMPT_VERSION = "b2-v2"
IMPLEMENTATION_RULES = """

Implementation discipline (pandas 2.3.3, numpy 2.1.3):
- Implement the user's requested edit with the smallest clear transformation. The contract describes checks on the result; do not implement each predicate as a separate edit.
- Use exact column names from the schema, including punctuation. Access columns with df["exact name"], not invented Python identifiers.
- Start with out = df.copy(). Preserve all unrelated columns, values and dtype families. Do not cast a column merely to filter or deduplicate it.
- For removal, build the complete condition identifying unwanted rows and keep its complement. Respect the grouping of AND/OR conditions and the request's treatment of nulls. Do not concatenate removed rows back into the result.
- Deduplicate on precisely the requested composite key and keep rule, in original input order. Avoid sorting, index resets and helper columns unless required. Remove any temporary helper columns before returning.
- Missing data means a pandas/numpy null, not the text "missing", unless the request explicitly asks for a literal string. Preserve existing non-missing values.
- Use current pandas APIs: pd.concat rather than DataFrame.append; Series.between uses inclusive="both", "left", "right" or "neither", never a boolean.
- Before returning, check that only requested rows, columns, values and types changed. Do not print this check or any explanation; return the complete transform(df) function in one Python code block.
"""
SYSTEM_PROMPT = original.CONTRACT_SYSTEM_PROMPT + IMPLEMENTATION_RULES
PROMPT_HASH = hashlib.sha256((PROMPT_VERSION + SYSTEM_PROMPT + original.USER_TEMPLATE
                              + original.CONTRACT_USER_SUFFIX).encode()).hexdigest()[:12]


def build_generator_messages(task, schema, contract=None):
    if contract is None:
        return original.build_generator_messages(task, schema)
    messages = original.build_generator_messages(task, schema, contract)
    return [dict(messages[0], content=SYSTEM_PROMPT), messages[1]]


def generator_prompt_info(with_contract):
    return dict(version=PROMPT_VERSION, hash=PROMPT_HASH) if with_contract else original.generator_prompt_info(False)
