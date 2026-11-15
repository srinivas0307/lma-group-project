"""At most one evidence-led parser correction after the unchanged first parse.

Decision rule (fixed before any model run; uses no reference information):
1. No issue fired: keep the first contract unchanged (no extra model call).
2. Otherwise send ONE correction turn to the same parser with the request and the issues.
   - Corrected contract parses, has no proof issue, and has no more issues than the first: use it.
   - Else, if the first contract parsed and had only evidence issues: keep the first contract.
   - Else: withhold at the parse stage with the remaining proof issues as the reason.
"""
import json
import time

from finetune.selective_facts_correction.checks import normalize
from igt.llm import LLMClient
from .diagnostics import VERSION, diagnose

MAX_ISSUES_SHOWN = 8
FEEDBACK = '''Review your previous contract once. Automatic checks compared it with the input data and found the issues below.
They use only the request, the input data and your own contract; they are not reference answers, and "evidence" items may be legitimate.
Return one complete corrected contract in a single ```json code fence, using the same vocabulary.
Rules: keep every requirement the request states; use exact input column names and values that occur in the data; do not delete a constraint just to silence a check; if a check is wrong for this request, keep that part unchanged.
Issues:
{issues}
Original request: {request}'''


def _conversation(parsed):
    messages = [dict(m) for m in parsed.get('messages') or []]
    reply = parsed.get('normalized_output') or parsed.get('raw_output') or ''
    if not messages or not reply:
        return None
    if messages[-1].get('role') != 'assistant':
        messages.append(dict(role='assistant', content=reply))
    return messages


def _has_proof(issues):
    return any(i['kind'] == 'proof' for i in issues)


def check_and_correct(task, before, parsed, backend, context_tokens, max_new_tokens=1024):
    """Return (parsed_record_with_final_contract, check_trace). `task` holds only id/request/data_file."""
    start = time.monotonic()
    first = parsed['extracted']
    issues = diagnose(task['request'], first, before)
    trace = dict(version=VERSION, first_issues=issues, attempted=False, decision='no_issue', llm_calls=0,
                 prompt_tokens=0, output_tokens=0, tokens=0)
    if not issues:
        trace['seconds'] = time.monotonic() - start
        return parsed, trace
    conversation = _conversation(parsed)
    if conversation is None:
        # e.g. selective facts unavailable or context refusal: no reply exists to correct.
        trace.update(decision='no_reply_to_correct', seconds=time.monotonic() - start)
        return parsed, trace
    shown = [dict(code=i['code'], kind=i['kind'], detail=i['detail']) for i in issues[:MAX_ISSUES_SHOWN]]
    conversation.append(dict(role='user', content=FEEDBACK.format(
        issues=json.dumps(shown, ensure_ascii=False, indent=1), request=task['request'])))
    if hasattr(backend, 'prompt_tokens'):
        need = backend.prompt_tokens(conversation) + max_new_tokens
        if need > context_tokens:
            trace.update(needed_tokens=need, seconds=time.monotonic() - start)
            record, trace = _resolve(parsed, None, issues, None, trace)
            trace['context_limit'] = True
            return record, trace
    client = LLMClient(backend)
    response = client.generate(conversation, max_new_tokens=max_new_tokens, temperature=0, seed=0)
    revised = normalize(response.text)
    remaining = diagnose(task['request'], revised['extracted'], before)
    trace.update(attempted=True, llm_calls=client.n_calls, tokens=client.tokens_used,
                 prompt_tokens=getattr(response, 'prompt_tokens', 0), output_tokens=getattr(response, 'output_tokens', 0),
                 correction_messages=conversation, correction_raw=response.text, correction_extracted=revised['extracted'],
                 correction_normalization=revised['changes'], remaining_issues=remaining,
                 seconds=time.monotonic() - start)
    return _resolve(parsed, revised, issues, remaining, trace)


def _resolve(parsed, revised, issues, remaining, trace):
    record = dict(parsed)
    record['tokens'] = parsed.get('tokens', 0) + trace['tokens']
    record['llm_calls'] = parsed.get('llm_calls', 0) + trace['llm_calls']
    record['seconds'] = parsed.get('seconds', 0) + trace.get('seconds', 0)
    record['first_extracted'] = parsed['extracted']
    ok = revised is not None and revised['extracted']['status'] == 'ok'
    if ok and not _has_proof(remaining) and len(remaining) <= len(issues):
        trace['decision'] = 'use_corrected'
        record['extracted'] = revised['extracted']
    elif parsed['extracted']['status'] == 'ok' and not _has_proof(issues):
        trace['decision'] = 'keep_first_evidence_only'
    else:
        trace['decision'] = 'withhold_proven_inconsistency'
        left = remaining if remaining is not None else issues
        reason = json.dumps([i for i in left if i['kind'] == 'proof'][:4] or left[:4], ensure_ascii=False)
        record['extracted'] = dict(status='contract_check_failed', contract=None, error=reason[:1600])
    return record, trace
