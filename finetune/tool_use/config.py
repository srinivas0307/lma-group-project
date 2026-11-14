"""Shared, pinned base models for both experimental arms."""
PROTOCOL='parser-tools-v2'
MODELS={
 '1_5b':dict(id='Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
 '3b':dict(id='Qwen/Qwen2.5-3B-Instruct',revision='aa8e72537993ba99e69dfaafa59ed015b17504d1')}
SETTINGS=dict(parser_context_tokens=12288,context_tokens=12288)
