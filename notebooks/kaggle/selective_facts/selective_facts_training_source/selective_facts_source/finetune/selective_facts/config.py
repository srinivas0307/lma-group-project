"""Pinned settings for the sf1 experiment, separate from completed Stage 2 runs."""
MODEL='Qwen/Qwen2.5-1.5B-Instruct'
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
MODELS={'parser':dict(id=MODEL,revision=REVISION,device=0,four_bit=True)}
SETTINGS=dict(parser_context_tokens=8192,context_tokens=8192)
TRAIN=dict(seed=1234,epochs=2,learning_rate=2e-4,lora_r=16,lora_alpha=32,lora_dropout=.05,
           batch_size=1,gradient_accumulation=16,max_seq_len=4096)
