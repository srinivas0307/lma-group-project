EXPERIMENT='stage2-independent-v1'
SYSTEMS=['baseline1','baseline2_repair','baseline3_repair','gold_contract','proposed','no_contract_to_generator']
MODELS={
 'generator':dict(id='Qwen/Qwen2.5-3B-Instruct',revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',device=1,four_bit=False),
 'agent':dict(id='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',device=0,four_bit=True),
 'parser':dict(id='Qwen/Qwen2.5-1.5B-Instruct',revision=None,device=0,four_bit=False),
}
# Hand-set configuration, not a result from an earlier tuning experiment.
THRESHOLDS=dict(max_dropped_columns=0,max_row_ratio=1.0,min_row_ratio=0.5,max_index_not_subset=0,
 max_null_increase=0.0,max_new_column_null_frac=0.5,dtype_mode='compatible',max_lost_unique_columns=0,
 max_duplicate_rows_increase=0,max_out_of_range_frac=0.0,max_mean_shift_z=0.5,max_category_tvd=0.1,max_new_value_frac=0.01)
SETTINGS=dict(seed=0,parser_input='input-profile-s2-v1',parser_max_new_tokens=1024,generator_max_new_tokens=512,
 max_attempts=3,max_tool_calls=8,token_budget=60000,parser_context_tokens=8192,context_tokens=32768,
 prompted_parser='p1-v3-zero-shot',finetuned_parser='p1-v2-zero-shot',conventional='hand-set-untuned-v1')
