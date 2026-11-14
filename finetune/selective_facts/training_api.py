"""One argument contract for the pinned Transformers CPU probe and real training."""
import inspect

def argument_options(config,output_dir,probe=False):
    # Transformers 5.20.2 accepts a fractional warmup_steps (ratio); warmup_ratio was removed.
    return dict(output_dir=str(output_dir),per_device_train_batch_size=1 if probe else config['batch_size'],
        per_device_eval_batch_size=1,gradient_accumulation_steps=1 if probe else config['gradient_accumulation'],
        num_train_epochs=1 if probe else config['epochs'],max_steps=1 if probe else config.get('max_steps',-1),
        learning_rate=1e-3 if probe else config['learning_rate'],warmup_steps=.03,weight_decay=0,
        lr_scheduler_type='cosine',logging_steps=1 if probe else 5,eval_strategy='epoch',save_strategy='no',
        fp16=not probe,bf16=False,optim='adamw_torch' if probe else 'paged_adamw_8bit',gradient_checkpointing=False,
        report_to=[],seed=config['seed'],data_seed=config['seed'],remove_unused_columns=False,
        dataloader_num_workers=0,use_cpu=probe)

def make_arguments(config,output_dir,probe=False,factory=None):
    if factory is None:
        from transformers import TrainingArguments
        factory=TrainingArguments
    options=argument_options(config,output_dir,probe)
    parameters=inspect.signature(factory).parameters
    unknown=set(options)-set(parameters)
    if unknown and not any(p.kind==inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        raise RuntimeError('Installed Transformers TrainingArguments API differs from the pinned sf1 configuration: '+', '.join(sorted(unknown)))
    return factory(**options)
