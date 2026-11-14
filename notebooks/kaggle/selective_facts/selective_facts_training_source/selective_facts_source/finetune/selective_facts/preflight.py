"""Kaggle-only tiny CPU Trainer/LoRA/save/reload probe, before downloading model weights."""
import tempfile
from pathlib import Path

def main():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM,Trainer,TrainingArguments
    from peft import LoraConfig,get_peft_model,PeftModel
    from .train import Collator
    with tempfile.TemporaryDirectory(prefix='sf1_trainer_probe_') as temp:
        config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,eos_token_id=2)
        model=get_peft_model(Qwen2ForCausalLM(config),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],task_type='CAUSAL_LM'))
        row=dict(input_ids=[3,4,5,2],attention_mask=[1]*4,labels=[-100,-100,5,2])
        args=TrainingArguments(output_dir=temp,per_device_train_batch_size=1,per_device_eval_batch_size=1,max_steps=1,
            learning_rate=1e-3,warmup_ratio=.03,eval_strategy='epoch',save_strategy='no',report_to=[],use_cpu=True,
            optim='adamw_torch',remove_unused_columns=False,dataloader_num_workers=0)
        trainer=Trainer(model=model,args=args,train_dataset=[row],eval_dataset=[row],data_collator=Collator(0))
        result=trainer.train()
        if not torch.isfinite(torch.tensor(result.training_loss)):raise RuntimeError('Nonfinite tiny training loss')
        adapter=Path(temp)/'adapter';model.save_pretrained(adapter)
        reloaded=PeftModel.from_pretrained(Qwen2ForCausalLM(config),adapter).eval()
        with torch.inference_mode():out=reloaded(input_ids=torch.tensor([[3,4,5,2]]))
        if out.logits.shape!=(1,4,32):raise RuntimeError('Tiny adapter reload failed')
    print('PASSED: tiny CPU Trainer, assistant labels, LoRA optimization, adapter save/reload',flush=True)
if __name__=='__main__':main()
