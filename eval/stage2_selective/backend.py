"""Explicit placement and model lifecycle; parser is unloaded before generator/repair."""
import gc
import time
from .config import MODELS,SETTINGS

class Backend:
    def __init__(self,role,adapter=None):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
        from huggingface_hub import model_info
        self.torch=torch;self.role=role;spec=MODELS[role]
        self.revision=spec['revision'] or model_info(spec['id']).sha
        self.model=spec['id']+(f'+adapter:{adapter}' if adapter else '')
        self.device=f"cuda:{spec['device']}"
        self.tok=AutoTokenizer.from_pretrained(spec['id'],revision=self.revision)
        if self.tok.pad_token is None:self.tok.pad_token=self.tok.eos_token
        options=dict(revision=self.revision,dtype=torch.float16,device_map={'':spec['device']},attn_implementation='sdpa')
        if spec['four_bit']:options['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16)
        self.lm=AutoModelForCausalLM.from_pretrained(spec['id'],**options).eval()
        if adapter:
            from peft import PeftModel
            self.lm=PeftModel.from_pretrained(self.lm,str(adapter)).eval()
        actual={str(p.device) for p in self.lm.parameters()}
        if actual!={self.device}:raise RuntimeError(f'Wrong device placement: {actual}')
        if spec['four_bit'] and not any(type(p).__name__=='Params4bit' for p in self.lm.parameters()):raise RuntimeError('Expected NF4 weights')

    def check_prompt(self,messages,max_new_tokens):
        prompt=self.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        count=len(self.tok(prompt,add_special_tokens=False)['input_ids'])
        if count+max_new_tokens>SETTINGS['parser_context_tokens' if self.role=='parser' else 'context_tokens']:raise ValueError(f'Prompt exceeds declared context budget: {count}+{max_new_tokens}')
        return prompt

    def complete(self,messages,max_new_tokens,temperature,top_p,seed):
        torch=self.torch
        if seed is not None:torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        prompt=self.check_prompt(messages,max_new_tokens)
        inputs=self.tok(prompt,return_tensors='pt',add_special_tokens=False).to(self.device)
        count=int(inputs['input_ids'].shape[1])
        with torch.inference_mode():
            out=self.lm.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=self.tok.pad_token_id)[0,count:]
        return self.tok.decode(out,skip_special_tokens=True),count,len(out)

    def close(self):
        del self.lm,self.tok
        gc.collect();self.torch.cuda.empty_cache()
