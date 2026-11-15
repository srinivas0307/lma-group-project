"""One resident model at a time, on an explicitly selected visible GPU."""
import gc
from igt.llm import BudgetExceeded
from finetune.tool_use.backend import ContextLimitError
from .config import MODELS,COMMON,PARSER_SETTINGS

class Backend:
    def __init__(self,role,method,adapter=None):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        self.torch=torch;self.role=role;self.method=method;self.refusals=[];spec=MODELS[role]
        self.revision=spec['revision'];self.model=spec['id']+(f'+adapter:{adapter}' if adapter else '')
        index=1 if role=='generator' and torch.cuda.device_count()>1 else 0
        self.device=f'cuda:{index}';self.limit=PARSER_SETTINGS[method]['context_tokens'] if role=='parser' else COMMON['context_tokens']
        self.tok=AutoTokenizer.from_pretrained(spec['id'],revision=self.revision)
        if self.tok.pad_token is None:self.tok.pad_token=self.tok.eos_token
        options=dict(revision=self.revision,dtype=torch.float16,device_map={'':index},attn_implementation='sdpa')
        if spec['four_bit']:options['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=False)
        self.lm=AutoModelForCausalLM.from_pretrained(spec['id'],**options).eval()
        if adapter:
            from peft import PeftModel
            self.lm=PeftModel.from_pretrained(self.lm,str(adapter)).eval()
        if {str(p.device) for p in self.lm.parameters()}!={self.device}:raise RuntimeError('Unexpected model device placement')
        if spec['four_bit'] and not any(type(p).__name__=='Params4bit' for p in self.lm.parameters()):raise RuntimeError('NF4 weights missing')
    def check_prompt(self,messages,max_new_tokens):
        prompt=self.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        count=len(self.tok(prompt,add_special_tokens=False)['input_ids'])
        if count+max_new_tokens>self.limit:
            detail=f'{self.role} prompt {count}+{max_new_tokens} exceeds {self.limit}'
            self.refusals.append(dict(role=self.role,prompt_tokens=count,max_new_tokens=max_new_tokens,context_limit=self.limit,reason='context_limit'))
            if self.role=='parser':raise ContextLimitError(detail)
            if self.role=='agent':raise BudgetExceeded('Declared repair context limit: '+detail)
            raise ValueError(detail)
        return prompt
    def prompt_tokens(self,messages):return len(self.tok(self.check_prompt(messages,1024),add_special_tokens=False)['input_ids'])
    def complete(self,messages,max_new_tokens,temperature,top_p,seed):
        torch=self.torch
        if seed is not None:torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        prompt=self.check_prompt(messages,max_new_tokens);inputs=self.tok(prompt,return_tensors='pt',add_special_tokens=False).to(self.device)
        count=int(inputs['input_ids'].shape[1])
        with torch.inference_mode():
            out=self.lm.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=self.tok.pad_token_id)[0,count:]
        return self.tok.decode(out,skip_special_tokens=True),count,len(out)
    def close(self):
        del self.lm,self.tok;gc.collect();self.torch.cuda.empty_cache()
