"""Exactly the selective-facts format used for SFT; prompted arm keeps its reference card."""
from finetune.selective_facts.context import messages,schema_entry,RULES
from igt.agents.intent_parser import PROMPTS

def parser_messages(task,df,prompt='p1-v2'):
    result=messages(task,df)
    if prompt!='p1-v2':result[0]['content']=PROMPTS[prompt]+RULES
    return result
