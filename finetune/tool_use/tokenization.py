"""Supervise every assistant turn, never user prompts or tool observations."""
def tokenize(rows,tok,max_length):
    result=[]
    for row in rows:
        messages=row['messages']
        full=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        ids=tok(full,add_special_tokens=False)['input_ids'];labels=[-100]*len(ids)
        if len(ids)>max_length:raise ValueError(f'{row["id"]}: {len(ids)} tokens exceeds {max_length}; no truncation allowed')
        for i,message in enumerate(messages):
            if message['role']!='assistant':continue
            prefix=tok.apply_chat_template(messages[:i],tokenize=False,add_generation_prompt=True)
            through=tok.apply_chat_template(messages[:i+1],tokenize=False,add_generation_prompt=False)
            before=tok(prefix,add_special_tokens=False)['input_ids'];after=tok(through,add_special_tokens=False)['input_ids']
            if not through.startswith(prefix) or ids[:len(before)]!=before or ids[:len(after)]!=after:raise ValueError('Chat template token boundary mismatch')
            if len(after)<=len(before):raise ValueError('Empty assistant supervision')
            labels[len(before):len(after)]=ids[len(before):len(after)]
        if all(v==-100 for v in labels):raise ValueError('No assistant targets')
        result.append(dict(input_ids=ids,attention_mask=[1]*len(ids),labels=labels))
    return result
