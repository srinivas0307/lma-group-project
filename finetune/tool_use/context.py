"""Identical prompt for base and trained models; observations are supplied only on demand."""
from eval.stage2_tools.context import parser_messages as original_messages
from .config import PROTOCOL
EXAMPLES='\nProtocol examples (illustrations, not facts about your input):\nRequest: Fill missing score with its full-column median.\nFirst reply: {"tool":"profile_column","args":{"column":"score"}}\nAfter a successful observation with statistics.median = 12.75, use 12.75 as the fill literal in the final contract, only on originally missing cells; preserve other cells and columns.\nRequest: Fill missing score in region = west with the median among rows with region = west.\nFirst reply: {"tool":"profile_column","args":{"column":"score","where":{"column":"region","op":"==","value":"west"}}}\nRequest: Fill missing score with 0.\nReply directly with the final contract; the value is already supplied.\nIf a requested mode has a tie without a specified tie rule, escalate rather than choose arbitrarily.\nIf a query fails, correct its arguments using the input schema, or escalate; never treat an error as a data value.\n'
def parser_messages(task,df,prompt='p1-v2'):
    messages=original_messages(task,df,prompt)
    messages[0]['content']=messages[0]['content'].replace('s2-parser-tools-v1',PROTOCOL)+EXAMPLES
    return messages

def observation_message(observation):
    import json
    return dict(role='user',content='Read-only input-data tool result:\n'+json.dumps(observation,ensure_ascii=False,allow_nan=False)+'\nChoose another tool, return the final contract, or escalate.')
