from pathlib import Path
from finetune.selective_facts.provenance import ROOT,digest

def hashes():
    paths=list((ROOT/'finetune/tool_use').glob('*.py'))+list((ROOT/'eval/stage2_tools').glob('*.py'))+list((ROOT/'finetune/selective_facts').glob('*.py'))+list((ROOT/'src/igt').rglob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}
