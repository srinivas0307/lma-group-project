"""CPU collector compatibility fix for JSON mapping keys; frozen experiment stays untouched."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]

def compatible_collector():
    from eval.stage2_combined_gold import collect as frozen
    source=Path(frozen.__file__).read_text()
    old="c['extracted']['contract']!=inventory[row['task_id']][1]['gold_contract']"
    new="c['extracted']['contract']!=json.loads(json.dumps(inventory[row['task_id']][1]['gold_contract']))"
    if source.count(old)!=1:raise ValueError('Unexpected frozen collector; review compatibility fix')
    namespace=dict(vars(frozen));namespace['__name__']='eval.stage2_combined_gold.collect_compatible'
    exec(compile(source.replace(old,new),str(frozen.__file__)+' [JSON-key compatibility]', 'exec'),namespace)
    return namespace['collect']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gold',required=True,type=Path);p.add_argument('--predicted',nargs='*',type=Path,default=[]);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();compatible_collector()(a.gold,a.out,a.predicted)
