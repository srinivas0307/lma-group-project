"""Version and provenance for the profile-plus-checking pilot."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'cq-profile-check-v2'
DECODING = dict(max_new_tokens=1024, do_sample=False, seed=0)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def runtime_hashes():
    paths = list((ROOT/'eval/contract_quality').glob('*.py'))
    paths += list((ROOT/'src/igt').rglob('*.py'))
    paths += [ROOT/'eval/equivalence.py', ROOT/'eval/stage2_clean/attention.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
