"""Selective input facts and matching parser training, version sf1."""
from pathlib import Path
import sys
_SOURCE=Path(__file__).resolve().parents[2]/'src'
if str(_SOURCE) not in sys.path:sys.path.insert(0,str(_SOURCE))
