"""Legitimate pandas code must pass the safety checker; dangerous code must not."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness
ns = harness.load()
F = lambda body: "def transform(df):\n" + "\n".join("    " + l for l in body.splitlines()) + "\n    return out"
legit = {
 "list.remove":   "cols = list(df.columns)\ncols.remove('fnlwgt')\nout = df.drop_duplicates(subset=cols, keep='first')",
 "query":         "out = df.query('age >= 18').copy()",
 "df.eval":       "out = df.copy()\nout['r'] = out.eval('a / b')",
 "np.where":      "import numpy as np\nout = df.copy()\nout['x'] = np.where(out['a'] > 0, 1, 0)",
 "re.sub":        "import re\nout = df.copy()\nout['n'] = out['n'].map(lambda s: re.sub(r'\\s+', ' ', s) if isinstance(s, str) else s)",
 "datetime":      "from datetime import datetime\nout = df.copy()\nout['d'] = pd.to_datetime(out['d'], format='%Y-%m-%d')",
 "CategoricalDtype": "from pandas.api.types import CategoricalDtype\nout = df.copy()\nout['c'] = out['c'].astype(CategoricalDtype(['a', 'b'], ordered=True))",
 "str methods":   "out = df.copy()\nout['n'] = out['n'].str.strip().str.lower()",
 "apply lambda":  "out = df.copy()\nout['y'] = out['x'].apply(lambda v: v * 2)",
 "loc assign":    "out = df.copy()\nout.loc[out['a'].isna(), 'a'] = 0",
 "math":          "import math\nout = df.copy()\nout['a'] = out['a'].map(lambda v: math.floor(v) if v == v else v)",
 "dict.pop/get":  "m = {'a': 1}\nm.pop('a', None)\nm.get('b')\nout = df.copy()",
 "drop/rename":   "out = df.rename(columns={'a': 'b'}).drop(columns=['c'])",
 "set ops":       "keep = set(df.columns) - {'x'}\nout = df[[c for c in df.columns if c in keep]].copy()",
 "isin/between":  "out = df[df['a'].isin([1, 2]) & df['b'].between(0, 5)].copy()",
}
bad = {
 "os import": "import os\nos.remove('x')\nout = df",
 "open":      "open('/etc/passwd').read()\nout = df",
 "to_csv":    "df.to_csv('x.csv')\nout = df",
 "np.save":   "import numpy as np\nnp.save('x', df.values)\nout = df",
 "dunder":    "out = df.__class__.__bases__",
 "eval":      "eval('1')\nout = df",
 "subprocess":"import subprocess\nout = df",
}
fails = []
for k, b in legit.items():
    p = ns["safety_check"](F(b))
    if p: fails.append(("legit blocked", k, p))
for k, b in bad.items():
    if not ns["safety_check"](F(b)): fails.append(("dangerous allowed", k))
for f in fails: print(f)
print("PROBLEMS:", len(fails))
