### R_test_003 (R, agent_full)

**Request:** Add a column is_married: 'Married' for Married-civ-spouse, Married-spouse-absent and Married-AF-spouse, and 'Not married' for everyone else.

**Outcome:** ACCEPT · evaluator: correct · attempts 1 · tool calls 1 · tokens 3985

**0. the generator's code is checked** → check: VIOLATION (failed: c2)
```python
import pandas as pd

def transform(df):
    df['is_married'] = df['workclass'].apply(lambda x: 'Married' if x in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'] else 'Not married')
    return df
```

**1. calls inspect_dataset(which=last_output, n_rows=5, columns=['is_married'])** → tool ok

**2. proposes a fix** → check: PASS
```python
def transform(df):
    def marital_status_to_is_married(status):
        if status in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
            return 'Married'
        else:
            return 'Not married'
    
    df['is_married'] = df['marital-status'].apply(marital_status_to_is_married)
    return df
```
