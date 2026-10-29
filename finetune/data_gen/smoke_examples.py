"""Pipeline-test data only (NOT training data): a synthetic shop dataset and ~20 template examples.

    .venv/bin/python finetune/data_gen/smoke_examples.py      # -> finetune/datasets/smoke_shop.csv, finetune/data/smoke/raw.jsonl

Used to smoke-test check_examples.py -> build_sft.py -> train_qlora.py -> parse_contracts.py --adapter.
It also shows the shape a real generator (docs/finetune_data_guide.md §5) should produce.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = "finetune/datasets/smoke_shop.csv"


def make_data():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "order_id": np.arange(1001, 1001 + n),
        "customer": rng.choice([f"cust_{i:03d}" for i in range(80)], n),
        "category": rng.choice(["books", "toys", "garden", "kitchen"], n),
        "quantity": rng.integers(1, 9, n),
        "unit_price": rng.choice([2.5, 4.0, 9.99, 15.0, 24.5], n),
        "coupon": np.where(rng.random(n) < 0.3, None, rng.choice(["SPRING", "VIP", "NONE"], n)),
    })
    df.loc[rng.choice(n, 12, replace=False), "quantity"] = -1          # returns
    dup = df.sample(15, random_state=3)
    df = pd.concat([df, dup.assign(order_id=dup["order_id"] + 5000)], ignore_index=True)
    (ROOT / DATA).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / DATA, index=False)
    return df


KEEP_ALL = {"type": "columns_unchanged", "target": "*", "except": []}


def examples(df):
    out = []
    styles = {
        "plain": "Remove the orders where {c} is {op_words} {v}.",
        "casual": "get rid of anything with {c} {op_words} {v}",
        "domain": "Drop every order whose {c} is {op_words} {v} please.",
    }
    words = {"<": "below", ">": "above"}
    i = 0
    for c in ["quantity", "unit_price"]:
        for op, inv in [("<", ">="), (">", "<=")]:
            vals = sorted(df[c].unique())
            v = float(vals[1]) if op == "<" else float(vals[-2])
            v = int(v) if float(v).is_integer() else v
            for style, tmpl in styles.items():
                i += 1
                out.append(dict(
                    id=f"smoke_filt_{i:03d}", source_dataset="smoke_shop" if i % 4 else "smoke_shop_val",
                    data_file=DATA, transformation_type="filtering", pattern=f"filter_{op}", style=style,
                    request=tmpl.format(c=c, op_words=words[op], v=v),
                    contract={"contract_version": "1.0",
                              "must_change": [{"type": "rows_removed_where", "target": None,
                                               "condition": {"column": c, "op": op, "value": v}},
                                              {"type": "row_count", "target": None, "op": "decreased"}],
                              "must_preserve": [{"type": "rows_retained_where", "target": None,
                                                 "condition": {"column": c, "op": inv, "value": v}}, KEEP_ALL]},
                    reference_code=f"def transform(df):\n    return df[~(df[{c!r}] {op} {v!r})].copy()\n"))
    for keep in ["first", "last"]:
        for style, req in {"plain": f"Orders with the same customer, category, quantity, unit_price and coupon are duplicates; keep the {keep} one.",
                           "casual": f"dedupe the orders (ignore order_id), keeping the {keep}"}.items():
            i += 1
            out.append(dict(
                id=f"smoke_dedup_{i:03d}", source_dataset="smoke_shop" if i % 4 else "smoke_shop_val",
                data_file=DATA, transformation_type="deduplication", pattern=f"dedup_except_{keep}", style=style,
                request=req,
                contract={"contract_version": "1.0",
                          "must_change": [{"type": "duplicate_count", "target": "*", "except": ["order_id"], "op": "eq",
                                           "value": 0, "keep": keep},
                                          {"type": "row_count", "target": None, "op": "decreased"}],
                          "must_preserve": [KEEP_ALL]},
                reference_code=("def transform(df):\n    keys = [c for c in df.columns if c != 'order_id']\n"
                                f"    return df.drop_duplicates(subset=keys, keep={keep!r}).copy()\n")))
    for style, req in {"plain": "Fill the missing coupon values with NONE.",
                       "casual": "where there's no coupon put NONE",
                       "domain": "Orders without a coupon code should get the code NONE."}.items():
        i += 1
        out.append(dict(
            id=f"smoke_miss_{i:03d}", source_dataset="smoke_shop" if i % 4 else "smoke_shop_val",
            data_file=DATA, transformation_type="missing_values", pattern="fill_const", style=style, request=req,
            contract={"contract_version": "1.0",
                      "must_change": [{"type": "column_value_relation", "target": "coupon",
                                       "where": {"column": "coupon", "op": "is_null"}, "fn": "const",
                                       "args": [{"lit": "NONE"}]},
                                      {"type": "null_count", "target": "coupon", "op": "eq", "value": 0}],
                      "must_preserve": [{"type": "column_values_unchanged", "target": "coupon",
                                         "where": {"column": "coupon", "op": "not_null"}},
                                        {"type": "columns_unchanged", "target": "*", "except": ["coupon"]},
                                        {"type": "row_count", "target": None, "op": "unchanged"}]},
            reference_code="def transform(df):\n    out = df.copy()\n    out['coupon'] = out['coupon'].fillna('NONE')\n    return out\n"))
    return out


def main():
    df = make_data()
    ex = examples(df)
    p = ROOT / "finetune" / "data" / "smoke" / "raw.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(e) + "\n" for e in ex))
    print(f"{len(df)} rows -> {DATA}; {len(ex)} examples -> {p}")


if __name__ == "__main__":
    main()
