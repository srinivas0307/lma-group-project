"""Step 1 of the fine-tuning data (docs/finetune_data_guide.md §3): fixed training snapshots from data/finetune/.

    .venv/bin/python finetune/data_gen/prepare_datasets.py      # -> finetune/datasets/<name>.csv + README.md

For each raw file: read with the right separator, drop index-like / very long columns, take a fixed sample of up to
1,000 rows (seed 0, original row order), then add a small, documented amount of "dirt" so every transformation type
has something to act on: missing values in a few columns, sentinel codes, exact duplicate rows, stray whitespace in
one text column. The injected dirt is listed in README.md. None of these are evaluation datasets.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "finetune"
OUT = ROOT / "finetune" / "datasets"
N_ROWS = 1000

# name: raw file, separator, columns to drop, float rounding, missing-value columns, sentinel (column, value, frac),
#       whitespace column, source, licence
SPECS = {
    "air_quality": dict(file="air_quality_health_impact_data.csv", sep=",", drop=["RecordID"], round=2,
                        nulls=["PM10", "NO2", "Humidity", "SO2", "WindSpeed"], sentinel=[("O3", -999.0, 0.03), ("AQI", -1.0, 0.02)], ws=None,
                        source="Kaggle: Air Quality and Health Impact Dataset (synthetic)", licence="CC BY 4.0 (per Kaggle page)"),
    "avocado": dict(file="avocado.csv", sep=",", drop=["Unnamed: 0"], round=None,
                    nulls=["Large Bags", "XLarge Bags", "Total Bags", "region"], sentinel=[("AveragePrice", 0.0, 0.02)], ws="region",
                    source="Kaggle: Avocado Prices (Hass Avocado Board)", licence="ODbL 1.0 (per Kaggle page)"),
    "bank": dict(file="bank-full.csv", sep=";", drop=[], round=None,
                 nulls=["job", "balance", "marital", "duration"], sentinel=[], ws=None,
                 source="UCI Bank Marketing (bank-full.csv)", licence="CC BY 4.0"),
    "diamonds": dict(file="diamonds.csv", sep=",", drop=["Unnamed: 0"], round=None,
                     nulls=["depth", "table", "color", "price"], sentinel=[("x", 0.0, 0.02), ("y", 0.0, 0.02)], ws=None,
                     source="ggplot2 diamonds", licence="MIT (ggplot2)"),
    "imdb": dict(file="imdb_top_1000.csv", sep=",", drop=["Poster_Link", "Overview"], round=None,
                 nulls=["Director", "Genre"], sentinel=[], ws="Series_Title",
                 source="Kaggle: IMDB Movies Dataset (top 1000)", licence="CC0 (per Kaggle page)"),
    "student": dict(file="student-mat.csv", sep=";", drop=[], round=None,
                    nulls=["absences", "Mjob", "guardian"], sentinel=[], ws=None,
                    source="UCI Student Performance (student-mat.csv)", licence="CC BY 4.0"),
    "superstore": dict(file="train.csv", sep=",", drop=["Row ID", "Country"], round=None,
                       nulls=["Sales", "Segment", "City"], sentinel=[("Sales", 0.0, 0.02)], ws="Customer Name",
                       source="Kaggle: Superstore Sales Dataset (train.csv)", licence="CC BY-SA 4.0 (per Kaggle page)"),
    "walmart": dict(file="Walmart.csv", sep=",", drop=[], round=None,
                    nulls=["CPI", "Unemployment", "Fuel_Price"], sentinel=[("Temperature", -99.0, 0.02), ("Weekly_Sales", 0.0, 0.02)], ws=None,
                    source="Kaggle: Walmart Sales", licence="CC0 (per Kaggle page)"),
}
VALIDATION = ["student"]          # whole datasets used only for validation (never in training)


def prepare(name, spec, rng):
    df = pd.read_csv(RAW / spec["file"], sep=spec["sep"]).drop(columns=spec["drop"])
    if len(df) > N_ROWS:
        df = df.loc[sorted(rng.choice(df.index, N_ROWS - 30, replace=False))].reset_index(drop=True)
    notes = []
    if spec["round"] is not None:
        for c in df.select_dtypes("float").columns:
            df[c] = df[c].round(spec["round"])
        notes.append(f"floats rounded to {spec['round']} decimals")
    if spec["ws"]:
        c = spec["ws"]
        idx = rng.choice(df.index, max(3, int(0.05 * len(df))), replace=False)
        df.loc[idx, c] = [(" " + v if i % 2 else v + "  ") for i, v in enumerate(df.loc[idx, c])]
        notes.append(f"{len(idx)} values of `{c}` given leading/trailing spaces")
    for c in spec["nulls"]:
        idx = rng.choice(df.index, max(3, int(0.05 * len(df))), replace=False)
        df.loc[idx, c] = np.nan
        notes.append(f"{len(idx)} missing values added to `{c}`")
    for c, v, frac in spec["sentinel"]:
        idx = rng.choice(df.index[df[c].notna()], max(3, int(frac * len(df))), replace=False)
        df.loc[idx, c] = v
        notes.append(f"{len(idx)} values of `{c}` set to the code {v} (meaning: not recorded)")
    n_dup = min(30, max(5, int(0.02 * len(df))))
    dup = df.loc[rng.choice(df.index, n_dup, replace=False)]
    pos = sorted(rng.choice(len(df), n_dup, replace=False))
    parts, last = [], 0
    for (_, row), p in zip(dup.iterrows(), pos):
        parts += [df.iloc[last:p + 1], row.to_frame().T]
        last = p + 1
    df = pd.concat(parts + [df.iloc[last:]]).reset_index(drop=True)
    df = df.astype({c: t for c, t in pd.read_csv(RAW / spec["file"], sep=spec["sep"], nrows=200)
                    .drop(columns=spec["drop"]).dtypes.items() if c in df and df[c].notna().all()
                    and str(t).startswith("int")}, errors="ignore")
    notes.append(f"{n_dup} exact duplicate rows inserted after their originals")
    return df, notes


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Fine-tuning datasets (not used in evaluation)", "",
             "Built by `finetune/data_gen/prepare_datasets.py` from the raw files in `data/finetune/` (fixed seed).",
             f"Validation-only datasets (never in training): {', '.join(VALIDATION)}.", ""]
    for name, spec in SPECS.items():
        df, notes = prepare(name, spec, rng)
        p = OUT / f"{name}.csv"
        df.to_csv(p, index=False)
        back = pd.read_csv(p)
        nulls = {c: int(back[c].isna().sum()) for c in back.columns if back[c].isna().any()}
        lines += [f"## {name}" + (" (validation)" if name in VALIDATION else ""), "",
                  f"- source: {spec['source']}; raw file `data/finetune/{spec['file']}`; licence: {spec['licence']}",
                  f"- rows {len(back)}, columns {back.shape[1]}: {', '.join(map(str, back.columns))}",
                  f"- missing values: {nulls or 'none'}",
                  f"- duplicate rows: {int(back.duplicated().sum())}",
                  f"- injected: {'; '.join(notes)}",
                  f"- sha256: `{hashlib.sha256(p.read_bytes()).hexdigest()}`", ""]
        print(f"{name:<12} {back.shape}  nulls={nulls}")
    (OUT / "README.md").write_text("\n".join(lines))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
