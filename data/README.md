# Data

Every file the project reads as input data. The evaluation files are the "before" data of all 120 tasks and every
agent scenario; task and scenario files refer to them by path (`data_file: data/<file>`), so they must not be moved,
edited or re-saved. Their SHA-256 checksums are in `data/checksums.json` and checked by `tests/test_data_checksums.py`.

Loading rule (README §4, G9): `pd.read_csv(path)` with no options. The default index 0..n-1 is the row id; nothing is
added to the data. Library versions are pinned (`requirements.txt`: pandas 2.3.3, numpy 2.1.3), because dtypes such as
`object` vs `string` differ between pandas versions.

## Evaluation datasets

| File | Dataset | Rows × columns | Missing cells (column: count) | Exact duplicate rows | Tasks |
|---|---|---|---|---|---|
| `adult_dev.csv` | UCI Adult (census income) | 1,000 × 15 | workclass 62, occupation 62, native-country 18 | 0 | 30 (`adult_*`) |
| `airbnb_dev.csv` | NYC Airbnb Open Data (2019 listings) | 1,000 × 16 | host_name 1, last_review 41, reviews_per_month 41 | 0 | 30 (`airbnb_*`) |
| `Titanic-Dataset.csv` | Titanic passenger list | 891 × 12 | Age 177, Cabin 687, Embarked 2 | 0 | 30 (`titanic_*`) |
| `yellow_tripdata_2026-01.csv` | NYC TLC Yellow Taxi trip records, January 2026 | 1,000 × 20 | none (missing data is coded as sentinels, e.g. `passenger_count` 0, `RatecodeID` 99) | 0 | 30 (`taxi_*`) |

| File | Bytes | SHA-256 |
|---|---|---|
| `adult_dev.csv` | 107,909 | `b2ebdd9b244c7fff5ebe1d6a328cc49f37b7a92aa4c2cb6ad0f2b5671e49d2ef` |
| `airbnb_dev.csv` | 142,151 | `c25f044eb00cd2c88bf05661bf50726c00890f8cb3aecd1a4770ec9f032e68a3` |
| `Titanic-Dataset.csv` | 61,194 | `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f` |
| `yellow_tripdata_2026-01.csv` | 110,519 | `b1ac75514b2adec94444606a9b5ec7b60a24b02530144592d2e64a2fce891ecc` |

Notes from profiling (guide §4.2):

- **adult:** missing values are empty cells (NaN), not the string `?` of the original UCI release.
- **airbnb:** `price` ranges 20 to 3000 (no zeros). `airbnb_dev.csv` has 1,003 text lines (header + 1,000 rows) because some
  listing names contain line breaks inside quotes.
- **adult, airbnb:** no exact duplicate rows, so every deduplication task uses a key subset.
- **taxi:** datetimes are text in the CSV (`tpep_pickup_datetime`, `tpep_dropoff_datetime`); type-conversion tasks
  convert them.

**Not recorded in the project (fill in):**

- How the 1,000-row files `adult_dev.csv`, `airbnb_dev.csv` and `yellow_tripdata_2026-01.csv` were drawn from the full
  downloads (first rows or a random sample, and the seed).
- The download URL and licence of each of the four datasets. Copy them from the pages they were downloaded from; the
  project write-up needs them.

The Wine Quality dataset named in the proposal is not used for evaluation (it has no missing values). It may appear in
fine-tuning data only and must never be added to evaluation (`docs/finetune_data_guide.md`).

## Fine-tuning datasets

`data/finetune/` holds the raw downloads used **only** for fine-tuning data. They are never evaluation data.
`finetune/data_gen/prepare_datasets.py` turns them into the ~1,000-row snapshots in `finetune/datasets/`, whose
sources, licences, injected values and checksums are documented in `finetune/datasets/README.md`.

| Raw file | Dataset |
|---|---|
| `air_quality_health_impact_data.csv` | Air Quality and Health Impact (Kaggle) |
| `avocado.csv` | Avocado Prices (Kaggle) |
| `bank-full.csv` | UCI Bank Marketing |
| `diamonds.csv` | ggplot2 diamonds |
| `imdb_top_1000.csv` | IMDB top 1000 movies (Kaggle) |
| `student-mat.csv` | UCI Student Performance (validation only) |
| `train.csv` | Superstore Sales (Kaggle) |
| `Walmart.csv` | Walmart sales (Kaggle) |
