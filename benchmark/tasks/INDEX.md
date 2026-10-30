# Task index (draft)

Tasks for `airbnb_dev.csv`, `adult_dev.csv`, `Titanic-Dataset.csv` (891 rows) and `yellow_tripdata_2026-01.csv` (1000 rows): 6 per transformation type per dataset (120 total). `dev` = used only for prompt/threshold tuning.

| id | split | difficulty | suggested idea |
|---|---|---|---|
| `airbnb_filt_001` | test | easy | Remove listings that are never available (availability_365 is 0). |
| `airbnb_filt_002` | test | easy | Keep only listings in Brooklyn and Manhattan. |
| `airbnb_filt_003` | test | medium | Remove listings priced at 500 or more. |
| `airbnb_filt_004` | test | medium | Remove listings that require a stay of more than 30 nights. |
| `airbnb_filt_005` | test | medium | Remove listings that have never been reviewed. |
| `airbnb_filt_006` | dev | hard | Remove private rooms in Manhattan that cost less than 60. |
| `airbnb_dedup_001` | test | medium | Listings with the same name are duplicates; keep the first one. |
| `airbnb_dedup_002` | test | medium | Remove duplicate listings where the same host posted the same name twice; keep the first. |
| `airbnb_dedup_003` | test | hard | Keep only one listing per host (the first one that appears). |
| `airbnb_dedup_004` | test | hard | Keep one listing per host and room type, keeping the last one. |
| `airbnb_dedup_005` | test | medium | Each host should have at most one listing per neighbourhood; keep the first. |
| `airbnb_dedup_006` | dev | hard | Remove repeat listings from the same host with the same room type and price; keep the first. |
| `airbnb_miss_001` | test | easy | Fill missing reviews_per_month with 0. |
| `airbnb_miss_002` | test | easy | Fill missing host names with 'Unknown'. |
| `airbnb_miss_003` | test | medium | Drop listings that have no last review date. |
| `airbnb_miss_004` | test | medium | Fill missing reviews_per_month with the median of that column. |
| `airbnb_miss_005` | test | hard | For listings with no reviews, set reviews_per_month to 0. |
| `airbnb_miss_006` | dev | medium | Where host_name is missing, use the listing name instead. |
| `airbnb_conv_001` | test | medium | Convert last_review to a datetime column. |
| `airbnb_conv_002` | test | easy | Store price as a float instead of an integer. |
| `airbnb_conv_003` | test | medium | Make room_type a categorical column. |
| `airbnb_conv_004` | test | medium | Make neighbourhood_group a categorical column. |
| `airbnb_conv_005` | test | hard | Make room_type an ordered category: Shared room < Private room < Entire home/apt. |
| `airbnb_conv_006` | dev | easy | Store the listing id as text instead of a number. |
| `airbnb_col_001` | test | easy | Add a column minimum_stay_cost equal to price times minimum_nights. |
| `airbnb_col_002` | test | medium | Strip leading and trailing whitespace from listing names. |
| `airbnb_col_003` | test | medium | Cap minimum_nights at 30 (values above 30 become 30). |
| `airbnb_col_004` | test | easy | Add availability_ratio = availability_365 divided by 365. |
| `airbnb_col_005` | test | medium | Shorten room_type values: 'Entire home/apt' -> 'entire', 'Private room' -> 'private', 'Shared room' -> 'shared'. |
| `airbnb_col_006` | dev | medium | Rename reviews_per_month to monthly_reviews. |
| `adult_filt_001` | test | easy | Remove people younger than 18. |
| `adult_filt_002` | test | medium | Remove records with more than 60 working hours per week. |
| `adult_filt_003` | test | medium | Keep only people whose native country is United-States. |
| `adult_filt_004` | test | medium | Remove people whose education is Preschool or 1st-4th. |
| `adult_filt_005` | test | hard | Remove self-employed people (incorporated or not) older than 60. |
| `adult_filt_006` | dev | hard | Remove private-sector workers who work fewer than 20 hours a week. |
| `adult_dedup_001` | test | medium | Remove duplicate records that are identical in every column except fnlwgt; keep the first. |
| `adult_dedup_002` | test | hard | Same key as dedup_001, but keep the last occurrence. |
| `adult_dedup_003` | test | medium | fnlwgt should identify a record; remove repeats of the same fnlwgt, keeping the first. |
| `adult_dedup_004` | test | hard | Treat records with the same age, workclass, education, occupation and hours-per-week as duplicates; keep the first. |
| `adult_dedup_005` | test | hard | Remove duplicates on age, education, marital-status, occupation, relationship, race and sex; keep the first. |
| `adult_dedup_006` | dev | medium | Remove records identical in every column except fnlwgt and income; keep the first. |
| `adult_miss_001` | test | easy | Fill missing workclass values with 'Unknown'. |
| `adult_miss_002` | test | easy | Fill missing occupation values with 'Unknown'. |
| `adult_miss_003` | test | medium | Drop records with a missing native country. |
| `adult_miss_004` | test | medium | Fill missing native-country with the most common country (United-States). |
| `adult_miss_005` | test | hard | Fill missing workclass and occupation with 'Not-working'. |
| `adult_miss_006` | dev | easy | Drop every record that has any missing value. |
| `adult_conv_001` | test | easy | Make income a categorical column. |
| `adult_conv_002` | test | medium | Convert income to a number: 1 for >50K and 0 for <=50K. |
| `adult_conv_003` | test | hard | Make education an ordered category following the education-num order (Preschool lowest, Doctorate highest). |
| `adult_conv_004` | test | easy | Store age as a float. |
| `adult_conv_005` | test | medium | Make sex a categorical column. |
| `adult_conv_006` | dev | medium | Make workclass a categorical column, keeping missing values as missing. |
| `adult_col_001` | test | easy | Add net_capital = capital-gain minus capital-loss. |
| `adult_col_002` | test | easy | Add hours_per_day = hours-per-week divided by 5. |
| `adult_col_003` | test | medium | Cap hours-per-week at 60 (values above 60 become 60). |
| `adult_col_004` | test | medium | Lower-case all occupation values. |
| `adult_col_005` | test | hard | Add is_married: 'Married' for Married-civ-spouse, Married-spouse-absent, Married-AF-spouse; 'Not married' otherwise. |
| `adult_col_006` | dev | medium | Rename education-num to education_years. |
| `titanic_filt_001` | test | easy | Remove passengers whose fare was 0. |
| `titanic_filt_002` | test | medium | Keep only passengers who embarked at Southampton or Cherbourg. |
| `titanic_filt_003` | test | medium | Remove passengers younger than 18; unknown ages stay. |
| `titanic_filt_004` | test | medium | Remove male passengers travelling in third class. |
| `titanic_filt_005` | test | medium | Keep only passengers travelling alone. |
| `titanic_filt_006` | dev | hard | Remove first-class passengers who paid less than 30. |
| `titanic_dedup_001` | test | medium | Passengers sharing a ticket are duplicates; keep the first. |
| `titanic_dedup_002` | test | hard | Same key as dedup_001, but keep the last passenger. |
| `titanic_dedup_003` | test | medium | Duplicates on Ticket, Pclass and Embarked; keep the first. |
| `titanic_dedup_004` | test | medium | Keep one passenger per class and sex (the first). |
| `titanic_dedup_005` | test | hard | Same ticket and same cabin are duplicates; keep the last. |
| `titanic_dedup_006` | dev | medium | Same ticket and fare is one booking; keep the first. |
| `titanic_miss_001` | test | easy | Fill missing Cabin values with 'Unknown'. |
| `titanic_miss_002` | test | medium | Fill missing Embarked with the most common port. |
| `titanic_miss_003` | test | easy | Drop passengers with no port of embarkation. |
| `titanic_miss_004` | test | medium | Fill missing Age with the median age. |
| `titanic_miss_005` | test | hard | Fill missing Age with the median age of the passenger's class. |
| `titanic_miss_006` | dev | easy | Drop passengers with a missing Age. |
| `titanic_conv_001` | test | easy | Make Sex a categorical column. |
| `titanic_conv_002` | test | hard | Make Pclass an ordered category: 3 < 2 < 1. |
| `titanic_conv_003` | test | medium | Convert Survived (0/1) to a boolean column. |
| `titanic_conv_004` | test | easy | Store PassengerId as text. |
| `titanic_conv_005` | test | medium | Make Embarked categorical; missing stays missing. |
| `titanic_conv_006` | dev | easy | Store SibSp and Parch as floats. |
| `titanic_col_001` | test | easy | Add num_relatives = SibSp + Parch. |
| `titanic_col_002` | test | medium | Strip leading/trailing whitespace from Name. |
| `titanic_col_003` | test | medium | Round Fare to 2 decimal places. |
| `titanic_col_004` | test | medium | Recode Sex: male -> M, female -> F. |
| `titanic_col_005` | test | hard | Cap Fare at 200 and raise ages below 1 to 1. |
| `titanic_col_006` | dev | medium | Rename Pclass to passenger_class. |
| `taxi_filt_001` | test | easy | Remove trips with a negative fare (refunds). |
| `taxi_filt_002` | test | easy | Remove trips with zero distance. |
| `taxi_filt_003` | test | medium | Remove trips with RatecodeID 99 (unknown). |
| `taxi_filt_004` | test | hard | Remove trips picked up before 1 January 2026. |
| `taxi_filt_005` | test | medium | Remove credit-card trips with no tip. |
| `taxi_filt_006` | dev | hard | Keep only trips with 1 to 4 passengers. |
| `taxi_dedup_001` | test | medium | Same vendor, pickup and dropoff time are duplicates; keep the first. |
| `taxi_dedup_002` | test | hard | Same key as dedup_001, keep the last. |
| `taxi_dedup_003` | test | hard | Rows matching in every non-money column are duplicates; keep the first. |
| `taxi_dedup_004` | test | medium | Same pickup time and pickup zone are duplicates; keep the first. |
| `taxi_dedup_005` | test | medium | Keep the first trip for each pickup/dropoff zone pair. |
| `taxi_dedup_006` | dev | hard | Same vendor and pickup time are duplicates; keep the last. |
| `taxi_miss_001` | test | medium | passenger_count 0 means not recorded; set it to missing. |
| `taxi_miss_002` | test | easy | RatecodeID 99 means unknown; set it to missing. |
| `taxi_miss_003` | test | easy | Replace passenger_count 0 with 1. |
| `taxi_miss_004` | test | medium | Replace RatecodeID 99 with 1 (standard rate). |
| `taxi_miss_005` | test | hard | Mark both sentinels as missing: passenger_count 0 and RatecodeID 99. |
| `taxi_miss_006` | dev | medium | trip_distance 0 means not recorded; set it to missing. |
| `taxi_conv_001` | test | medium | Convert the pickup time to a datetime column. |
| `taxi_conv_002` | test | hard | Convert both pickup and dropoff times to datetime. |
| `taxi_conv_003` | test | easy | Store passenger_count as an integer. |
| `taxi_conv_004` | test | easy | Make VendorID a categorical column. |
| `taxi_conv_005` | test | hard | Convert store_and_fwd_flag (Y/N) to boolean. |
| `taxi_conv_006` | dev | easy | Store RatecodeID as an integer. |
| `taxi_col_001` | test | easy | Add fare_plus_tip = fare_amount + tip_amount. |
| `taxi_col_002` | test | medium | Replace payment_type codes with names. |
| `taxi_col_003` | test | medium | Cap trip_distance at 30 miles. |
| `taxi_col_004` | test | easy | Round trip_distance to 1 decimal place. |
| `taxi_col_005` | test | hard | Add fare_per_mile; zero-distance trips get missing. |
| `taxi_col_006` | dev | medium | Rename Airport_fee to airport_fee. |
