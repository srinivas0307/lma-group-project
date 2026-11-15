# 3B incorrect-acceptance and withheld-case inventory

All 54 unsuccessful method/task outcomes are covered below: 27 per method, across 30 distinct tasks. An omitted method succeeded on that task. Reference compatibility is audit information only and must not be used to route ordinary inference.

## adult_col_005

Add a column is_married: 'Married' for Married-civ-spouse, Married-spouse-absent and Married-AF-spouse, and 'Not married' for everyone else.

Map encodes an "Other" key as if it were a default. The contract does not represent the requested fallback for every unmatched marital-status value.

- selective: ESCALATE; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.

## adult_conv_003

Make education an ordered category that follows the education-num order, from Preschool (lowest) to Doctorate (highest).

Selective: invalid JSON contains an ellipsis in the category list. Tools: invented education labels and a mismatched mapping from education-num. Needs input-derived category/rank pairs and explicit ordered-category semantics.

- selective: ESCALATE; category=parser_stop; initially correct=None; reference=None; repeated code=False.
- tools: REJECT; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.

## adult_dedup_003

fnlwgt should identify a record, so remove repeated fnlwgt values and keep the first record for each.

Usable contract. Initial merge raises KeyError; repair inspects nonexistent last_output and keeps complicating the merge/helper-column approach. A fresh formulation should preserve original records and deduplicate on fnlwgt.

- selective: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=False.
- tools: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=False.

## adult_filt_004

Remove the people whose education level is Preschool or 1st-4th.

Both contracts compare the categorical education strings with numeric thresholds; verification is uncertain. The request names two string categories.

- selective: ESCALATE; category=contract_reference_mismatch; initially correct=False; reference=UNCERTAIN; repeated code=False.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=False; reference=UNCERTAIN; repeated code=False.

## adult_filt_005

Remove self-employed people (incorporated or not) who are older than 60.

Selective binds self-employment to education=Private instead of workclass categories. Tools emits a predicate name as an unavailable tool and never produces a usable contract.

- selective: REJECT; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.
- tools: ESCALATE; category=parser_stop; initially correct=None; reference=None; repeated code=False.

## adult_miss_005

People with no workclass and no occupation listed are not working. Fill both missing workclass and missing occupation with 'Not-working'.

Initially correct code is withheld. Filling null cells conflicts with preserving null count; tools additionally preserves the edited cell values. Resolve the contract, not the code alone.

- selective: REJECT; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

## airbnb_conv_005

Make room_type an ordered category, from smallest to largest: Shared room, then Private room, then Entire home/apt.

Usable contract. set_categories result is discarded; ordered=False remains. The repair agent repeats the same code despite an explicit ordered-flag failure.

- selective: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## airbnb_dedup_001

Some listings share exactly the same name. Treat those as duplicates and keep only the first one.

Initially correct code is withheld because the contract deduplicates all columns instead of listing name.

- selective: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

## airbnb_dedup_002

Remove listings where the same host has posted the same listing name more than once; keep the first copy.

Initially correct code is withheld; contract uses nonexistent listing_name and the wrong host identity field. Need exact input schema and request-to-key binding.

- selective: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=UNCERTAIN; repeated code=False.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=UNCERTAIN; repeated code=False.

## airbnb_dedup_003

Keep only one listing per host: the first one that appears in the data.

Selective withholds initially correct code because the contract uses all columns rather than host_id. Tools succeeds.

- selective: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

## airbnb_dedup_004

For each host, keep just one listing per room type, using the last one that appears.

Selective omits host_id from the composite key. Tools has a reference-compatible contract but filters to groups of size one before keeping last, deleting all repeated groups. Adding a sort does not restore the lost groups.

- selective: ESCALATE; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.
- tools: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## airbnb_filt_002

I only want to keep listings located in Brooklyn and Manhattan.

Both incorrectly accept an empty output. Contract selects neighbourhood instead of the borough-valued neighbourhood_group; correct reference has 927 rows. Relevant categorical-value evidence is missing.

- selective: ACCEPT; category=incorrect_acceptance; initially correct=False; reference=VIOLATION; repeated code=False.
- tools: ACCEPT; category=incorrect_acceptance; initially correct=False; reference=VIOLATION; repeated code=False.

## taxi_col_004

Round trip_distance to 1 decimal place.

Usable contract under the current NumPy rounding convention. Scalar Python round(x,1) differs from pandas/NumPy round on 42 input rows; retries keep using scalar round.

- selective: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## taxi_col_005

Add a column fare_per_mile equal to fare_amount divided by trip_distance. For trips with a distance of 0 the value should be missing.

Selective contract attempts to preserve values of a newly added column that does not exist in the input. Tools succeeds. Zero-distance behavior must also remain represented.

- selective: REJECT; category=contract_reference_mismatch; initially correct=False; reference=UNCERTAIN; repeated code=False.

## taxi_dedup_005

Keep only the first trip for each combination of pickup zone (PULocationID) and dropoff zone (DOLocationID).

Usable contract. Sorting by pickup timestamp before deduplication changes which original row is first; 99 key groups retain the wrong record. Repeated proposals preserve this error.

- selective: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## taxi_filt_004

Remove the trips that were picked up before 1 January 2026. The pickup and dropoff times are text in the form YYYY-MM-DD HH:MM:SS (for example 2026-01-01 00:54:04).

Both incorrectly accept the result with an extra pickup_datetime helper column. Existing columns_unchanged does not forbid extra output columns; both gold and predicted contracts pass the wrong output.

- selective: ACCEPT; category=incorrect_acceptance; initially correct=False; reference=PASS; repeated code=False.
- tools: ACCEPT; category=incorrect_acceptance; initially correct=False; reference=PASS; repeated code=False.

## taxi_miss_005

In this file a passenger_count of 0 and a RatecodeID of 99 both mean 'not recorded'. Mark those values as missing.

Selective router mistakes ordinary "mean" (signify) for a statistic and stops before parsing. Tools succeeds.

- selective: ESCALATE; category=parser_stop; initially correct=None; reference=None; repeated code=False.

## titanic_col_001

Add a column num_relatives with the number of relatives aboard: SibSp plus Parch.

Usable contract. Code adds num_relatives but removes SibSp/Parch (and sometimes more columns); repeated repair fails to preserve inputs.

- selective: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## titanic_col_003

Round Fare to 2 decimal places.

Usable contract under current NumPy rounding convention. Scalar Python round(x,2) differs from pandas/NumPy round on 29 rows; repeated proposals preserve the discrepancy.

- selective: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## titanic_col_005

Cap Fare at 200 (higher fares become 200) and raise every Age below 1 to 1. Missing ages stay missing.

Selective encodes a threshold with a literal map key "<1"; tools multiplies Age by one instead of setting ages below one to one. Tools initially has correct code and loses it.

- selective: ESCALATE; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

## titanic_conv_002

Make Pclass an ordered categorical column where third class (3) is the lowest and first class (1) is the highest.

Selective emits invalid map arguments. Tools emits the wrong category order [1,3,2] and a spurious remapping. Request needs category order [3,2,1] with original values retained.

- selective: ESCALATE; category=parser_stop; initially correct=None; reference=None; repeated code=False.
- tools: REJECT; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.

## titanic_conv_003

Convert Survived to a boolean column: True for 1 and False for 0.

Both withhold initially correct code. Contract mapping uses string keys True/False rather than the input values 1/0 under current typed mapping semantics.

- selective: REJECT; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.
- tools: REJECT; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

## titanic_dedup_002

Passengers who share the same ticket number are duplicates. Keep only the last passenger listed for each ticket.

Usable contract. Sorting Ticket before keeping last changes original record order/tie selection; repair repeats the initial program.

- selective: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: ESCALATE; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## titanic_dedup_004

Keep only one passenger for each combination of class and sex: the first one listed.

Usable contract. Invalid groupby keep argument is followed by groupby.first(), which can aggregate first nonmissing values instead of retaining one complete original record.

- selective: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## titanic_filt_003

Remove passengers younger than 18. Passengers whose age is unknown should stay.

Usable contract. Code drops unknown ages although request preserves them; the final failure report identifies 177 missing rows that should stay.

- selective: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.
- tools: REJECT; category=repair_with_reference_compatible_contract; initially correct=False; reference=PASS; repeated code=True.

## titanic_filt_005

Keep only the passengers who travelled alone, with no siblings, spouses, parents or children aboard.

Selective emits invalid null operator; both contracts invent Cabin/Fare conditions unrelated to travelling alone. Tools initially has correct code but is withheld.

- selective: ESCALATE; category=parser_stop; initially correct=None; reference=None; repeated code=False.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=UNCERTAIN; repeated code=False.

## titanic_miss_005

Fill each missing Age with the median age of passengers in the same class (Pclass).

Selective cannot compute grouped statistics in its current facts router. Tools queries Pclass rather than Age grouped by Pclass and fills with 3. Existing group_profile can retrieve the required values but was not selected.

- selective: ESCALATE; category=parser_stop; initially correct=None; reference=None; repeated code=False.
- tools: ESCALATE; category=contract_reference_mismatch; initially correct=False; reference=VIOLATION; repeated code=True.

## taxi_conv_002

Convert both tpep_pickup_datetime and tpep_dropoff_datetime into proper datetime columns. The pickup and dropoff times are text in the form YYYY-MM-DD HH:MM:SS (for example 2026-01-01 00:54:04).

Tools withholds initially correct code: it requires timestamp conversion while preserving the original string values under the current verifier semantics. Selective succeeds.

- tools: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

## taxi_miss_003

A passenger_count of 0 means it was not recorded. Replace those values with 1.

Tools withholds initially correct code: replacing zero with one conflicts with a required increase in null count while the remaining cells are preserved. Selective succeeds.

- tools: ESCALATE; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

## taxi_miss_004

RatecodeID 99 means the rate code is unknown. Assume the standard rate and replace those values with 1.

Tools withholds initially correct code: replacing 99 with one conflicts with a required increase in null count. Selective succeeds.

- tools: REJECT; category=contract_reference_mismatch; initially correct=True; reference=VIOLATION; repeated code=True.

