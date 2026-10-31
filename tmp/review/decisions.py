import csv
import json
from pathlib import Path

notes = {
1: "Removal of traveltime matches the request. Remaining columns and all rows are preserved.",
2: "Sales is capped at 350 with no lower bound. Other columns and row count are preserved.",
3: "O3 is floored at 24. Values above the floor and nulls remain unchanged.",
4: "The vocabulary's map function preserves unlisted values, matching replace and the request to change only VVS2.",
5: "Consumer price index correctly resolves to CPI. Only that column is removed.",
6: "Unemployment is divided by 100 in place. Nulls and unrelated columns are preserved.",
7: "Rename is captured by addition, removal, and an identity relation preserving the old values.",
8: "The requested division overwrites No_of_Votes rather than adding a column.",
9: "The explicitly requested multiplication and new column name match both contract and code. The derived feature is synthetic but unambiguous.",
10: "Fedu/G1 and the added column match the request. G1 has no zero denominators in this snapshot. The vocabulary defines division by zero as null on other inputs.",
11: "The explicitly requested subtraction z-price matches the new column and code. It is an arbitrary feature, but the intended arithmetic is unambiguous.",
12: "Temperature is floored at 42 without imposing an upper bound.",
13: "A new dataset_version column is set to v2 on every row. Original columns and rows are preserved.",
14: "A new currency column is set to USD on every row. Original columns and rows are preserved.",
15: "The request clearly adds a currency tag. The contract sets USD and preserves the original data.",
16: "Total Bags is rounded to one decimal place. Missing values remain missing and other columns are preserved.",
17: "Uppercase conversion applies only to cut, matching the request and reference.",
18: "The rename age to age_value preserves values through an identity relation and preserves other columns and rows.",
19: "Metascore resolves to Meta_score. The contract adds meta_score_x2 and doubles the original values without overwriting them.",
20: "Small Bags is floored at 400 with no upper cap and no changes to other columns.",
22: "All columns except Humidity and RespiratoryCases form the key. keep:first enforces one original first row per group.",
24: "The three requested key columns match. Earliest means first in original row order, as specified by the schema, rather than earliest date.",
25: "Medu and failures are excluded only from the deduplication key. keep:first retains one first row per group with original values.",
27: "Only y is excluded from the key. keep:last enforces the last original row for each group.",
28: "Contract and reference agree, but 'Each combination of number of votes' is awkward for a single key. Rephrase: 'Keep only the last movie for each distinct number of votes.'",
29: "WindSpeed is excluded from the key and keep:first enforces the requested survivor for every group.",
31: "depth and x are excluded from the key. keep:last matches the stated survivor rule.",
32: "loan, duration and pdays form the requested key. keep:first preserves one original first row per group.",
34: "address is excluded from the key. The first row of every group is retained and surviving values stay unchanged.",
36: "Total Volume, 4046 and 4770 are literal column names forming the key. keep:first matches the request.",
37: "Runtime, Star2 and Star4 form the key, with the last original row retained for every group.",
38: "color and depth are excluded only from the key. keep:first and preservation of all surviving values match the request.",
39: "table is excluded from the key. keep:last enforces one last original row per group.",
41: "The removal condition is clarity=VS2 AND x<5.7, with the complementary rows retained. Both condition columns have no nulls in this snapshot.",
42: "Strict AveragePrice<1.1 removal keeps the boundary 1.1. AveragePrice has no missing cells in this snapshot.",
43: "Contract and code correctly remove HealthImpactClass>=0. All 989 rows match, so the result is empty and the retention check is vacuous. Prefer a threshold leaving survivors for more informative training coverage.",
44: "Both organic type and 4046<2300 are required for removal. Other types and the threshold boundary are retained.",
45: "Removal requires romantic=yes AND internet=no. The complementary rows and all surviving column values are preserved.",
46: "RespiratoryCases<7 is removed and 7 is retained. The condition column has no nulls in this snapshot.",
47: "The retained sales range includes both 610000 and 1600000. Weekly_Sales has no nulls in this snapshot.",
48: "OR removal and its AND complement are correct. Votes=400000 and rating=7.7 remain eligible.",
49: "Only Good, Premium and Very Good cut values survive. cut has no nulls in this snapshot.",
50: "The OR condition and complementary retention correctly implement votes>600000 or rating<7.7.",
51: "'No more than 370' means <=370. The contract removes that boundary and retains values strictly above it.",
52: "'At most 0.34' means <=0.34. The code and preservation predicate use the correct strict complement.",
53: "The NOT equality condition explicitly retains missing color values. The object column's NaN values are also retained by the reference's != comparison.",
54: "Only clarity SI2 or VS1 rows are removed. clarity has no nulls in this snapshot, so not_in is a complete complement here.",
55: "Only month apr or feb rows are removed. month has no nulls in this snapshot.",
56: "The contract and code correctly retain missing Sales. The phrase 'Orders without a sales stay in' is unnatural. Rephrase: 'Keep orders with missing sales values.'",
57: "Removal requires both conventional type and AveragePrice<1.3. The complementary rows are retained.",
58: "The 4046 range is inclusive at 80 and 400000. The condition column has no nulls in this snapshot.",
59: "The retained vote range includes 51000 and 730000, matching 'below' and 'above' removal.",
60: "'And also the ones' describes two removal groups, so OR is correct. Store<=41 AND Temperature>=35 is the retention condition.",
61: "Only missing Meta_score values become -1. Nonmissing scores, other columns and all rows are preserved.",
62: "A missing value in either NO2 or WindSpeed removes the row. Rows complete in both columns are retained.",
63: "The snapshot median duration is exactly 182.5. A precomputed const literal is explicitly allowed by contract_schema sections 6 and 12, so this label is correct for this snapshot. Parser input needs that statistic to infer the literal reliably.",
64: "Missing Gross values receive the literal text Not available. Existing values and unrelated data are preserved.",
65: "Only null job values become Other. Existing job labels, including any nonnull unknown sentinel, remain unchanged.",
66: "Only missing Sales values become 1.5. Existing sales and unrelated columns are preserved.",
67: "A null in any of PM10, SO2 or WindSpeed removes the row. All three must be present to retain it.",
68: "Only missing Genre values become Other. Nonmissing genres remain unchanged.",
69: "Rows missing duration are removed. Every row with a recorded duration survives unchanged.",
70: "Only missing Unemployment values become 0. Recorded rates remain unchanged.",
71: "The lowercase text none is a literal string, not Python None or a null. Contract and reference agree.",
72: "Large Bags must be nonnull to retain the row. Zero counts as a recorded value and is retained.",
73: "Both Sales and Segment must be present. The contract removes rows missing either and preserves complete rows.",
74: "Any missing Certificate, Director, Gross or Meta_score removes a row. All four are required for retention.",
75: "All of Director, Gross and Meta_score must be nonnull. Other columns need not be filled.",
76: "Only null job values receive Not available. Existing labels, other columns and all rows are preserved.",
77: "'Table width' resolves to the dataset's table column. Only missing table rows are removed.",
78: "The OR applies to missingness in Humidity, NO2 and PM10. Rows with all three recorded are retained.",
79: "Missingness in any of Certificate, Director or Genre removes the row. All other complete rows survive.",
80: "Rows missing either Sales or Segment are removed. Both values must be present for retention.",
81: "Only school's dtype changes to category. Its values and all other columns and rows are preserved.",
82: "age, day and previous become float, preserving numeric values. 'Contact day' correctly resolves to day.",
83: "Mjob, activities and higher become category without changing values. Missing Mjob values remain missing.",
84: "job, contact and month become category. Existing values and job missingness are preserved.",
85: "age and day become text via cast-to-string relations. Both columns have no nulls in this snapshot.",
86: "Personal loan flag resolves to loan. loan and month become category without value changes.",
87: "education, loan and month are all converted to category, preserving values and unrelated columns.",
88: "age, famrel and Walc become float. Their numeric values, other columns and row count are preserved.",
89: "housing and contact become category. The requested labels resolve to the correct columns.",
90: "HospitalAdmissions becomes string with its numeric representation preserved by a cast relation. The column has no nulls in this snapshot.",
91: "famsize, schoolsup and higher become category while values remain unchanged, as explicitly requested.",
92: "studytime and freetime become text using cast relations. Both have no missing cells in this snapshot.",
93: "job, default and month become category. The values and job null count remain unchanged.",
94: "G1 becomes float while grades retain their numeric values.",
95: "education becomes category without changing its labels or unrelated data.",
96: "Housing loan flag correctly resolves to housing, which becomes category without value changes.",
97: "Father's education resolves to Fedu and weekend alcohol use to Walc. Both become text, with no missing cells in this snapshot.",
98: "age, Medu and G1 become float. First-period grade correctly resolves to G1 and all numeric values are preserved.",
99: "day becomes float. The numeric day values and other columns remain unchanged.",
100: "Ship Mode, Segment and Category become category. Missing Segment values remain missing and unrelated data is preserved.",
}

evidence = json.loads(Path('tmp/review/check_results.json').read_text())
rows = list(csv.DictReader(open('tmp/review/review.original.csv')))
decisions = []
for i, (row, result) in enumerate(zip(rows, evidence), 1):
    assert result['id'] == row['id']
    assert result['reference_verdict'] == 'PASS'
    assert result['noop_verdict'] == 'VIOLATION'
    verdict = 'OK'
    if result.get('empty_output_verdict') == 'PASS':
        verdict = 'wrong_contract'
        note = ('Incomplete preservation: duplicate_count eq 0 without keep checks uniqueness only. '
                'Both an empty output and a one-row output pass this contract on the actual snapshot, '
                'although the request requires one survivor per key group. Enforce group coverage '
                'without inventing a first/last policy, or explicitly request a first/last rule and regenerate. '
                'Reference code itself is correct.')
    else:
        note = notes[i]
        if i in (28, 56):
            verdict = 'unnatural_request'
    decisions.append({'id': row['id'], 'verdict': verdict, 'note': note})
assert len(decisions) == 100
Path('tmp/review/decisions.json').write_text(json.dumps(decisions, indent=2))
