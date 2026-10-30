# Fine-tuning datasets (not used in evaluation)

Built by `finetune/data_gen/prepare_datasets.py` from the raw files in `data/finetune/` (fixed seed).
Validation-only datasets (never in training): student.

## air_quality

- source: Kaggle: Air Quality and Health Impact Dataset (synthetic); raw file `data/finetune/air_quality_health_impact_data.csv`; licence: CC BY 4.0 (per Kaggle page)
- rows 989, columns 14: AQI, PM10, PM2_5, NO2, SO2, O3, Temperature, Humidity, WindSpeed, RespiratoryCases, CardiovascularCases, HospitalAdmissions, HealthImpactScore, HealthImpactClass
- missing values: {'PM10': 48, 'NO2': 49, 'SO2': 51, 'Humidity': 48, 'WindSpeed': 49}
- duplicate rows: 19
- injected: floats rounded to 2 decimals; 48 missing values added to `PM10`; 48 missing values added to `NO2`; 48 missing values added to `Humidity`; 48 missing values added to `SO2`; 48 missing values added to `WindSpeed`; 29 values of `O3` set to the code -999.0 (meaning: not recorded); 19 values of `AQI` set to the code -1.0 (meaning: not recorded); 19 exact duplicate rows inserted after their originals
- sha256: `4f945fa30c62121411237d7bab76dc83cb4909884a3648b0fb760956506da5a4`

## avocado

- source: Kaggle: Avocado Prices (Hass Avocado Board); raw file `data/finetune/avocado.csv`; licence: ODbL 1.0 (per Kaggle page)
- rows 989, columns 13: Date, AveragePrice, Total Volume, 4046, 4225, 4770, Total Bags, Small Bags, Large Bags, XLarge Bags, type, year, region
- missing values: {'Total Bags': 48, 'Large Bags': 51, 'XLarge Bags': 50, 'region': 49}
- duplicate rows: 19
- injected: 48 values of `region` given leading/trailing spaces; 48 missing values added to `Large Bags`; 48 missing values added to `XLarge Bags`; 48 missing values added to `Total Bags`; 48 missing values added to `region`; 19 values of `AveragePrice` set to the code 0.0 (meaning: not recorded); 19 exact duplicate rows inserted after their originals
- sha256: `7f2e5b9e497b6f609dfe25c8a5867fb628562a50e72994acf45ac9c5ca0def5d`

## bank

- source: UCI Bank Marketing (bank-full.csv); raw file `data/finetune/bank-full.csv`; licence: CC BY 4.0
- rows 989, columns 17: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, y
- missing values: {'job': 49, 'marital': 48, 'balance': 49, 'duration': 49}
- duplicate rows: 19
- injected: 48 missing values added to `job`; 48 missing values added to `balance`; 48 missing values added to `marital`; 48 missing values added to `duration`; 19 exact duplicate rows inserted after their originals
- sha256: `4bfc4cf405e19845006e3cf576666346113fda9e962067966dddedea3fb4ae48`

## diamonds

- source: ggplot2 diamonds; raw file `data/finetune/diamonds.csv`; licence: MIT (ggplot2)
- rows 989, columns 10: carat, cut, color, clarity, depth, table, price, x, y, z
- missing values: {'color': 49, 'depth': 49, 'table': 48, 'price': 49}
- duplicate rows: 19
- injected: 48 missing values added to `depth`; 48 missing values added to `table`; 48 missing values added to `color`; 48 missing values added to `price`; 19 values of `x` set to the code 0.0 (meaning: not recorded); 19 values of `y` set to the code 0.0 (meaning: not recorded); 19 exact duplicate rows inserted after their originals
- sha256: `5c0036b0869141dd0c08e2e50b3fed98523528aa58903de23877f66f1c39a5b2`

## imdb

- source: Kaggle: IMDB Movies Dataset (top 1000); raw file `data/finetune/imdb_top_1000.csv`; licence: CC0 (per Kaggle page)
- rows 1020, columns 14: Series_Title, Released_Year, Certificate, Runtime, Genre, IMDB_Rating, Meta_score, Director, Star1, Star2, Star3, Star4, No_of_Votes, Gross
- missing values: {'Certificate': 102, 'Genre': 50, 'Meta_score': 160, 'Director': 50, 'Gross': 171}
- duplicate rows: 20
- injected: 50 values of `Series_Title` given leading/trailing spaces; 50 missing values added to `Director`; 50 missing values added to `Genre`; 20 exact duplicate rows inserted after their originals
- sha256: `c093e60318dc0b6f70ab5a464bf9a83d73c9e91a28d6b0814b2083033f86f8aa`

## student (validation)

- source: UCI Student Performance (student-mat.csv); raw file `data/finetune/student-mat.csv`; licence: CC BY 4.0
- rows 402, columns 33: school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, G3
- missing values: {'Mjob': 20, 'guardian': 19, 'absences': 19}
- duplicate rows: 7
- injected: 19 missing values added to `absences`; 19 missing values added to `Mjob`; 19 missing values added to `guardian`; 7 exact duplicate rows inserted after their originals
- sha256: `d1e424412874fe6baba4f323fafaba7b61568dce074fde926257fe6fd40dc170`

## superstore

- source: Kaggle: Superstore Sales Dataset (train.csv); raw file `data/finetune/train.csv`; licence: CC BY-SA 4.0 (per Kaggle page)
- rows 989, columns 16: Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name, Segment, City, State, Postal Code, Region, Product ID, Category, Sub-Category, Product Name, Sales
- missing values: {'Segment': 50, 'City': 49, 'Postal Code': 2, 'Sales': 48}
- duplicate rows: 19
- injected: 48 values of `Customer Name` given leading/trailing spaces; 48 missing values added to `Sales`; 48 missing values added to `Segment`; 48 missing values added to `City`; 19 values of `Sales` set to the code 0.0 (meaning: not recorded); 19 exact duplicate rows inserted after their originals
- sha256: `5c9cea3e03bb327b795606b2cadac327dc3162be6257b6e02778445d9b3fa086`

## walmart

- source: Kaggle: Walmart Sales; raw file `data/finetune/Walmart.csv`; licence: CC0 (per Kaggle page)
- rows 989, columns 8: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
- missing values: {'Fuel_Price': 48, 'CPI': 49, 'Unemployment': 49}
- duplicate rows: 19
- injected: 48 missing values added to `CPI`; 48 missing values added to `Unemployment`; 48 missing values added to `Fuel_Price`; 19 values of `Temperature` set to the code -99.0 (meaning: not recorded); 19 values of `Weekly_Sales` set to the code 0.0 (meaning: not recorded); 19 exact duplicate rows inserted after their originals
- sha256: `eb4a364e1d6b832a4eb84058a44147ac26583df7bb7e791037c5b2478358403f`
