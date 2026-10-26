# Heart Disease Prediction using Machine Learning

A project to prepare a medical dataset that is ready for Machine Learning, in order to predict the presence or absence of heart disease in a patient based on their clinical characteristics.

---

## 1. Context and business problem

Cardiovascular diseases are among the leading causes of death worldwide. Hospitals and healthcare professionals need to quickly identify patients at high risk of developing heart disease so that appropriate preventive measures can be put in place.

Assessing this risk manually can, however, be complex because of the large number of medical factors to consider at the same time. The organisation therefore wants to make use of patients' medical data to improve the early identification of people at risk.

## 2. Project objectives

**Main objective:** develop a dataset that is ready for Machine Learning in order to predict the presence (`1`) or absence (`0`) of heart disease in a patient.

This project covers the **data preparation** part: exploration, cleaning, feature engineering, selection, train/test splitting, scaling and export. The model training phase is not included in the notebook.

## 3. Data description

The expected source file is `heart.csv` (918 rows, 12 columns on import). Each row corresponds to one patient.

| Variable | Description | Values |
|---|---|---|
| **Age** | Patient's age | in years |
| **Sex** | Patient's sex | `M`: male, `F`: female |
| **ChestPainType** | Type of chest pain | `TA`: typical angina, `ATA`: atypical angina, `NAP`: non-anginal pain, `ASY`: asymptomatic |
| **RestingBP** | Resting blood pressure | in mm Hg |
| **Cholesterol** | Serum cholesterol | in mg/dl |
| **FastingBS** | Fasting blood sugar | `1` if > 120 mg/dl, `0` otherwise |
| **RestingECG** | Resting electrocardiogram result | `Normal`, `ST` (ST-T wave abnormality), `LVH` (probable or definite left ventricular hypertrophy according to Estes criteria) |
| **MaxHR** | Maximum heart rate achieved | numerical value between 60 and 202 |
| **ExerciseAngina** | Exercise-induced angina | `Y`: yes, `N`: no |
| **Oldpeak** | ST segment depression (ST = ST depression) | numerical value |
| **ST_Slope** | Slope of the ST segment at peak exercise | `Up`: upsloping, `Flat`: flat, `Down`: downsloping |
| **HeartDisease** | Target variable | `1`: heart disease, `0`: normal |

On loading, the dataset has 6 integer variables, 1 decimal variable (`Oldpeak`) and 5 text variables. No missing values and no duplicates were detected.

## 4. Environment and dependencies

The notebook was run with **Python 3.13.5**.

Libraries used:

- `pandas` and `numpy`: data manipulation
- `matplotlib` and `seaborn`: visualisations
- `scipy`: statistical tests (chi-square)
- `scikit-learn`: train/test split and standardisation

Installation:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

Note: the notebook also imports `tables.tests.test_suite` (PyTables), which is not used. See the [Limitations](#9-limitations-and-points-for-improvement) section.

## 5. Project structure

```
.
├── Heart_Failure_ML.ipynb   # Main notebook (exploration, cleaning, preparation)
├── heart.csv                  # Source data (to be placed at the project root)
├──cleaned_dataset.csv     # Cleaned data, without encoding or scaling
├── Machine_Learning_ready_DatasetL.csv        # Final dataset ready for Machine Learning
└── README.md                  # This file
└── Data_Preprocessing_Report_Week2 
└── Business_Understanding_Report
```

## 6. Project steps

### Step 0: Importing libraries and the dataset

The visualisation and computation libraries are imported, then the file `heart.csv` is loaded into a DataFrame `df`. The first five rows are displayed for a first visual check.

### Step 1: Checking the data structure

- **Dimensions**: 918 rows and 12 columns.
- **Types**: `int64` for 6 columns, `float64` for `Oldpeak`, `object` for the 5 categorical columns.
- **Missing values**: none, for all columns.
- **Duplicates**: none.

### Step 2: Data cleaning

The notebook checks again for missing values before moving on to feature engineering. No imputation is necessary.

### Step 3: Feature engineering

Binary variables are converted into readable categories to make exploratory analysis and interpretation easier:

- `HeartDisease`: converted to integer (`int`), with `0` and `1` kept.
- `FastingBS`: `0` → `No`, `1` → `Yes`.

This step makes charts and tables more meaningful without changing the meaning of the values.

### Step 4: Detecting and removing outliers

The **interquartile range (IQR)** method is applied to all numerical variables:

- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

The thresholds are computed on the original DataFrame and kept, then the rows containing at least one value outside the bounds are removed.

| Variable | Lower threshold | Upper threshold | Outliers |
|---|---|---|---|
| Age | 27.5 | 79.5 | 0 |
| RestingBP | 90 | 170 | 28 |
| Cholesterol | 32.625 | 407.625 | 183 |
| MaxHR | 66 | 210 | 2 |
| Oldpeak | −2.25 | 3.75 | 16 |
| HeartDisease | −1.5 | 2.5 | 0 |

**Result:** 918 rows before, 702 rows after, i.e. **216 rows removed** (about 23.5% of the data). A check confirms that no value exceeds the original thresholds any more.

### Step 5: Descriptive statistical analysis

A statistical summary (`describe`) is produced for the numerical and categorical variables after cleaning.

### Step 6: Distribution visualisation

- **Boxplots** of each numerical variable by `HeartDisease` (outliers hidden in the display).
- **Bar charts** of the categorical variables, with percentages shown on each bar.
- **Histograms** of the numerical variables.

### Step 7: Bivariate analysis

- **Categorical variables vs target**: grouped bar charts, showing the proportion of sick patients in each category.
- **Pairs of numerical variables**: scatter plots coloured by `HeartDisease`, to observe how well the classes can be separated.

### Step 8: Intermediate save

The cleaned DataFrame is exported to `HeartDisease_clean.csv` (without index), then reloaded for verification. This file contains readable values, before encoding.

### Step 9: Encoding categorical variables

Categorical variables are transformed into indicator variables with `pd.get_dummies`, dropping one reference category per variable (`drop_first=True`) to avoid perfect multicollinearity.

Encoded variables: `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`, `FastingBS`.

The target `y` is separated from the features `X`.

### Step 10: Variable selection

The relevance of the variables is assessed using two complementary approaches.

**a) Correlation with the target.** A correlation matrix (features + target, categorical variables encoded as 0/1) is plotted as a heatmap. For each original variable, the notebook keeps the maximum absolute correlation with `HeartDisease`, taking the maximum across its indicator variables.

| Variable | Maximum absolute correlation |
|---|---|
| ST_Slope (Up) | 0.65 |
| ST_Slope (Flat) | 0.60 |
| ExerciseAngina (Y) | 0.55 |
| Oldpeak | 0.50 |
| MaxHR | 0.40 |
| ChestPainType (ATA) | 0.38 |
| Sex (M) | 0.31 |
| Age | 0.31 |
| ChestPainType (NAP) | 0.21 |
| RestingBP | 0.17 |
| FastingBS (Yes) | 0.16 |
| RestingECG (Normal) | 0.13 |
| Cholesterol | 0.10 |
| RestingECG (ST) | 0.09 |
| ChestPainType (TA) | 0.05 |

Variables are split into two groups according to the 0.5 threshold: those above it (`ST_Slope`, `ExerciseAngina`, `Oldpeak`) and those below it.

**b) Statistical association tests with the target.**

- **Chi-square** for binary variables (2 categories):

| Variable | Chi-square | Degrees of freedom | p-value |
|---|---|---|---|
| ExerciseAngina | 210.33 | 1 | 1.16e-47 |
| Sex | 64.52 | 1 | 9.55e-16 |
| FastingBS | 18.09 | 1 | 2.11e-05 |

- **Tschuprow's T** for variables with more than 2 categories:

| Variable | Categories | Tschuprow's T | p-value |
|---|---|---|---|
| ST_Slope | 3 | 0.546 | 5.35e-65 |
| ChestPainType | 4 | 0.410 | 4.71e-44 |
| RestingECG | 3 | 0.112 | 2.08e-03 |

All tested variables are significantly associated with the target at the 5% level. The notebook keeps all variables for the rest of the processing.

### Step 11: Train / test split

The dataset is split into training and test sets with the following parameters:

- `test_size = 0.2` (80% training, 20% test)
- `random_state = 42` (reproducibility)
- `stratify = y` (keeps the proportion of sick patients in both subsets)

Expected result: about 561 rows in training and 141 in test.

### Step 12: Standardisation

The continuous numerical variables are standardised with `StandardScaler` (mean 0, standard deviation 1):

`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`

The scaler is **fitted only on the training set** (`fit_transform`), then applied to the test set (`transform`). This prevents any information leakage from the test data. Binary variables resulting from the encoding are not standardised.

### Step 13: Building the final dataset

The standardised sets are combined with their target and a `Jeu` column indicating membership (`train` or `test`). The result is sorted by the original index and exported to `HeartDisease_ML.csv`.

## 7. Output files

| File | Content | Rows | Columns |
|---|---|---|---|
| `cleaned_dataset.csv` | Cleaned data, readable values, no encoding | 702 | 12 |
| `Machine_Learning_ready_Dataset.csv` | Encoded and standardised data, with target and `Jeu` column | 702 | 17 |

Columns of `Machine_Learning_ready_Dataset.csv`:

- **Standardised numerical features**: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`
- **Binary features**: `Sex_M`, `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`, `RestingECG_Normal`, `RestingECG_ST`, `ExerciseAngina_Y`, `ST_Slope_Flat`, `ST_Slope_Up`, `FastingBS_Yes`
- **Target**: `HeartDisease` (0 or 1)
- **Split**: `Jeu` (`train` or `test`)

## 8. Key results

- The final dataset contains **702 patients** after removing outliers.
- The variables most related to the disease are the **slope of the ST segment** (`ST_Slope`), **exercise-induced angina** (`ExerciseAngina`), and **ST depression** (`Oldpeak`).
- Sex, age, maximum heart rate and type of chest pain also show significant associations with the target.
- The dataset is ready to be used directly to train a binary classification model.

## 9. Usage

1. Place the file `heart.csv` at the project root.
2. Install the dependencies (see section 4).
3. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook HeartDiseaseFor_ML.ipynb
   ```

4. Run all cells in order (menu *Kernel → Restart & Run All*).
5. Retrieve `Machine_Learning_ready_Dataset.csv` for model training.

Example of loading the data for a model:

```python
import pandas as pd

data = pd.read_csv("Machine_Learning_ready_Dataset.csv")

train = data[data["Jeu"] == "train"].drop(columns="Jeu")
test = data[data["Jeu"] == "test"].drop(columns="Jeu")

X_train, y_train = train.drop(columns="HeartDisease"), train["HeartDisease"]
X_test, y_test = test.drop(columns="HeartDisease"), test["HeartDisease"]
```