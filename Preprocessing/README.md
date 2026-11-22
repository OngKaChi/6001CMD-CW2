# eCommerce Customer Churn — Data Preprocessing


**Dataset:** https://www.kaggle.com/datasets/srisyra02/ecommerce-customer-behavior-dataset/data

**Source provenance:** The supplied CSV does not document whether its observations were collected from real customers or generated synthetically. 

## Files

| Path | Role |
| --- | --- |
| `src/preprocessing.py` | Main automated audit, cleaning, feature engineering, train/test split, model preprocessing functions |
| `src/preprocessing_comparison.py` | Exploratory three-fold training-only Logistic Regression comparison |
| `notebooks/ecommerce_preprocessing.ipynb` | Notebook demonstrating the dataset audit and preprocessing steps (outputs cleared; execute locally) |
| `tests/test_preprocessing.py` | Data and pipeline tests |
| `data/raw/eCommerce-dataset.csv` | 50,000-row project input (reduced predictor schema) |
| `data/processed/cleaned_dataset.csv` | Intermediate audited data; missing predictor values intentionally retained |
| `data/processed/train_cleaned.csv` | 40,000 training observations for model cross-validation |
| `data/processed/test_cleaned.csv` | 10,000 holdout observations, reserved for final test evaluation |
| `data/processed/fully_cleaned_dataset.csv` | Full 50,000-row numeric export, train-fitted transforms, 40 predictors + target, no missing values |
| `data/processed/model_ready_train.csv`, `model_ready_test.csv` | Fixed train-fitted transformed matrices; NOT for model-selection CV |
| `data/processed/preprocessing_audit.json` | Data quality and split audit |
| `data/processed/preprocessing_cv_sensitivity.json` | Preliminary training-CV preprocessing comparison |
| `data/processed/holdout_preprocessor.joblib` | Saved fitted holdout preprocessor; only load when its source is trusted |

## Setup and run (Windows PowerShell)

Open a terminal **in the folder containing this README.md**, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.preprocessing --input data/raw/eCommerce-dataset.csv --task classification --target Churned --output-dir data/processed --export-model-ready
python -m unittest discover -s tests -v
```

Optional comparison: `python -m src.preprocessing_comparison`

Open `notebooks/ecommerce_preprocessing.ipynb` in VS Code using the `.venv` kernel, then select **Notebook: Run All**. The notebook has no precomputed outputs: all outputs you see after running it are generated locally.

## Verified results on the supplied CSV

- 50,000 rows × 21 input columns; 49,081 originally missing cells; 0 exact duplicates.
- 327 values outside explicit domain ranges were replaced with missing data (50 ages, 30 cart-abandonment rates, 207 discount-usage rates, 40 negative purchase amounts).
- Added `login_page_activity` and `service_calls_per_year_plus_one`.
- Stratified split: 40,000 train / 10,000 test (random_state=42).
- 40 transformed predictor columns with 0 missing values in the model-ready output.
- Class frequencies: 35,550 retained vs 14,450 churned.
- No additional statistical feature selection, dimensionality reduction or class resampling is automatically applied.

## CRITICAL modelling usage

For model comparison, tune or cross-validate only on `train_cleaned.csv` using a **fresh** preprocessing pipeline fitted within each training fold. Reserve `test_cleaned.csv` for final testing. **Do not** use `fully_cleaned_dataset.csv` or `model_ready_train.csv` for hyperparameter CV: their preprocessing was already fitted using all 40,000 training observations, so the fold validation data would have influenced preprocessing statistics.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.preprocessing import build_model_pipeline

train = pd.read_csv('data/processed/train_cleaned.csv')
X = train.drop(columns=['churned'])
y = train['churned']
model = build_model_pipeline(
    X, LogisticRegression(max_iter=1000, solver='liblinear'), task='classification'
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(model, X, y, cv=cv, scoring={'roc_auc': 'roc_auc', 'f1': 'f1'})
print(results['test_roc_auc'].mean(), results['test_f1'].mean())
```

Median imputation, missing-value indicators and RobustScaler are training-fitted and repeated within model cross-validation. All retained predictors are numerical; no categorical encoding is needed for this dataset. Oversampling and feature selection, if chosen, must also be fitted only on training folds. The prediction cutoff and any post-outcome features require confirmation before deployment.

