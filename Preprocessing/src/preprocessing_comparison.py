"""Training-only comparison of preprocessing strategies.

Reproduce with
python -m src.preprocessing_comparison from the project root. Fits only
within cross-validation folds, no access to holdout test labels.
"""
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from src.preprocessing import build_model_pipeline

train = pd.read_csv('data/processed/train_cleaned.csv')
y = train['churned']
base_features = [c for c in train.columns if c not in ['churned','login_page_activity','service_calls_per_year_plus_one']]
all_features = base_features + ['login_page_activity','service_calls_per_year_plus_one']
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}
config = [
    ('without_engineered_features', base_features, False, False),
    ('with_engineered_features_robust_scaler', all_features, False, False),
    ('with_engineered_features_standard_scaler', all_features, True, False),
    ('with_engineered_features_balanced_logistic', all_features, False, True),
]
for name, cols, standard, balanced in config:
    est = LogisticRegression(max_iter=800, solver='liblinear', class_weight='balanced' if balanced else None)
    pipe = build_model_pipeline(train[cols], est, task='classification')
    if standard:pipe.set_params(preprocess__numeric__scale=StandardScaler())
    start = time.monotonic()
    scores = cross_validate(pipe, train[cols], y, cv=cv, n_jobs=1,
                            scoring={'roc_auc':'roc_auc','f1':'f1', 'precision':'precision', 'recall':'recall'},
                            error_score='raise')
    results[name] = {key:round(float(np.mean(scores['test_'+key])),5) for key in ['roc_auc','f1','precision','recall']}
    results[name]['elapsed_seconds']=round(time.monotonic()-start,2)
    print(name,results[name])
Path('data/processed/preprocessing_cv_sensitivity.json').write_text(json.dumps({'approach':'three-fold cross-validation using only the 40,000-row training split; binary churn, Logistic Regression; exploratory comparison, not final model-selection results','results':results},indent=2))
