"""Data-integrity and preprocessing pipeline tests."""
import json, unittest
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.preprocessing import EXCLUDED_PREDICTORS, build_model_pipeline, load_clean_base, build_preprocessor, column_name

ROOT=Path(__file__).resolve().parents[1]

class TestActualUploadedDataset(unittest.TestCase):
    def test_original_data_verified(self):
        raw=pd.read_csv(ROOT/'data/raw/eCommerce-dataset.csv')
        self.assertEqual(raw.shape,(50000,21))
        self.assertEqual(int(raw.isna().sum().sum()),49081)
        self.assertEqual(int(raw.duplicated().sum()),0)
        self.assertEqual(raw.Churned.value_counts().to_dict(),{0:35550,1:14450})
    def test_domain_and_features(self):
        cleaned,audit=load_clean_base(ROOT/'data/raw/eCommerce-dataset.csv',[])
        self.assertEqual(audit['exact_duplicate_rows_removed'],0)
        self.assertEqual(sum(v['count_replaced_by_missing'] for v in audit['invalid_domain_values_replaced_with_missing'].values()),327)
        self.assertEqual(len(audit['engineered_feature_columns']),2)
        self.assertEqual(int((cleaned.age<18).sum()),0)
        self.assertEqual(int((cleaned.cart_abandonment_rate>100).sum()),0)
    def test_exports_are_complete_and_stratified(self):
        d=ROOT/'data/processed'
        a=json.loads((d/'preprocessing_audit.json').read_text())
        self.assertEqual(a['split']['train_rows'],40000)
        self.assertEqual(a['split']['test_rows'],10000)
        self.assertEqual(a['model_ready_feature_count'],40)
        self.assertEqual(a['fully_cleaned_total_nulls'],0)
        self.assertEqual(a['model_ready_train_total_nulls'],0)
        self.assertEqual(a['model_ready_test_total_nulls'],0)
        self.assertEqual(a['split']['test_class_distribution'],{'0':7110,'1':2890})
        df=pd.read_csv(d/'fully_cleaned_dataset.csv')
        self.assertEqual(df.shape,(50000,41))
        self.assertEqual(df.isna().sum().sum(),0)
    def test_export_schema(self):
        from joblib import load
        d=ROOT/'data/processed'
        for path in [ROOT/'data/raw/eCommerce-dataset.csv', *d.glob('*.csv')]:
            columns=pd.read_csv(path,nrows=0).columns
            self.assertFalse(any(
                column_name(c) in EXCLUDED_PREDICTORS or
                any(c.lower().startswith('categorical__'+name+'_') for name in EXCLUDED_PREDICTORS)
                for c in columns
            ), path.name)
        transformer=load(d/'holdout_preprocessor.joblib')
        self.assertFalse(set(transformer.feature_names_in_) & EXCLUDED_PREDICTORS)
        self.assertEqual(len(transformer.get_feature_names_out()),40)
        self.assertTrue(all(c.startswith('numeric__') for c in transformer.get_feature_names_out()))
        self.assertEqual(set(EXCLUDED_PREDICTORS), {'country', 'gender', 'city', 'signup_quarter'})
        X=pd.DataFrame({'kept':[1,2], **{name:[0,1] for name in EXCLUDED_PREDICTORS}})
        self.assertEqual(list(build_preprocessor(X).fit(X).get_feature_names_out()),['numeric__kept'])
    def test_fold_safe_pipeline_fits(self):
        train=pd.read_csv(ROOT/'data/processed/train_cleaned.csv',nrows=400)
        X=train.drop(columns=['churned']);y=train.churned
        p=build_model_pipeline(X,LogisticRegression(max_iter=300,solver='liblinear'),task='classification')
        s=cross_val_score(p,X,y,scoring='roc_auc',cv=StratifiedKFold(n_splits=2))
        self.assertEqual(len(s),2)
if __name__=='__main__':unittest.main()
