"""Reproducible eCommerce data preprocessing.

The default invocation uses the team-uploaded 50,000-row eCommerce dataset with
target Churned for binary customer-churn classification.
Train/test separation precedes learned imputation, scaling and encoding.
Do not use the holdout-transformed full CSV as input for model-selection CV.

Examples:
  python -m src.preprocessing --input data/raw/eCommerce-dataset.csv \
      --task classification --target Churned --export-model-ready

Run from the repository root. See README.md for all options.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Values must be unambiguous placeholders; strings such as "None" are left untouched
# because in some source systems they denote a legitimate category.
EXCLUDED_PREDICTORS = frozenset({"country", "gender", "city", "signup_quarter"})

MISSING_TOKENS = {"", "na", "n/a", "null", "nan", "?", "not available"}


def column_name(name: str) -> str:
    """Convert source headings to stable snake_case without changing data values."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name).strip())
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower()


def clean_strings(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        # Do not turn a mixed numeric field into a string without reviewing its type.
        s = out[col]
        out[col] = s.map(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in MISSING_TOKENS
            else v.strip() if isinstance(v, str) else v
        )
    return out


def validate_ecommerce_domain(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Set demonstrably implausible observations to NaN, retaining their rows.

    Age range 18–100 is an explicit ADULT customer assumption; change if the
    company's permitted customer ages differ. Percentage ranges assume 0–100.
    No winsorisation of plausible high monetary values is performed.
    """
    out = df.copy()
    bounds = {
        "age": (18, 100),
        "cart_abandonment_rate": (0, 100),
        "discount_usage_rate": (0, 100),
        "total_purchases": (0, None),
    }
    counts: dict[str, dict] = {}
    for col, (lower, upper) in bounds.items():
        if col not in out:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        invalid = s.lt(lower)
        if upper is not None:
            invalid = invalid | s.gt(upper)
        counts[col] = {
            "below_min": int(s.lt(lower).sum()),
            "above_max": int(s.gt(upper).sum()) if upper is not None else 0,
            "count_replaced_by_missing": int(invalid.sum()),
            "permitted_interval": [lower, upper],
        }
        out.loc[invalid, col] = np.nan
    return out, counts


def engineer_ecommerce_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Deterministic behavioural features; no training statistics are learned."""
    out = df.copy()
    engineered = []
    if {"login_frequency", "pages_per_session"}.issubset(out.columns):
        out["login_page_activity"] = out["login_frequency"] * out["pages_per_session"]
        engineered.append("login_page_activity")
    if {"customer_service_calls", "membership_years"}.issubset(out.columns):
        out["service_calls_per_year_plus_one"] = out["customer_service_calls"] / (1.0 + out["membership_years"].clip(lower=0))
        engineered.append("service_calls_per_year_plus_one")
    return out, engineered


def load_clean_base(path: Path, date_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Source CSV not found: {path}")
    original = pd.read_csv(
        path, low_memory=False, usecols=lambda c: column_name(c) not in EXCLUDED_PREDICTORS
    )
    if original.empty:
        raise ValueError("The source CSV contains no observations.")
    renamed = [column_name(c) for c in original.columns]
    if len(set(renamed)) != len(renamed) or not all(renamed):
        raise ValueError("Column-name normalization caused duplicate/empty names. Rename source columns first.")
    df = original.copy()
    rename_map = dict(zip(list(original.columns), renamed))
    df.columns = renamed
    df = clean_strings(df)
    initial_missing = {c: int(df[c].isna().sum()) for c in df.columns}
    df, invalid_counts = validate_ecommerce_domain(df)
    missing_after_rules = {c: int(df[c].isna().sum()) for c in df.columns}
    duplicate_n = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df, engineered_cols = engineer_ecommerce_features(df)

    date_diagnostics = {}
    for col in date_columns:
        if col not in df:
            raise ValueError(f"Date column '{col}' not found. Available columns: {list(df.columns)}")
        parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
        invalid = int((df[col].notna() & parsed.isna()).sum())
        date_diagnostics[col] = {"unparseable_non_null": invalid}
        # Keep the source column in ISO-compatible text form in the cleaned CSV.
        # Date-derived features are deterministic and therefore safe before splitting.
        df[f"{col}_month"] = parsed.dt.month.astype("float64")
        df[f"{col}_dayofweek"] = parsed.dt.dayofweek.astype("float64")
        df[f"{col}_quarter"] = parsed.dt.quarter.astype("float64")
        df[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).where(parsed.notna(), np.nan).astype("float64")
        df[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")

    audit = {
        "source_csv": str(path),
        "source_rows": int(len(original)),
        "source_columns": int(len(original.columns)),
        "rubric_minimum_5000_records_met": bool(len(original) >= 5000),
        "rubric_minimum_10_attributes_met": bool(len(original.columns) >= 10),
        "renamed_columns": rename_map,
        "missing_values_in_raw_csv": initial_missing,
        "missing_values_after_domain_rules_before_engineering": missing_after_rules,
        "invalid_domain_values_replaced_with_missing": invalid_counts,
        "engineered_feature_columns": engineered_cols,
        "fractional_total_purchases_observed_not_corrected": int(
            ((original["Total_Purchases"].dropna() % 1) != 0).sum()
        ) if "Total_Purchases" in original.columns else None,
        "exact_duplicate_rows_removed": duplicate_n,
        "rows_after_exact_deduplication": int(len(df)),
        "missing_values_after_basic_cleaning": {c: int(df[c].isna().sum()) for c in df.columns},
        "date_parsing": date_diagnostics,
        "column_dtypes_after_cleaning": {c: str(df[c].dtype) for c in df.columns},
    }
    return df, audit


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessor to be fitted within the training folds of cross-validation."""
    features = features.loc[:, [c for c in features.columns if column_name(c) not in EXCLUDED_PREDICTORS]]
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    # Treat booleans and categorical fields as categorical; IDs must be excluded by caller.
    categorical_cols = [c for c in features.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
            ("scale", RobustScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    if not transformers:
        raise ValueError("No predictive columns remain. Reconsider excluded/ID/date columns.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_model_pipeline(
    X_train: pd.DataFrame, estimator: Any, *, task: str,
    balance: bool = False, select_k: int | None = None,
    svd_components: int | None = None, random_state: int = 42,
):
    """Construct a cross-validation-safe model pipeline.

    Hyperparameter search must run on this *whole* pipeline using only X_train / y_train.
    Balance is applied only to each CV training fold; never to validation/test rows.
    PCA/SVD and feature selection are optional, not default operations.
    """
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError("Unsupported ML category")
    if balance and task != "classification":
        raise ValueError("Class balancing is meaningful only for classification.")
    steps = [("preprocess", build_preprocessor(X_train))]
    if balance:
        try:
            from imblearn.over_sampling import RandomOverSampler
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:
            raise ImportError("Install imbalanced-learn to enable training-fold resampling.") from exc
        steps.append(("balance", RandomOverSampler(random_state=random_state)))
    if select_k is not None:
        if task == "clustering":
            raise ValueError("Supervised SelectKBest is not valid for clustering without labels.")
        if select_k < 1:
            raise ValueError("select_k must be a positive number")
        steps.append(("feature_selection", SelectKBest(
            score_func=f_classif if task == "classification" else f_regression, k=select_k)))
    if svd_components is not None:
        if svd_components < 1:
            raise ValueError("svd_components must be a positive integer")
        steps.append(("dimension_reduction", TruncatedSVD(n_components=svd_components, random_state=random_state)))
    steps.append(("model", estimator))
    if balance:
        return ImbPipeline(steps)
    return Pipeline(steps)


def parse_fields(value: list[str] | None) -> list[str]:
    if not value:
        return []
    return [column_name(field) for field in value]


def run(args: argparse.Namespace) -> dict:
    target = column_name(args.target) if args.target else None
    ids = parse_fields(args.id_columns)
    excludes = parse_fields(args.exclude_columns)
    dates = parse_fields(args.date_columns)
    group = column_name(args.group_column) if args.group_column else None
    chrono = column_name(args.chrono_column) if args.chrono_column else None

    if args.task in {"classification", "regression"} and not target:
        raise ValueError("--target is REQUIRED for classification/regression; do not guess it.")
    if args.task == "clustering" and target:
        raise ValueError("Clustering is unsupervised. Omit --target.")
    if args.split == "chronological" and not chrono:
        raise ValueError("--split chronological requires --chrono-column.")
    if args.split == "chronological" and group:
        raise ValueError("Specify EITHER a chronological split OR a group split, not both.")
    if chrono and chrono not in dates:
        raise ValueError("Pass your --chrono-column in --date-columns, too.")

    clean, audit = load_clean_base(Path(args.input), dates)
    required = set(ids + excludes + dates + ([target] if target else []) + ([group] if group else []))
    unknown = sorted(c for c in required if c not in clean.columns)
    if unknown:
        raise ValueError(f"Unknown columns: {unknown}; available: {list(clean.columns)}")
    if target in ids or target in excludes or target in dates:
        raise ValueError("Target must not be listed among IDs, excluded or date columns.")
    if target and target == group:
        raise ValueError("Target and grouping identifier cannot be the same column.")
    if args.task == "regression":
        as_numeric = pd.to_numeric(clean[target], errors="coerce")
        audit["regression_target_unparseable_non_null"] = int((clean[target].notna() & as_numeric.isna()).sum())
        clean[target] = as_numeric
    if target:
        audit["missing_target_rows_removed"] = int(clean[target].isna().sum())
        clean = clean.loc[clean[target].notna()].reset_index(drop=True)
    else:
        audit["missing_target_rows_removed"] = 0
    if len(clean) < 2:
        raise ValueError("Not enough usable observations after deduplication/target validation.")

    # Do NOT silently discard unknown post-outcome features. The modeller must explicitly
    # specify them with --exclude-columns once the prediction time is agreed.
    excluded_model = set(ids + excludes + dates + ([target] if target else []) + ([group] if group else []))
    predictor_cols = [c for c in clean.columns if c not in excluded_model]
    if not predictor_cols:
        raise ValueError("No predictor columns remain after exclusions.")
    X = clean[predictor_cols].copy()
    y = clean[target].copy() if target else None
    high_cardinality = {}
    for col in X.select_dtypes(include=["object", "string"]).columns:
        unique = int(X[col].nunique(dropna=True))
        if unique > 30 and (unique / len(X)) > 0.5:
            high_cardinality[col] = {"unique_values": unique, "unique_ratio": round(unique / len(X), 3)}
    if high_cardinality:
        warnings.warn(
            "Potential identifier/high-cardinality columns: " + ", ".join(high_cardinality) +
            ". Check --id-columns / --exclude-columns before modelling.", stacklevel=1)

    audit.update({
        "task": args.task, "target": target, "id_columns_excluded_from_model": ids,
        "other_columns_excluded_from_model": excludes,
        "date_columns": dates, "model_features": predictor_cols,
        "potential_identifier_columns_to_review": high_cardinality,
        "rows_in_cleaned_dataset": len(clean),
        "remaining_predictor_nulls_before_training_imputation": int(X.isna().sum().sum()),
        "predictor_missing_rows_before_training_imputation": int(X.isna().any(axis=1).sum()),
        "important_note": (
            "cleaned_dataset.csv preserves predictor nulls to avoid leakage. "
            "fully_cleaned_dataset.csv is imputed and robust-scaled (with one-hot encoding only if categorical predictors remain) "
            "using parameters FIT ON TRAIN only; NOT for hyperparameter cross-validation. "
            "No row outliers are deleted or classes resampled automatically."
        ),
    })
    if y is not None and args.task == "classification":
        audit["class_distribution"] = {str(k): int(v) for k, v in y.value_counts(dropna=False).items()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_dir / "cleaned_dataset.csv", index=False)

    if args.task in {"classification", "regression"}:
        if group:
            splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
            tr_i, te_i = next(splitter.split(X, y, groups=clean[group]))
            split_strategy = f"group-disjoint split by {group}"
        elif args.split == "chronological":
            datetimes = pd.to_datetime(clean[chrono], errors="coerce")
            if datetimes.isna().any():
                raise ValueError("Chronological split requested but date column has missing/unparseable dates.")
            sorted_ids = np.argsort(datetimes.to_numpy(), kind="stable")
            cut = int(len(clean) * (1 - args.test_size))
            tr_i, te_i = sorted_ids[:cut], sorted_ids[cut:]
            split_strategy = f"chronological split by {chrono}"
        else:
            stratify = None
            if args.task == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2:
                stratify = y
            tr_i, te_i = train_test_split(
                np.arange(len(clean)), test_size=args.test_size,
                random_state=args.random_state, stratify=stratify)
            split_strategy = "stratified random" if stratify is not None else "random"
        train = clean.iloc[tr_i].copy()
        test = clean.iloc[te_i].copy()
        train.to_csv(out_dir / "train_cleaned.csv", index=False)
        test.to_csv(out_dir / "test_cleaned.csv", index=False)
        audit["split"] = {
            "method": split_strategy, "train_rows": len(train), "test_rows": len(test),
            "test_size_requested": args.test_size, "random_state": args.random_state,
        }
        if args.task == "classification":
            audit["split"]["train_class_distribution"] = {str(k): int(v) for k, v in train[target].value_counts().items()}
            audit["split"]["test_class_distribution"] = {str(k): int(v) for k, v in test[target].value_counts().items()}
        if args.export_model_ready:
            # For illustrations or a *single fixed holdout evaluation* only.
            # During hyperparameter CV, use build_model_pipeline with raw train_cleaned.
            from joblib import dump
            feature_train = train[predictor_cols]
            feature_test = test[predictor_cols]
            transformer = build_preprocessor(feature_train)
            a = transformer.fit_transform(feature_train)
            b = transformer.transform(feature_test)
            names = transformer.get_feature_names_out()
            final_train = pd.DataFrame(a, columns=names, index=train.index)
            final_test = pd.DataFrame(b, columns=names, index=test.index)
            final_train[target] = train[target]
            final_test[target] = test[target]
            # Transform all rows using the TRAIN-fitted transformer; preserve original order.
            final_full = pd.DataFrame(transformer.transform(clean[predictor_cols]),
                                      columns=names, index=clean.index)
            final_full[target] = clean[target].to_numpy()
            final_full.to_csv(out_dir / "fully_cleaned_dataset.csv", index=False)
            final_train.to_csv(out_dir / "model_ready_train.csv", index=False)
            final_test.to_csv(out_dir / "model_ready_test.csv", index=False)
            dump(transformer, out_dir / "holdout_preprocessor.joblib")
            audit["model_ready_feature_count"] = len(names)
            audit["model_ready_columns"] = list(names)
            audit["fully_cleaned_total_nulls"] = int(final_full.isna().sum().sum())
            audit["model_ready_train_total_nulls"] = int(final_train.isna().sum().sum())
            audit["model_ready_test_total_nulls"] = int(final_test.isna().sum().sum())
            audit["model_ready_warning"] = (
                "Do not run model selection or cross-validation on pretransformed fully_cleaned_dataset.csv or model_ready_train.csv; "
                "use the fit-within-fold build_model_pipeline on train_cleaned.csv instead."
            )
    else:
        audit["split"] = None
        audit["unsupervised_note"] = (
            "No target labels and no supervised train/test split. Fit preprocessing within "
            "the chosen clustering validation protocol."
        )
    audit["generated_utc"] = datetime.now(timezone.utc).isoformat()
    with (out_dir / "preprocessing_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)
    print(f"Cleaned dataset saved: {out_dir / 'cleaned_dataset.csv'}")
    print(f"Audit report saved: {out_dir / 'preprocessing_audit.json'}")
    if audit["split"]:
        print(f"Split: {audit['split']['train_rows']} train / {audit['split']['test_rows']} test")
    print("Reminder: final missing-value imputation/scaling/encoding belongs INSIDE model CV pipelines.")
    return audit


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the Kaggle source CSV")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--task", required=True, choices=["classification", "regression", "clustering"])
    parser.add_argument("--target", help="The supervised label name, as defined for the modelling task")
    parser.add_argument("--id-columns", nargs="*", default=[], help="Identifiers to keep in clean data but exclude from modelling")
    parser.add_argument("--exclude-columns", nargs="*", default=[], help="Leakage/post-outcome variables or irrelevant predictors")
    parser.add_argument("--date-columns", nargs="*", default=[], help="Parse these as dates and create month/day/quarter/weekend features")
    parser.add_argument("--group-column", help="Customer/entity identifier: prevents repeated entities appearing in both splits")
    parser.add_argument("--split", choices=["random", "chronological"], default="random")
    parser.add_argument("--chrono-column", help="Date column used if --split chronological")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--export-model-ready", action="store_true", help="Export extra holdout-ready matrix; NOT for cross-validation")
    args = parser.parse_args(argv)
    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.5")
    return run(args)


if __name__ == "__main__":
    main()
