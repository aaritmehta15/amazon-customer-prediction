#!/usr/bin/env python3
"""
Buy Predictor Script for Amazon Consumer Behaviour Dataset

- Loads preprocessed data from 'cleaned_pre_imputation.csv'
- Uses 'permutation_importance.csv' to select top features
- Incorporates cluster labels as an additional feature (tries, in order):
  1) Existing column in the main CSV (e.g., 'Cluster', 'cluster', 'Cluster_ID')
  2) 'amazon_customers_annotated.csv' if length matches (index alignment)
  3) Predict clusters using saved pipelines: 'preprocessor_final.joblib', 'pca.joblib', 'kmeans.joblib'
  4) Fallback to a quick KMeans fit on the current data

- Trains models:
  * LightGBM (primary) with simple GridSearchCV tuning
  * Support Vector Classifier (SVC)
  * Optional Keras neural network (if TensorFlow/Keras installed)

- Saves: 'buy_predictor_model.joblib', 'buy_predictor_predictions.csv', 'buy_predictor_metrics.csv'
- Includes example prediction on a single row

Assumes input files are in the current working directory.
"""

from __future__ import annotations
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import joblib

# LightGBM (required)
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    print("ERROR: lightgbm is required. Please install with `pip install lightgbm`.", file=sys.stderr)
    raise

# Optional: TensorFlow/Keras
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Config:
    cleaned_csv: str = "cleaned_pre_imputation.csv"
    perm_importance_csv: str = "permutation_importance.csv"
    annotated_csv: str = "amazon_customers_annotated.csv"
    preprocessor_joblib: str = "preprocessor_final.joblib"
    pca_joblib: str = "pca.joblib"
    kmeans_joblib: str = "kmeans.joblib"
    model_out: str = "buy_predictor_model.joblib"
    preds_out: str = "buy_predictor_predictions.csv"
    metrics_out: str = "buy_predictor_metrics.csv"
    top_k_features: int = 15
    test_size: float = 0.2
    random_state: int = 42


def load_main_dataset(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.cleaned_csv):
        raise FileNotFoundError(f"Missing required file: {cfg.cleaned_csv}")
    df = pd.read_csv(cfg.cleaned_csv)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Create binary target 'Likely_To_Buy'.
    Logic:
      - 1 if Purchase_Frequency in {'Few times a week', 'Few times a month'}
      - else 0
      - If Purchase_Frequency missing/unusable, fallback to Shopping_Satisfaction > 3 => 1 else 0
    """
    target_col = "Likely_To_Buy"
    pf_col_candidates = [
        "Purchase_Frequency",
        "purchase_frequency",
        "Purchase frequency",
    ]

    def resolve_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    pf_col = resolve_col(pf_col_candidates)
    sat_col = None
    for c in ["Shopping_Satisfaction", "shopping_satisfaction", "Satisfaction", "Overall_Satisfaction"]:
        if c in df.columns:
            sat_col = c
            break

    likely = pd.Series(np.zeros(len(df), dtype=int))
    used_pf = False
    if pf_col is not None:
        vals = df[pf_col].astype(str).str.strip().str.lower()
        frequent = vals.isin({"few times a week", "few times a month"})
        if frequent.any():
            likely = frequent.astype(int)
            used_pf = True

    if not used_pf and sat_col is not None:
        # fallback on satisfaction > 3
        try:
            likely = (pd.to_numeric(df[sat_col], errors="coerce") > 3).fillna(0).astype(int)
        except Exception:
            likely = pd.Series(np.zeros(len(df), dtype=int))

    df[target_col] = likely.values
    return df, target_col


def load_cluster_labels(cfg: Config, df: pd.DataFrame) -> pd.Series:
    """Derive cluster labels as a Series aligned with df index.
    Tries multiple strategies. Returns categorical-like labels.
    """
    # 1) Existing column
    for col in ["Cluster_ID", "Cluster", "cluster", "cluster_id"]:
        if col in df.columns:
            print(f"Using existing cluster column: {col}")
            return df[col].astype(str)

    # 2) Annotated CSV index-alignment
    if os.path.exists(cfg.annotated_csv):
        try:
            ann = pd.read_csv(cfg.annotated_csv)
            if len(ann) == len(df):
                for col in ["Cluster_ID", "Cluster", "cluster", "cluster_id"]:
                    if col in ann.columns:
                        print(f"Using clusters from annotated CSV column: {col}")
                        return ann[col].astype(str)
            else:
                print("Annotated CSV present but length mismatch; skipping index alignment.")
        except Exception as e:
            print(f"Warning: Failed to use annotated CSV for clusters: {e}")

    # 3) Predict clusters via saved pipeline + PCA + KMeans
    try:
        if os.path.exists(cfg.preprocessor_joblib) and os.path.exists(cfg.kmeans_joblib):
            preproc = joblib.load(cfg.preprocessor_joblib)
            km = joblib.load(cfg.kmeans_joblib)
            X_trans = preproc.transform(df)
            if os.path.exists(cfg.pca_joblib):
                pca = joblib.load(cfg.pca_joblib)
                X_trans = pca.transform(X_trans)
            clabs = km.predict(X_trans)
            print("Using predicted clusters from saved pipeline and models")
            return pd.Series(clabs).astype(str)
    except Exception as e:
        print(f"Warning: Failed to predict clusters from saved models: {e}")

    # 4) Fallback KMeans on a quick numeric-only subset
    try:
        num_df = df.select_dtypes(include=[np.number]).copy()
        if num_df.shape[1] >= 1:
            imputer = SimpleImputer(strategy="median")
            num_X = imputer.fit_transform(num_df)
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            clabs = km.fit_predict(num_X)
            print("Using fallback KMeans clustering on numeric features")
            return pd.Series(clabs).astype(str)
    except Exception as e:
        print(f"Warning: Fallback KMeans failed: {e}")

    print("No cluster labels available; using 'unknown'")
    return pd.Series(["unknown"] * len(df))


def pick_top_features(cfg: Config, df: pd.DataFrame, include_cols: Optional[List[str]] = None) -> List[str]:
    features: List[str] = []
    if os.path.exists(cfg.perm_importance_csv):
        try:
            imp = pd.read_csv(cfg.perm_importance_csv)
            # Try common column names
            cand_cols = [
                "feature", "Feature", "features",
                "column", "Column",
            ]
            col_name = None
            for c in cand_cols:
                if c in imp.columns:
                    col_name = c
                    break
            if col_name is None:
                # assume first column is feature name
                col_name = imp.columns[0]
            feat_list = imp[col_name].astype(str).str.strip().tolist()
            # Intersect with current df columns
            features = [f for f in feat_list if f in df.columns]
        except Exception as e:
            print(f"Warning: Could not parse permutation_importance.csv: {e}")

    # Ensure we have enough features
    if not features:
        # Fallback: take non-target columns, prefer non-ID-like
        features = [c for c in df.columns if c not in {"Likely_To_Buy"}]
    if include_cols:
        for c in include_cols:
            if c not in features and c in df.columns:
                features.append(c)
    # Limit to top_k
    return features[: cfg.top_k_features]


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def evaluate_and_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def train_lgbm(X_train, X_test, y_train, y_test, preprocessor: ColumnTransformer, cfg: Config) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Pipeline]:
    lgbm = LGBMClassifier(random_state=cfg.random_state)
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", lgbm),
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [-1, 5, 10],
        "clf__num_leaves": [31, 63],
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    try:
        y_prob = best_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = evaluate_and_report(y_test, y_pred, y_prob)
    metrics["model"] = "LightGBM"
    metrics["best_params"] = gs.best_params_

    return metrics, y_pred, (y_prob if y_prob is not None else np.full_like(y_pred, np.nan, dtype=float)), best_model


def train_svc(X_train, X_test, y_train, y_test, preprocessor: ColumnTransformer, cfg: Config) -> Tuple[Dict[str, Any], Pipeline]:
    svc = SVC(probability=True, random_state=cfg.random_state)
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", svc),
    ])

    param_grid = {
        "clf__C": [0.5, 1.0, 2.0],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    try:
        y_prob = best_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = evaluate_and_report(y_test, y_pred, y_prob)
    metrics["model"] = "SVC"
    metrics["best_params"] = gs.best_params_
    return metrics, best_model


def train_keras(X_train, X_test, y_train, y_test, preprocessor: ColumnTransformer, cfg: Config) -> Optional[Dict[str, Any]]:
    if not TF_AVAILABLE:
        print("TensorFlow/Keras not available; skipping neural network experiment")
        return None

    # Build a pipeline to transform features then feed to Keras
    preprocessor.fit(X_train)
    Xtr = preprocessor.transform(X_train)
    Xte = preprocessor.transform(X_test)

    input_dim = Xtr.shape[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(Xtr, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[es], verbose=0)

    y_prob = model.predict(Xte, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluate_and_report(y_test, y_pred, y_prob)
    metrics["model"] = "KerasNN"
    metrics["params"] = {
        "layers": [64, 32],
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32,
    }
    return metrics


def main():
    cfg = Config()
    print("Loading main dataset...")
    df = load_main_dataset(cfg)

    print("Engineering target variable...")
    df, target_col = engineer_target(df)

    print("Deriving cluster labels...")
    clusters = load_cluster_labels(cfg, df)
    df["Cluster_ID"] = clusters

    print("Selecting top features from permutation importance...")
    top_features = pick_top_features(cfg, df, include_cols=["Cluster_ID"])
    # Ensure target not in features
    top_features = [c for c in top_features if c != target_col]

    # Prepare features/target
    X = df[top_features].copy()
    y = df[target_col].astype(int).values

    print(f"Feature columns ({len(top_features)}): {top_features}")

    print("Building preprocessor...")
    preprocessor, num_cols, cat_cols = build_preprocessor(df, top_features)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    all_metrics: List[Dict[str, Any]] = []

    print("Training LightGBM model (primary)...")
    lgbm_metrics, y_pred_lgbm, y_prob_lgbm, best_model = train_lgbm(X_train, X_test, y_train, y_test, preprocessor, cfg)
    all_metrics.append(lgbm_metrics)

    print("Training SVC model...")
    svc_metrics, svc_model = train_svc(X_train, X_test, y_train, y_test, preprocessor, cfg)
    all_metrics.append(svc_metrics)

    print("Training Keras NN (if available)...")
    nn_metrics = train_keras(X_train, X_test, y_train, y_test, preprocessor, cfg)
    if nn_metrics is not None:
        all_metrics.append(nn_metrics)

    print("Saving models and reports...")
    # Always save both individual models
    lgbm_path = "buy_predictor_model_lgbm.joblib"
    svc_path = "buy_predictor_model_svc.joblib"
    joblib.dump(best_model, lgbm_path)
    joblib.dump(svc_model, svc_path)

    # Choose best by F1 and save as default
    best_entry = max(all_metrics, key=lambda m: m.get("f1", float("-inf")))
    chosen = best_entry.get("model", "LightGBM")
    if chosen == "SVC":
        joblib.dump(svc_model, cfg.model_out)
    else:
        joblib.dump(best_model, cfg.model_out)
    print(f"Best model by F1: {chosen}. Saved as '{cfg.model_out}'. Also saved LGBM -> {lgbm_path}, SVC -> {svc_path}")

    # Save predictions from primary model
    preds_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred_lgbm,
        "y_prob": y_prob_lgbm,
    })
    preds_df.to_csv(cfg.preds_out, index=False)

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(cfg.metrics_out, index=False)

    # Print a brief report
    print("\n=== Metrics Summary ===")
    print(metrics_df)

    # Example prediction on a single row (first test sample)
    try:
        sample_row = X_test.iloc[[0]]
        sample_pred = best_model.predict(sample_row)[0]
        try:
            sample_prob = best_model.predict_proba(sample_row)[0, 1]
        except Exception:
            sample_prob = np.nan
        print("\nExample prediction on a held-out sample:")
        print({"prediction": int(sample_pred), "prob_buy": float(sample_prob) if sample_prob == sample_prob else None})
    except Exception as e:
        print(f"Could not generate sample prediction: {e}")


if __name__ == "__main__":
    main()
