#!/usr/bin/env python3
"""
Quick tester for the Buy Predictor models.

Usage examples (PowerShell from project root or buy_predictor folder):

- Print expected input feature columns for the default model:
  python buy_predictor\buy_predictor_test.py --print-schema

- Predict on a CSV containing one or more rows with those columns:
  python buy_predictor\buy_predictor_test.py --input-csv buy_predictor\my_samples.csv

- Choose a specific saved model (SVC or LGBM):
  python buy_predictor\buy_predictor_test.py --model buy_predictor\buy_predictor_model_svc.joblib --input-csv buy_predictor\my_samples.csv

Notes:
- The saved pipeline includes preprocessing; provide the original feature columns
  printed by --print-schema. Missing columns will be added as NaN and imputed.
- The script saves predictions to buy_predictor/predictions_inference.csv by default.
"""

import argparse
import os
from typing import List

import joblib
import pandas as pd


def get_expected_columns(pipeline) -> List[str]:
    # Extract original input column names expected by the ColumnTransformer
    pre = pipeline.named_steps.get("pre")
    if pre is None:
        # If no preprocessor, we cannot infer; expect raw columns
        return []
    cols: List[str] = []
    for name, transformer, columns in pre.transformers:
        if isinstance(columns, list):
            cols.extend(columns)
    # Remove duplicates while preserving order
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Test Buy Predictor model on new data")
    parser.add_argument("--model", default=os.path.join("buy_predictor", "buy_predictor_model.joblib"), help="Path to saved model pipeline (.joblib)")
    parser.add_argument("--input-csv", default=None, help="Path to CSV with samples to predict")
    parser.add_argument("--out-csv", default=os.path.join("buy_predictor", "predictions_inference.csv"), help="Where to save predictions CSV")
    parser.add_argument("--print-schema", action="store_true", help="Print required input feature columns and exit")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    pipe = joblib.load(args.model)

    expected_cols = get_expected_columns(pipe)
    if args.print-schema:
        print("Expected input feature columns (order not important):")
        for c in expected_cols:
            print(c)
        return

    if args.input_csv is None:
        raise SystemExit("Please provide --input-csv or use --print-schema to see required columns.")

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    # Drop target if present
    for t in ["Likely_To_Buy", "likely_to_buy", "target"]:
        if t in df.columns:
            df = df.drop(columns=[t])

    # Align columns
    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = pd.NA
        # Keep only expected columns (model ignores extras)
        df = df[expected_cols]

    # Predict
    preds = pipe.predict(df)
    try:
        probs = pipe.predict_proba(df)[:, 1]
    except Exception:
        probs = [float("nan")] * len(preds)

    out = df.copy()
    out["pred_buy"] = preds
    out["prob_buy"] = probs

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Saved predictions to: {args.out_csv}")
    # Show a small preview
    print(out.head(min(10, len(out))))


if __name__ == "__main__":
    main()
