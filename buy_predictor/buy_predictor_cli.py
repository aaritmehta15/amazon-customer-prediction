#!/usr/bin/env python3
"""
Interactive CLI to use the Buy Predictor (best saved model) with helpful prompts.

- Loads best model by default: buy_predictor/buy_predictor_model.joblib
- Detects expected input columns from the saved preprocessing pipeline
- Prompts you to enter values per column (press Enter to leave blank = imputed)
- Prints predicted class and probability

Run from project root or from buy_predictor/:
  python buy_predictor\buy_predictor_cli.py

Useful options:
  - Print expected schema with example ranges/options:
      python buy_predictor\buy_predictor_cli.py --print-schema
  - Choose a specific model:
      python buy_predictor\buy_predictor_cli.py --model buy_predictor\buy_predictor_model_svc.joblib
"""

import argparse
import os
from typing import List, Tuple, Dict
import joblib
import pandas as pd


def get_schema(pipeline) -> Tuple[List[str], List[str], List[str]]:
    pre = pipeline.named_steps.get("pre")
    if pre is None:
        return [], [], []
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for name, transformer, columns in pre.transformers:
        if name == "num":
            if isinstance(columns, list):
                num_cols.extend(columns)
        elif name == "cat":
            if isinstance(columns, list):
                cat_cols.extend(columns)
    # Merge in order
    cols: List[str] = []
    seen = set()
    for c in num_cols + cat_cols:
        if c not in seen:
            seen.add(c)
            cols.append(c)
    return cols, num_cols, cat_cols


def build_hints(cols: List[str], num_cols: List[str]) -> Dict[str, str]:
    """Build friendly hints based on cleaned_pre_imputation.csv if available."""
    hints: Dict[str, str] = {}
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "cleaned_pre_imputation.csv")
    df = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None
    for c in cols:
        if df is not None and c in df.columns:
            if c in num_cols:
                s = pd.to_numeric(df[c], errors="coerce")
                s = s.dropna()
                if len(s) > 0:
                    hints[c] = f"numeric (range approx: {s.min():.0f}-{s.max():.0f})"
                else:
                    hints[c] = "numeric"
            else:
                s = df[c].astype(str)
                top = s.value_counts().head(5).index.tolist()
                if top:
                    hints[c] = "categorical (examples: " + ", ".join(top) + ")"
                else:
                    hints[c] = "categorical"
        else:
            hints[c] = "numeric" if c in num_cols else "categorical"
    return hints


def prompt_for_row(cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    print("\nEnter values for the following features. Press Enter to leave blank (will be imputed).")
    hints = build_hints(cols, num_cols)
    data = {}
    for c in cols:
        hint = f"({hints.get(c, 'value')})"
        try:
            val = input(f"  {c} {hint}: ").strip()
        except EOFError:
            val = ""
        if val == "":
            data[c] = pd.NA
        else:
            # Try cast numeric if expected numeric
            if c in num_cols:
                try:
                    # accept ints and floats
                    if "." in val:
                        data[c] = float(val)
                    else:
                        data[c] = int(val)
                except Exception:
                    # fallback keep as string; preprocessor may error if wrong type
                    data[c] = val
            else:
                data[c] = val
    return pd.DataFrame([data], columns=cols)


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for Buy Predictor")
    default_model = os.path.join(os.path.dirname(__file__), "buy_predictor_model.joblib")
    parser.add_argument("--model", default=default_model, help="Path to saved model (.joblib)")
    parser.add_argument("--print-schema", action="store_true", help="Print required input feature columns and exit")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    pipe = joblib.load(args.model)
    cols, num_cols, cat_cols = get_schema(pipe)

    print("Loaded model:", args.model)
    if cols:
        print(f"Model expects {len(cols)} input columns.")
    else:
        print("Note: No preprocessor found; predictions require a DataFrame with the original training columns.")

    if args.print_schema:
        # print columns with hints and exit
        hints = build_hints(cols, num_cols)
        print("\nExpected input feature columns with examples/hints:")
        for c in cols:
            print(f" - {c}: {hints.get(c, '')}")
        return

    while True:
        row = prompt_for_row(cols, num_cols)
        try:
            pred = pipe.predict(row)[0]
            try:
                prob = float(pipe.predict_proba(row)[0, 1])
            except Exception:
                prob = float('nan')
            print("\nPrediction:")
            print({"pred_buy": int(pred), "prob_buy": prob})
        except Exception as e:
            print(f"Error while predicting: {e}")

        again = input("\nPredict another? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
