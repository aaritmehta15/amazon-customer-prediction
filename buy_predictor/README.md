# Buy Predictor Module

This module contains the core functionality for predicting customer purchase behavior based on their interactions and profile data.

## Files

- `buy_predictor.py`: Main module for training and evaluating the purchase prediction models
- `buy_predictor_test.py`: Script for testing the trained model with new data
- `buy_predictor_cli.py`: Command-line interface for making predictions interactively

## Models

### LightGBM
- **File**: `buy_predictor_model_lgbm.joblib`
- **Performance**:
  - Accuracy: 0.63
  - F1-Score: 0.31
  - ROC-AUC: 0.52

### SVC
- **File**: `buy_predictor_model_svc.joblib`
- **Performance**:
  - Accuracy: 0.64
  - F1-Score: 0.19
  - ROC-AUC: 0.55

## Usage

### Training
```bash
python buy_predictor.py
```

### Testing with New Data
```bash
python buy_predictor_test.py --input your_data.csv --output predictions.csv
```

### Interactive Prediction
```bash
python buy_predictor_cli.py
```

## Input Data Format

Refer to `../DATA_DICTIONARY.md` for detailed information about input features.

## Output Files

- `buy_predictor_metrics.csv`: Model performance metrics
- `buy_predictor_predictions.csv`: Prediction results
- `permutation_importance.csv`: Feature importance scores
