# Amazon Customer Prediction - Runbook

This guide provides detailed instructions for setting up and using the Amazon Customer Prediction system.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running the Model](#running-the-model)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
    - [Interactive CLI](#interactive-cli)
    - [File-based Prediction](#file-based-prediction)
- [Understanding the Outputs](#understanding-the-outputs)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd ML-MINI-PROJ
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Model

### Training the Model

1. **Navigate to the project directory**:
   ```bash
   cd path/to/ML-MINI-PROJ
   ```

2. **Run the training script**:
   ```bash
   python buy_predictor/buy_predictor.py
   ```

   This will:
   - Load and preprocess the data
   - Train the LightGBM and SVC models
   - Save the best model to `buy_predictor/buy_predictor_model.joblib`
   - Generate evaluation metrics in `buy_predictor/buy_predictor_metrics.csv`

### Making Predictions

#### Interactive CLI

1. **Run the interactive CLI**:
   ```bash
   python buy_predictor/buy_predictor_cli.py
   ```

2. **Follow the prompts** to enter values for each feature. Example:
   ```
   Enter values for the following features. Press Enter to leave blank (will be imputed).
     shopping_satisfaction_num (numeric (range approx: 1-5)): 4
     customer_reviews_importance_num (numeric (range approx: 1-5)): 5
     ...
   ```

3. **View the prediction**:
   ```
   {'pred_buy': 1, 'prob_buy': 0.85}
   ```
   - `pred_buy`: 1 (likely to buy) or 0 (unlikely to buy)
   - `prob_buy`: Confidence score between 0 and 1

#### File-based Prediction

1. **Prepare a CSV file** with input data. Use the schema from:
   ```bash
   python buy_predictor/buy_predictor_cli.py --print-schema
   ```

2. **Run predictions on the file**:
   ```bash
   python buy_predictor/buy_predictor_test.py --input-csv your_data.csv
   ```

3. **Results** are saved to `buy_predictor/predictions_inference.csv` by default.

## Understanding the Outputs

### Model Outputs
- `buy_predictor_model.joblib`: The trained model (best by F1 score)
- `buy_predictor_model_lgbm.joblib`: LightGBM model
- `buy_predictor_model_svc.joblib`: SVC model
- `buy_predictor_metrics.csv`: Performance metrics for all models
- `buy_predictor_predictions.csv`: Test set predictions

### Prediction Output
- `predictions_inference.csv`: Contains:
  - Original input features
  - `pred_buy`: Binary prediction (0/1)
  - `prob_buy`: Probability of class 1 (buy)

## Troubleshooting

### Common Issues

1. **Module Not Found Error**
   ```
   ModuleNotFoundError: No module named 'lightgbm'
   ```
   **Solution**: Install the missing package:
   ```bash
   pip install lightgbm
   ```

2. **File Not Found Error**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'buy_predictor_model.joblib'
   ```
   **Solution**: Ensure you're running the command from the correct directory or provide the full path to the model file.

3. **Version Mismatch**
   ```
   AttributeError: 'OneHotEncoder' object has no attribute 'sparse'
   ```
   **Solution**: Update scikit-learn:
   ```bash
   pip install --upgrade scikit-learn
   ```

### Getting Help
For additional support, please provide:
1. The exact command you ran
2. The full error message
3. Your Python version (`python --version`)
4. The output of `pip list`
