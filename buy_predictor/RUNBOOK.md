# Buy Predictor - Runbook

This document provides operational guidance for the Buy Predictor module.

## Setup

### Prerequisites
- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Create and activate virtual environment
python -m venv .venv
.\\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Model Training

### Training Command
```bash
python buy_predictor.py
```

### Expected Output
- Trained model files:
  - `buy_predictor_model.joblib` (best model)
  - `buy_predictor_model_lgbm.joblib` (LightGBM model)
  - `buy_predictor_model_svc.joblib` (SVC model)
- Evaluation metrics: `buy_predictor_metrics.csv`
- Feature importance: `permutation_importance.csv`

## Making Predictions

### Batch Prediction
```bash
python buy_predictor_test.py --input input_data.csv --output predictions.csv
```

### Interactive Prediction
```bash
python buy_predictor_cli.py
```

## Monitoring and Maintenance

### Performance Monitoring
- Check `buy_predictor_metrics.csv` for model performance metrics
- Monitor prediction distribution in `buy_predictor_predictions.csv`

### Retraining
1. Update the training data in `amazon_customers_annotated.csv`
2. Run the training command
3. Compare new metrics with previous performance
4. Deploy the new model if performance improves

## Troubleshooting

### Common Issues

#### Missing Dependencies
```
ModuleNotFoundError: No module named 'lightgbm'
```
**Solution**: Install the missing package:
```bash
pip install lightgbm
```

#### File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'cleaned_pre_imputation.csv'
```
**Solution**: Make sure to run the script from the project root directory.

#### Model Loading Error
```
_pickle.UnpicklingError: invalid load key, '<'.
```
**Solution**: The model file might be corrupted. Retrain the model.

## Performance Tuning

### Hyperparameter Tuning
Edit `buy_predictor.py` to modify the model parameters in the `train_lightgbm` and `train_svc` functions.

### Feature Engineering
- Add new features to `preprocess_data` function
- Update feature selection in `select_features`

## Backup and Recovery

### Regular Backups
1. Model files (`.joblib`)
2. Training data (`amazon_customers_annotated.csv`)
3. Configuration files

### Recovery Steps
1. Restore from the latest backup
2. Retrain the model if necessary
3. Verify model performance

## Security

### Data Protection
- Ensure input data is properly sanitized
- Store sensitive information in environment variables
- Use secure file permissions for model files

### Access Control
- Restrict write access to model files
- Use version control for code changes
- Maintain audit logs of model deployments
