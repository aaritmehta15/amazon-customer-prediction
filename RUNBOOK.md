# Amazon Customer Segmentation - Runbook

## Table of Contents
- [Environment Setup](#environment-setup)
- [Running the Analysis](#running-the-analysis)
- [Output Files](#output-files)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)
- Jupyter Notebook

### Installation

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Analysis

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open and run the notebook**:
   - Navigate to `clustering-amazon-customers-pca-k-means.ipynb`
   - Run all cells (Kernel > Restart & Run All)

## Output Files

### Visualizations
- `outputs/elbow.png`: Elbow method plot showing optimal number of clusters (2)
- `outputs/silhouette.png`: Silhouette analysis for cluster validation
- `outputs/pca_clusters.png`: 2D visualization of customer segments

### Data Files
- `outputs/cleaned_pre_imputation.csv`: Preprocessed dataset
- `outputs/amazon_customers_annotated.csv`: Dataset with cluster assignments

### Analysis Results
- `outputs/clustering_results_summary.json`: Complete clustering metrics
- `outputs/cluster_profile.json`: Detailed cluster characteristics
- `outputs/stat_tests_*.json`: Statistical test results for cluster differences

## Interpreting Results

### Cluster 0 (50.3% of customers)
- **Average Age**: 31.1 years
- **Top Categories**:
  - Beauty and Personal Care (26.7%)
  - Clothing and Fashion (22.4%)
  - Combination categories (20.1%)

### Cluster 1 (49.7% of customers)
- **Average Age**: 30.5 years
- **Top Categories**:
  - More diverse category preferences
  - Higher preference for multi-category purchases
  - Stronger preference for bundled purchases

### Key Differences (p < 0.05)
1. **Shopping Satisfaction**: Significant difference between clusters
2. **Customer Reviews Importance**: Notable variation
3. **Rating Accuracy**: Statistically significant difference
4. **Age**: Not a significant differentiator (p = 0.53)

## Troubleshooting

### Common Issues
1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Kernel Issues**:
   - Restart kernel if cells don't execute
   - Clear all outputs and restart if needed

3. **File Not Found**:
   - Ensure working directory is set to project root
   - Check file paths in the notebook

### Getting Help
For additional assistance, please refer to the Jupyter notebook documentation or open an issue in the project repository.

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
