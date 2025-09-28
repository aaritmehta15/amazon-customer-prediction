# Amazon Customer Purchase Prediction

This project analyzes customer behavior from Amazon purchase data to predict the likelihood of future purchases. It includes data preprocessing, exploratory analysis, clustering of customer segments, and a predictive model for purchase intent.

## Project Structure

```
ML MINI PROJ/
├── buy_predictor/           # Prediction module
│   ├── buy_predictor.py     # Main training script
│   ├── buy_predictor_cli.py # Interactive prediction tool
│   ├── buy_predictor_test.py# File-based prediction tool
│   └── README.md            # Module-specific documentation
├── outputs/                 # Analysis outputs and artifacts
│   ├── clustering_results_summary.json
│   ├── cleaned_pre_imputation.csv
│   └── permutation_importance.csv
├── RUNBOOK.md              # Detailed how-to guide
├── RESULTS.md              # Analysis and model results
└── DATA_DICTIONARY.md      # Feature documentation
```

## Prerequisites

- **Git** (for cloning the repository)
  - Download: [git-scm.com](https://git-scm.com/downloads)
- **Python 3.8 or later**
  - Download: [python.org/downloads](https://www.python.org/downloads/)
  - During installation, check "Add Python to PATH"
- **pip** (Python package manager)
  - Usually comes with Python installation

## Quick Start

1. **Setup Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   .\.venv\Scripts\activate
   
   # Install dependencies
   pip install numpy pandas scikit-learn lightgbm joblib
   ```

2. **Train Model**
   ```bash
   python buy_predictor\buy_predictor.py
   ```

3. **Make Predictions**
   - Interactive mode:
     ```bash
     python buy_predictor\buy_predictor_cli.py
     ```
   - File-based mode:
     ```bash
     python buy_predictor\buy_predictor_test.py --input-csv your_data.csv
     ```

## Documentation

- [RUNBOOK.md](RUNBOOK.md): Complete guide to running and using the project
- [RESULTS.md](RESULTS.md): Analysis of model performance and findings
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md): Detailed description of all features
- [buy_predictor/README.md](buy_predictor/README.md): Prediction module documentation

## Key Features

- Customer segmentation using K-means clustering
- Purchase prediction using LightGBM and SVC models
- Interactive CLI for real-time predictions
- Comprehensive model evaluation metrics

## Core Dependencies

- numpy
- pandas
- scikit-learn
- lightgbm
- joblib

Install all dependencies with:
```bash
pip install numpy pandas scikit-learn lightgbm joblib
```

Or create a `requirements.txt` file and use:
```bash
pip install -r requirements.txt
```

## License

This project is for educational purposes.
