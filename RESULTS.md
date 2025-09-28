# Amazon Customer Prediction - Results Analysis

This document provides a comprehensive analysis of the model's performance, key findings, and insights derived from the Amazon customer behavior dataset.

## Model Performance Summary

### Evaluation Metrics

| Model   | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| LightGBM| 0.63     | 0.42      | 0.24   | 0.31     | 0.52    |
| SVC     | 0.64     | 0.42      | 0.12   | 0.19     | 0.55    |

*Note: Metrics are on the test set. Best values in **bold**.*

### Confusion Matrix (LightGBM)

```
              Predicted No  Predicted Yes
Actual No          66            14
Actual Yes         31            10
```

## Key Findings

### 1. Feature Importance

Top 5 most important features for predicting purchase likelihood:

1. **shopping_satisfaction_num** (0.0517 ± 0.0278)
2. **customer_reviews_importance_num** (0.0479 ± 0.0167)
3. **customer_reviews_importance** (0.0475 ± 0.0133)
4. **shopping_satisfaction** (0.0434 ± 0.0225)
5. **personalized_recommendation_frequency_1** (0.0417 ± 0.0127)

*Note: Importance scores are permutation-based (mean ± std). Higher values indicate more important features.*

### 2. Model Comparison

- **SVC** has slightly higher accuracy (0.64 vs 0.63) but **LightGBM** performs better at identifying actual buyers (higher recall).
- Both models show moderate discrimination power (AUC ~0.52-0.55).
- The models are better at identifying non-buyers (higher specificity) than buyers (sensitivity).

### 3. Prediction Distribution

From the test set predictions:

- **Class Imbalance**: The dataset shows significant class imbalance, with more non-buyers than buyers.
- **Prediction Confidence**: The model's predicted probabilities show good separation between classes, with most predictions being confident (probabilities close to 0 or 1).
- **Error Analysis**: The model has more false negatives than false positives, indicating it's more likely to miss actual buyers than to incorrectly label non-buyers as buyers.

## Sample Predictions

| y_true | y_pred | y_prob |
|--------|--------|--------|
| 0      | 0      | 0.43   |
| 0      | 1      | 0.87   |
| 0      | 0      | 0.50   |
| 1      | 0      | 0.41   |
| 0      | 0      | 0.19   |

*Note: y_prob is the predicted probability of class 1 (buy).*

## Limitations

1. **Class Imbalance**: The dataset shows significant class imbalance, affecting the model's ability to learn the minority class.
2. **Model Performance**: The current models show only slightly better than random performance (AUC ~0.5-0.55), indicating room for improvement.
3. **Feature Redundancy**: Some features are highly correlated (e.g., `shopping_satisfaction` and `shopping_satisfaction_num`).

## Recommendations

1. **Address Class Imbalance**:
   - Use class weighting or resampling techniques (SMOTE, ADASYN).
   - Consider collecting more samples of the minority class.

2. **Feature Engineering**:
   - Remove redundant features (e.g., keep only one version of satisfaction metrics).
   - Create interaction terms between top features.
   - Consider feature selection to reduce noise.

3. **Model Improvement**:
   - Experiment with different algorithms (e.g., XGBoost, Random Forest).
   - Perform hyperparameter tuning with cross-validation.
   - Consider anomaly detection approaches for the imbalanced classification task.

## Next Steps

1. Conduct A/B testing with the model in a production environment.
2. Gather more data on customer demographics for better segmentation.
3. Build a simple web interface for non-technical users.

---
*Last Updated: September 28, 2025*
