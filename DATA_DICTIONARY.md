# Data Dictionary

This document provides detailed information about the features used in the Amazon Customer Prediction project.

## Customer Profile Features

| Feature Name | Type | Description | Possible Values |
|--------------|------|-------------|-----------------|
| gender | Categorical | Customer's gender | Male, Female, Prefer not to say |
| age | Numeric | Customer's age | Integer values |
| shopping_satisfaction | Categorical | Self-reported shopping satisfaction | 1-5 (Likert scale) |
| shopping_satisfaction_num | Numeric | Numeric version of shopping satisfaction | 1.0-5.0 |
| customer_reviews_importance | Categorical | Importance of customer reviews in purchase decisions | 1-5 (Likert scale) |
| customer_reviews_importance_num | Numeric | Numeric version of review importance | 1.0-5.0 |
| personalized_recommendation_frequency | Categorical | How often personalized recommendations are useful | 1-5 (Likert scale) |

## Review and Rating Features

| Feature Name | Type | Description | Possible Values |
|--------------|------|-------------|-----------------|
| rating_accuracy | Categorical | Accuracy of product ratings | 1-5 (Likert scale) |
| rating_accuracy_num | Numeric | Numeric version of rating accuracy | 1.0-5.0 |
| review_helpfulness | Categorical | Helpfulness of reviews | Yes, No |
| review_helpfulness_Yes | Binary | Binary indicator for helpful reviews | 0, 1 |

## Topic Modeling Features

| Feature Name | Type | Description |
|--------------|------|-------------|
| timestamp_topic_0 | Numeric | Topic 0 weight from timestamp analysis |
| timestamp_topic_1 | Numeric | Topic 1 weight from timestamp analysis |
| ... | ... | ... |

## Target Variable

| Feature Name | Type | Description |
|--------------|------|-------------|
| buy | Binary | Whether the customer made a purchase (1) or not (0) |

## Derived Features

| Feature Name | Type | Description |
|--------------|------|-------------|
| improvement_areas_* | Binary | One-hot encoded improvement areas from feedback |
| gender_* | Binary | One-hot encoded gender categories |

## Notes
- All numeric features are standardized (z-score normalized)
- Categorical features are one-hot encoded for model input
- Missing values are imputed with appropriate strategies (mean for numeric, mode for categorical)
