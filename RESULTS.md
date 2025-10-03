# Amazon Customer Segmentation - Analysis Results

## Executive Summary
This analysis segments Amazon customers into distinct groups based on shopping behavior and preferences using K-means clustering with PCA. The optimal number of clusters was determined to be 2, providing meaningful segmentation while maintaining interpretability.

## Clustering Performance

### Model Evaluation Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | 0.094 | Slight separation between clusters |
| Davies-Bouldin Index | 2.91 | Lower values indicate better separation |
| Calinski-Harabasz Index | 69.49 | Higher values indicate better defined clusters |

### Cluster Distribution
| Cluster | Customers | Percentage |
|---------|-----------|------------|
| 0 | 303 | 50.3% |
| 1 | 299 | 49.7% |

## Cluster Profiles

### Cluster 0: Beauty-Focused Shoppers (50.3%)
- **Average Age**: 31.1 years
- **Top Purchase Categories**:
  1. Beauty and Personal Care (26.7%)
  2. Clothing and Fashion (22.4%)
  3. Multi-category purchases (20.1%)

### Cluster 1: Diverse Category Shoppers (49.7%)
- **Average Age**: 30.5 years
- **Key Characteristics**:
  - More diverse purchasing patterns
  - Higher likelihood of bundled purchases
  - Strong preference for multi-category combinations

## Statistical Significance

### Significant Differences (p < 0.05)
1. **Shopping Satisfaction** (p = 9.27e-86)
   - Strong statistical significance between clusters
   - Indicates meaningful differences in customer satisfaction levels

2. **Customer Reviews Importance** (p = 1.51e-51)
   - Clusters differ in how they value customer reviews
   - Important for review-based marketing strategies

3. **Rating Accuracy** (p = 1.75e-51)
   - Clusters perceive product ratings differently
   - Impacts how ratings should be presented to each segment

### Non-Significant Differences
- **Age** (p = 0.53)
  - Age is not a significant differentiator between clusters
  - Segments are primarily behaviorally defined

## Business Implications

### Marketing Recommendations
1. **Targeted Campaigns**:
   - Cluster 0: Focus on beauty and personal care bundles
   - Cluster 1: Emphasize cross-category promotions

2. **Review Strategy**:
   - Cluster 0: Highlight top-rated products in beauty categories
   - Cluster 1: Showcase verified purchase reviews for multiple categories

3. **Product Recommendations**:
   - Cluster 0: Personalized beauty product suggestions
   - Cluster 1: "Frequently bought together" recommendations

## Technical Implementation

### Data Processing Pipeline
1. **Preprocessing**:
   - Handling missing values
   - Feature scaling (StandardScaler)
   - Categorical variable encoding

2. **Dimensionality Reduction**:
   - PCA for feature reduction
   - Explained variance analysis

3. **Clustering**:
   - K-means algorithm
   - Optimal K determination using elbow method
   - Cluster validation using silhouette analysis

### Model Artifacts
- `outputs/kmeans.joblib`: Trained K-means model
- `outputs/pca.joblib`: Fitted PCA transformer
- `outputs/preprocessor_final.joblib`: Data preprocessing pipeline

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
