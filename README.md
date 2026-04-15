# Retail_Sales_Predictor

# Project Overview
This project focuses on predicting retail sales for the Big Mart dataset using machine learning. My goal was to build a robust pipeline, experiment with feature engineering, and evaluate multiple models to maximize predictive accuracy.

I faced challenges with accuracy early on, where baseline models struggled to explain more than ~57–60% of the variance. Through iterative experimentation, feature engineering, and model tuning, I improved performance significantly.

# Challenges Faced
Accuracy Plateau: Initial models such as Linear Regression and Decision Trees produced R² scores around 0.55–0.60, highlighting underfitting.
Feature Noise: Raw features like Item_Visibility and Item_MRP introduced noise and did not capture business meaning.
Categorical Inconsistencies: Labels such as LF, low fat, and Low Fat required cleaning to avoid misleading encodings.
Model Bias: Linear models could not capture nonlinear retail dynamics, limiting accuracy.

# Feature Engineering
To overcome these issues, I applied several transformations:
Outlet Age: Derived from Outlet_Establishment_Year to capture the effect of outlet maturity.
Item MRP Binning: Grouped continuous MRP values into four price bands (Low, Medium, High, Very High) to reflect pricing tiers.
Item Visibility Binning: Converted continuous visibility into categorical bins (Low, Medium, High, Very High) to reduce noise.
Log Transformation: Applied np.log1p to the target variable (Item_Outlet_Sales) to stabilize variance and improve model fit.

These engineered features provided clearer business context and improved model interpretability.

# Model Evaluation
I tested multiple models to identify the best performer:

Linear Regression: R² = 0.7230
SVR: R² = 0.7206
XGBoost (default): R² = 0.6948
SGD: R² = 0.7207
Decision Tree: R² = 0.4677
Random Forest (default): R² = 0.6986

# Hyperparameter Tuning
Using GridSearchCV, I tuned Random Forest and XGBoost:

Best parameters for XGBoost:
learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8
Best CV R²: 0.7160
Tuned Test R²: 0.7309

Best parameters for Random Forest:
max_depth=5, min_samples_leaf=1, min_samples_split=10, n_estimators=300
Best CV R²: 0.7147
Tuned Test R²: 0.7307

# Results
Through feature engineering and model tuning, I improved accuracy from ~0.57 baseline to 0.73 R² on the test set. This demonstrates the importance of:
-> Cleaning categorical variables.
-> Transforming continuous features into meaningful bins.
-> Using outlet age as a numeric predictor.
-> Applying log transformation to stabilize target variance.
-> Leveraging ensemble methods with tuned hyperparameters.

