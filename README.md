### 🐧 Penguin Species Classification

This project focuses on predicting penguin species using machine learning techniques.
The dataset is based on the Palmer Penguins Dataset, extended with additional features for testing.

### Exploratory Data Analysis (EDA)
Checked missing values and outliers.
Visualized distributions of numerical features (bill length, bill depth, flipper length, body mass).
Investigated correlations and species distributions.

### Preprocessing
Removed outliers using the IQR method.
Encoded categorical features with both:
Label Encoding (baseline)
One-Hot Encoding (better performance)
Applied StandardScaler to numerical features.

### Modeling
Logistic Regression: Used as a baseline model.
XGBoost: Main model, provided the best performance.
Compared results between Label Encoding and One-Hot Encoding
Feature importance analysis showed that both biological measures and categorical data contribute to classification.
Final model: XGBoost + One-Hot Encoding, saved for prediction.

''
Model is saved in saved_models/xgb_onehot.json.
Predictions are made via terminal input:
python -m model.predict
''
