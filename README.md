
# 📊 Credit Risk Modeling Project – Codebasics ML Course

This project builds a **credit risk prediction model** using customer, loan, and bureau data. It follows a complete machine learning pipeline from data preprocessing to model evaluation and deployment.

---

## 📁 Dataset
- `customers.csv`
- `loans.csv`
- `bureau_data.csv`

These are merged on `cust_id` to form the final dataset.

---

## 🔧 Key Steps in the Code

### 📌 1. **Data Loading & Merging**
- Loads customer, loan, and bureau data.
- Merges them into a single DataFrame.

### 📌 2. **Train-Test Split**
- Splits data before EDA to avoid leakage.
- Stratified split on the `default` target variable.

### 📌 3. **Data Cleaning**
- Handles missing values (e.g., `residence_type`).
- Removes outliers (e.g., high `processing_fee`).
- Applies business rules (e.g., GST < 20%).

### 📌 4. **Exploratory Data Analysis (EDA)**
- Visualizes distributions and outliers using boxplots and KDE plots.
- Identifies strong predictors like:
  - `loan_tenure_months`
  - `delinquent_months`
  - `total_dpd`
  - `credit_utilization_ratio`

### 📌 5. **Feature Engineering**
- Creates new features:
  - `loan_to_income`
  - `delinquency_ratio`
  - `avg_dpd_per_delinquency`

### 📌 6. **Feature Selection**
- Removes irrelevant columns (IDs, dates).
- Uses:
  - **VIF** for multicollinearity
  - **Information Value (IV)** for predictive power

### 📌 7. **Encoding & Scaling**
- One-hot encodes categorical variables.
- Scales numeric features using MinMaxScaler.

### 📌 8. **Model Training**
- Trains multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Handles class imbalance using:
  - Undersampling
  - SMOTE-Tomek

### 📌 9. **Hyperparameter Tuning**
- Uses:
  - `RandomizedSearchCV` for Logistic & XGBoost
  - `Optuna` for advanced tuning

### 📌 10. **Model Evaluation**
- Evaluates using:
  - Classification report
  - ROC-AUC
  - KS Statistic
  - Gini Coefficient
  - Rank ordering

### 📌 11. **Model Finalization**
- Selects best Logistic Regression model.
- Saves model and preprocessing artifacts using `joblib`.

---

## 📦 Output
- Trained model saved as: `artifacts/model_data.joblib`

---

## 📈 Performance
- **AUC**: 0.98
- **Gini Coefficient**: 0.96
- **KS Statistic**: 85.98%

---

