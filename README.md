# Loan Credit Risk Evaluation System

## Table of Contents

- [Project Description](#project-description)  
- [Problem Statement](#problem-statement)  
- [Literature Survey](#literature-survey)  
- [Tech Stack & Libraries Used](#tech-stack--libraries-used)  
- [Data & Features](#data--features)  
- [Alternate Models & Their Disadvantages](#alternate-models--their-disadvantages)  
- [System Architecture & Workflow](#system-architecture--workflow)  
- [Modeling & Hyperparameter Tuning](#modeling--hyperparameter-tuning)  
- [Streamlit GUI](#streamlit-gui)  
- [Evaluation Metrics & Results](#evaluation-metrics--results)  
- [Conclusion & Future Scope](#conclusion--future-scope)  
- [Folder Structure](#folder-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)

---

## Project Description

The **Loan Credit Risk Evaluation System** is designed to help financial institutions make accurate, fair, and fast loan approval decisions. By using historical data on previous loan applications, the system classifies new applicants as **high-risk** (likely to default) or **low-risk** (likely to repay). A web interface (via Streamlit) allows loan officers or users to input applicant details and quickly receive predictions.

The full pipeline includes:

- Data cleaning, preprocessing, feature engineering  
- Trying out different machine learning models (Logistic Regression, Random Forest, XGBoost)  
- Handling class imbalance  
- Hyperparameter tuning  
- Model evaluation  
- Deployment through a user-friendly interface  

---

## Problem Statement

- Increase in volume of loan applications (personal, education, auto, business) poses challenges for timely, consistent evaluation.  
- Incorrectly approving high‐risk applicants causes financial loss; wrongly rejecting good applicants harms business and customer trust.  
- The goals are:  
  1. Analyze historical loan application data  
  2. Build ML models to classify applicants as *low risk* (approve) or *high risk* (reject)  
  3. Reduce manual effort, speed up decisions  
  4. Provide a usable real-time interface for credit risk prediction  

---

## Literature Survey

### Key Models Compared:

1. **Logistic Regression**  
   - Pros: Simplicity, interpretability; useful baseline.  
   - Cons: Assumes linear relationships; struggles with non-linear interactions; sensitive to multicollinearity.

2. **Random Forest**  
   - Pros: Captures non-linear relationships; robust to overfitting; handles categorical & numerical data.  
   - Cons: Less interpretable; can be more memory / compute intensive.

3. **XGBoost (Extreme Gradient Boosting)**  
   - Pros: Very strong predictive performance; handles missing values; supports regularization; can scale well.  
   - Cons: More complex; less immediately interpretable without additional tools like SHAP or LIME.

### Other Alternate Models Explored (and Why Not Chosen):

- **Support Vector Machine (SVM)** – Good for high dimensions but computationally expensive, limited interpretability.  
- **K-Nearest Neighbors (KNN)** – Simple but slow for large datasets; sensitive to noise and irrelevant features.  
- **Naive Bayes** – Fast but strong assumptions (feature independence) rarely hold; less accurate in complex, correlated feature spaces.

---

## Tech Stack & Libraries Used

- **Language**: Python  
- **Big Data / Parallel Processing**: PySpark  
- **Libraries**:
  - `pandas` – data manipulation, cleaning  
  - `numpy` – numerical operations  
  - `matplotlib`, `seaborn` – visualizations (histograms, boxplots, heatmaps etc.)  
  - `scikit-learn` – preprocessing, baseline models, evaluation metrics, hyperparameter search  
  - `xgboost` – main model for classification  
  - `joblib` – saving / loading trained models  
  - `streamlit` – web application for user interface  

---

## Data & Features

- Dataset includes features like:  
  - Income, loan amount, credit history, employment type, education level, demographics  
  - Engineered features such as loan-to-income ratio, credit term etc.  

- Data preprocessing steps included:  
  1. Handling missing values  
  2. Encoding categorical variables (label encoding / one-hot encoding)  
  3. Scaling or normalizing numerical features if necessary  
  4. Handling class imbalance (e.g. by computing class weights)  

---

## Alternate Models & Their Disadvantages

| Model | Disadvantages in this project context |
|---|---|
| Support Vector Machine (SVM) | High computational cost on large dataset; poor interpretability; not directly giving probabilistic outputs. |
| K-Nearest Neighbors (KNN) | Slow prediction with large data; sensitive to noisy or irrelevant features; large memory requirement. |
| Naive Bayes | Assumes feature independence (often unrealistic in financial data); struggles with correlated or complex features; usually less accurate than tree-based models. |

---

## System Architecture & Workflow

### Architecture Overview

```
+------------------+ +-------------------+ +------------------+
| Raw Dataset | → | Pre-processing & | → | Feature Engineering |
+------------------+ | Cleaning, Encoding,| | e.g. loan-to-income |
| Handling Missing, | | ratio, etc. |
+-------------------+ +------------------+
|
v
+-------------------------+
| Train / Validate Models |
+-------------------------+
|
v
+------------------------+
| Model Selection & |
| Hyperparameter Tuning |
+------------------------+
|
v
+------------------------+
| Deployment via Streamlit|
+------------------------+
|
v
+-------------------------------+
| Real-time Prediction Interface |
+-------------------------------+
```


---

## Project Workflow

Here is the step-by-step workflow the team followed:

1. **Data Acquisition** – Obtain historical loan application data.  
2. **Exploratory Data Analysis (EDA)** – Visualize distributions, correlations; uncover outliers; decide on feature transformations.  
3. **Feature Engineering** – Create derived variables like loan-to-income ratio, credit term; encode categorical variables.  
4. **Data Splitting** – Split into training and test sets (e.g. 80/20) using PySpark / sklearn.  
5. **Model Training & Comparison** – Train logistic regression, random forest, and XGBoost.  
6. **Handling Class Imbalance** – Use class weights or other imbalance strategies.  
7. **Hyperparameter Tuning** – For XGBoost (and possibly others) using RandomizedSearchCV.  
8. **Model Evaluation** – Using metrics like AUC, Accuracy, Precision, Recall, F1-Score.  
9. **Model Persistence** – Save the best model (e.g. tuned XGBoost) using joblib.  
10. **User Interface / Deployment** – Build a Streamlit app to allow user input and real-time prediction.  
11. **Visualization & Reporting** – Present EDA, model performance plots, feature importance etc.

---
