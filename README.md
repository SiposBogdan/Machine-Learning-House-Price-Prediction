# 🏡 California Housing Price Prediction Models

This project implements a **machine learning regression pipeline** to predict **median house values** using demographic and geographical data from the **California Housing dataset**. The notebook explores multiple regression algorithms, data preprocessing techniques, and evaluation metrics for model comparison.

---

## 📘 Project Overview

The primary objective is to predict **median_house_value** based on various housing attributes, including:
- **median_income**
- **total_rooms**, **total_bedrooms**
- **housing_median_age**
- **ocean_proximity**
- **population**, **households**, and more

Dataset used:  
`data/housing/housing.csv`

### Workflow:
1. **Data loading and exploration**
2. **Handling missing values with KNN imputation**
3. **Categorical encoding with OneHotEncoder**
4. **Feature scaling and engineering**
5. **Model training and comparison**
6. **Evaluation using multiple regression metrics**

---

## 🧹 Data Preprocessing

Key transformations include:
- **Missing Values:** Filled using `KNNImputer` (k=5)
- **Categorical Encoding:** Applied `OneHotEncoder` to `ocean_proximity`
- **Feature Engineering:** Added ratio features like:
  - `bedrooms_per_house`
  - `people_per_house`
  - `rooms_per_house`
- **Scaling:** Used `StandardScaler` for numerical features
- **Data Split:** Stratified using `train_test_split` into training and test sets

---

## 🧠 Models Implemented

| Model | Library | Description |
|--------|----------|-------------|
| **Linear Regression** | `sklearn.linear_model` | Baseline linear model for continuous prediction. |
| **Decision Tree Regressor** | `sklearn.tree` | Non-linear model that splits data based on decision rules. |
| **Random Forest Regressor** | `sklearn.ensemble` | Ensemble of decision trees to reduce overfitting and improve generalization. |
| **K-Nearest Neighbors Regressor (KNN)** | `sklearn.neighbors` | Predicts target by averaging nearest samples. |
| **Support Vector Regressor (SVR)** | `sklearn.svm` | Regression model using hyperplane fitting with kernels. |
| **XGBoost Regressor** | `xgboost` | High-performance gradient boosting model. |

---

## ⚙️ Model Training and Evaluation

### Cross Validation
Used `cross_val_score` for reliable performance estimation.

### Hyperparameter Optimization
Performed with both `GridSearchCV` and `RandomizedSearchCV` for optimal parameter tuning.

### Evaluation Metrics
Models were evaluated using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

## 📊 Results

Models were compared on MAE and RMSE.  
Typically, ensemble models (e.g., **Random Forest**, **XGBoost**) achieved the highest accuracy, while **Linear Regression** served as the baseline benchmark.

---



**Main Libraries:**
- `scikit-learn`
- `xgboost`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
