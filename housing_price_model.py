
"""
King County Housing Price Modeling
Author: Michelle Regalado

Goal: Build and compare regression models to predict residential house prices in
King County (Seattle area) and identify the main drivers of price variation.

Business Context:
----------------
A Real Estate Investment Trust (REIT) is interested in entering the
residential housing market. The objective is to use historical sales data
to understand how features such as square footage, number of bedrooms,
location, and quality grade influence sale price, and to build predictive
models that can support acquisition and pricing decisions.
"""

# =========================================================
# 1. Imports
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# =========================================================
# 2. Data Loading & Basic Info
# =========================================================

# Dataset
filepath = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv"
)

df = pd.read_csv(filepath, index_col=0)

print("Raw data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics (before cleaning):")
print(df.describe())


# =========================================================
# 3. Data Wrangling
# =========================================================

# Drop ID column 
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

print("\nMissing values per column (before imputation):")
print(df.isnull().sum())

# Impute missing values in bedrooms and bathrooms with their mean
mean_bedrooms = df["bedrooms"].mean()
mean_bathrooms = df["bathrooms"].mean()

df["bedrooms"].replace(np.nan, mean_bedrooms, inplace=True)
df["bathrooms"].replace(np.nan, mean_bathrooms, inplace=True)

print("\nMissing values per column (after imputation):")
print(df.isnull().sum())

print("\nSummary statistics (after cleaning):")
print(df.describe())


# =========================================================
# 4. Exploratory Data Analysis (EDA)
# =========================================================

# Distribution of floors
floors_counts = df["floors"].value_counts().to_frame(name="count")
print("\nNumber of houses by floors:")
print(floors_counts)

# Boxplot: price by waterfront
plt.figure(figsize=(6, 4))
sns.boxplot(x="waterfront", y="price", data=df)
plt.title("House Prices by Waterfront Status")
plt.tight_layout()
plt.show()

# Regression plot: sqft_above vs price
plt.figure(figsize=(6, 4))
sns.regplot(x="sqft_above", y="price", data=df)
plt.title("Price vs Above-Ground Living Area (sqft_above)")
plt.tight_layout()
plt.show()

# Correlation with price
df_numeric = df.select_dtypes(include=[np.number])
corr_with_price = df_numeric.corr()["price"].sort_values()
print("\nCorrelation of numeric features with price:")
print(corr_with_price)


# =========================================================
# 5. Model Development (Full Data)
# =========================================================

# ---- Model 1: Simple Linear Regression with sqft_living ----
X_simple = df[["sqft_living"]]
y = df["price"]

lm_simple = LinearRegression()
lm_simple.fit(X_simple, y)
r2_simple = lm_simple.score(X_simple, y)
print(f"\nR² (Linear, sqft_living only): {r2_simple:.3f}")

# ---- Model 2: Multiple Linear Regression with selected features ----
features = [
    "floors",
    "waterfront",
    "lat",
    "bedrooms",
    "sqft_basement",
    "view",
    "bathrooms",
    "sqft_living15",
    "sqft_above",
    "grade",
    "sqft_living",
]

X = df[features]

lm_multi = LinearRegression()
lm_multi.fit(X, y)
r2_multi = lm_multi.score(X, y)
print(f"R² (Linear, multiple features): {r2_multi:.3f}")


# ---- Model 3: Polynomial Regression via Pipeline (Full Data) ----
pipeline_steps = [
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression()),
]

pipe = Pipeline(pipeline_steps)
pipe.fit(X, y)
y_pipe = pipe.predict(X)
r2_poly_full = r2_score(y, y_pipe)
print(f"R² (Polynomial Regression, degree=2, full data): {r2_poly_full:.3f}")


# =========================================================
# 6. Train/Test Split & Ridge Regression
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=1
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ---- Model 4: Ridge Regression (linear features only) ----
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

yhat_ridge = ridge_model.predict(X_test)
r2_ridge = r2_score(y_test, yhat_ridge)
rmse_ridge = mean_squared_error(y_test, yhat_ridge, squared=False)

print(f"\nR² (Ridge, linear features, test set): {r2_ridge:.3f}")
print(f"RMSE (Ridge, linear features, test set): {rmse_ridge:,.0f}")


# =========================================================
# 7. Polynomial Ridge Regression (Degree 2, Test Evaluation)
# =========================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

ridge_poly = Ridge(alpha=0.1)
ridge_poly.fit(X_train_poly, y_train)

yhat_poly = ridge_poly.predict(X_test_poly)
r2_poly = r2_score(y_test, yhat_poly)
rmse_poly = mean_squared_error(y_test, yhat_poly, squared=False)

print(f"\nR² (Ridge + Polynomial degree=2, test set): {r2_poly:.3f}")
print(f"RMSE (Ridge + Polynomial degree=2, test set): {rmse_poly:,.0f}")


# =========================================================
# 8. Model Comparison Summary
# =========================================================

print("\n=== Model Performance Summary ===")
print(f"1) Linear (sqft_living only, full data)          R² = {r2_simple:.3f}")
print(f"2) Linear (multiple features, full data)         R² = {r2_multi:.3f}")
print(f"3) Polynomial (degree=2, full data)              R² = {r2_poly_full:.3f}")
print(f"4) Ridge (linear features, test set)             R² = {r2_ridge:.3f},  RMSE = {rmse_ridge:,.0f}")
print(f"5) Ridge + Polynomial deg=2 (test set)           R² = {r2_poly:.3f},   RMSE = {rmse_poly:,.0f}")