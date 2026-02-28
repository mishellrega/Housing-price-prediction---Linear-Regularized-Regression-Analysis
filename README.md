# King County Residential Property Price Modeling

## Project Overview

This project analyzes residential property sales data from King County (Seattle area) to develop predictive models for estimating house prices based on structural and location-based features.

The objective is to evaluate how property characteristics such as square footage, grade, bathrooms, waterfront status, and location influence price, and to compare different regression approaches in terms of explanatory power and generalization performance.

This analysis simulates a real-world scenario in which a Real Estate Investment Trust (REIT) seeks to support acquisition and pricing decisions using data-driven valuation models.

---

## Dataset Description

The dataset contains residential property sales between May 2014 and May 2015 in King County, Washington.

Key variables include:

- `price` – Sale price (target variable)
- `sqft_living` – Total living area
- `sqft_above` – Above-ground living area
- `bedrooms`
- `bathrooms`
- `floors`
- `grade` – Construction and design quality rating
- `waterfront` – Binary indicator
- `view`
- `lat` and `long` – Geographic location
- Additional structural attributes

Missing values in bedrooms and bathrooms were imputed using mean substitution.

---

## Exploratory Analysis

### Structural Drivers of Price

- `sqft_living` shows the strongest linear correlation with price (~0.70).
- `grade` and `sqft_above` also demonstrate strong positive correlations.
- `bathrooms` and `sqft_living15` provide additional explanatory value.

### Waterfront Effect

Boxplot analysis shows a substantial price premium for waterfront properties.  
Waterfront homes exhibit both higher median prices and greater price dispersion.

### Above-Ground Living Area

Regression analysis between `sqft_above` and price confirms a clear positive linear relationship, indicating that usable living space above ground significantly impacts valuation.

---

## Modeling Approach

Four modeling strategies were developed and compared:

### 1. Simple Linear Regression  
Predictor: `sqft_living`  
R² ≈ 0.49  

Approximately 49% of the variability in housing prices is explained by total living area alone.

---

### 2. Multiple Linear Regression  
Predictors: structural and location features  

R² ≈ 0.66  

Including additional property characteristics improves explanatory power significantly.

---

### 3. Ridge Regression (Test Set Evaluation)  
R² ≈ 0.65  

Regularization stabilizes the model and maintains strong performance on unseen data.

---

### 4. Polynomial + Ridge Regression (Degree 2)  
R² ≈ 0.65 (test set)

Introducing second-order polynomial features does not significantly improve predictive performance, suggesting that the relationship between features and price is largely linear in nature.

---

## Model Comparison Summary

| Model | R² |
|--------|------|
| Linear (sqft_living only) | ~0.49 |
| Linear (multiple features) | ~0.66 |
| Ridge (test set) | ~0.65 |
| Polynomial Ridge (test set) | ~0.65 |

The multiple linear regression and Ridge models provide the best balance between explanatory strength and generalization.

---

## Business Interpretation

The analysis suggests:

- Total living area is the strongest individual predictor of property value.
- Construction grade significantly influences price, reflecting quality premiums.
- Waterfront status commands a substantial valuation premium.
- Most relationships appear approximately linear, as polynomial expansion did not materially improve predictive performance.
- Regularization improves model stability without sacrificing accuracy.

From an investment perspective, these findings support a valuation framework centered on structural size, quality grade, and high-impact amenities such as waterfront access.

---

## Conclusion

The final Ridge regression model explains approximately 65% of the variance in housing prices on unseen data. This level of predictive accuracy is strong for a real estate pricing model based on structural and geographic features.

The model can support:

- Initial acquisition screening
- Price benchmarking
- Risk-adjusted valuation strategies
- Investment decision support

---

## Tools & Libraries

- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn

---

**Author:**  
Michelle Regalado  
Economist | Data Analytics & Predictive Modeling