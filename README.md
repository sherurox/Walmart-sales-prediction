# Walmart Sales Prediction

## Overview
This project aims to analyze and predict **weekly sales** across different Walmart stores using **machine learning techniques**. The dataset consists of various economic and seasonal factors that affect sales trends. The objective is to build a robust model that provides accurate sales forecasts for better business decision-making.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Selection & Training](#model-selection--training)
6. [Model Performance & Results](#model-performance--results)
7. [Future Improvements](#future-improvements)
8. [How to Run the Project](#how-to-run-the-project)
9. [Contributors](#contributors)

---

## Dataset Overview
- **Source:** Walmart Weekly Sales Dataset
- **Total Samples:** 6,435 (before processing)
- **Features:** 8 numerical and categorical variables
- **Target Variable:** `Weekly_Sales`

### Columns in the Dataset:
| Feature | Description |
|---------|-------------|
| `Store` | Store ID |
| `Date` | Date of the sales record |
| `Weekly_Sales` | Weekly sales revenue (target variable) |
| `Holiday_Flag` | 1 = Holiday week, 0 = Non-Holiday week |
| `Temperature` | Weekly average temperature |
| `Fuel_Price` | Fuel price during the week |
| `CPI` | Consumer Price Index |
| `Unemployment` | Unemployment rate |

---

## Data Preprocessing
1. **Handling Date Variable**
   - Converted `Date` to a datetime object.
   - Extracted `Year`, `Month`, and `Weekday` as new features.
   - Dropped the original `Date` column.

2. **Checking for Missing Values**
   - No missing values were found.

3. **Encoding Categorical Variables**
   - **One-Hot Encoding** for binary categorical features (`Holiday_Flag`).
   - **Dummy Encoding** for multi-category variables (`Year`, `Month`, `Weekday`, `Store`).

4. **Removing Duplicates**
   - No duplicate records were found in the dataset.

5. **Outlier Detection & Removal**
   - Used **Interquartile Range (IQR)** method to remove extreme values.
   - **Dataset reduced from 6,435 to 5,953 samples** after cleaning.

---

## Exploratory Data Analysis (EDA)
- **Distribution of Target Variable (`Weekly_Sales`)**: Identified seasonal trends and outliers.
- **Feature Correlation Analysis**: Used heatmaps to find relationships between variables.
- **Category Distributions**: Analyzed sales patterns based on stores, months, and holidays.

---

## Feature Engineering
- **Created new time-based features:**
  - `Year`, `Month`, `Weekday`
- **Dropped redundant features:**
  - `Date` column removed after extracting useful date components.
- **Feature Scaling:**
  - Used `StandardScaler` to normalize numerical features.

---

## Model Selection & Training
### **Train-Test Split**
- **Training Set:** 80% (4,762 samples)
- **Testing Set:** 20% (1,191 samples)

### **Machine Learning Models Used**
1. **Linear Regression (OLS Model)**
   - Standardized all numerical features.
   - Checked feature significance using p-values.
   - Removed highly collinear variables based on Variance Inflation Factor (VIF).

---

## Model Performance & Results
### **Key Metrics from Linear Regression**
| Metric | Train Value | Test Value |
|--------|------------|------------|
| **R-Squared (R²)** | 0.145 | 0.153 |
| **Root Mean Squared Error (RMSE)** | 521,024 | 522,253 |

### **Observations:**
- **R² value is low (~15%)**, indicating that the linear model does not capture all variability in sales.
- **RMSE values (~520K)** suggest large deviations in predictions, meaning improvements are needed.
- Some economic factors like **CPI, Fuel Price, and Unemployment Rate** were **insignificant** predictors of sales and were removed from the final model.

### **Visualizations Included:**
- Feature importance analysis (Coefficient analysis from the regression model).
- Residual distribution plot to analyze errors.
- Train vs. Test RMSE comparison to check model performance.

---

## Future Improvements
1. **Test Advanced Models**:
   - **Random Forest Regressor** or **XGBoost** for capturing non-linear relationships.
   - **Neural Networks (LSTM)** for time-series forecasting.

2. **Feature Engineering**:
   - Add external factors such as store promotions, seasonality trends, and holiday discounts.
   
3. **Hyperparameter Optimization**:
   - Tune parameters for **Ridge/Lasso Regression** to reduce overfitting.

4. **Time-Series Models**:
   - Try **ARIMA, Prophet, or Exponential Smoothing** for better forecasting.

---

## How to Run the Project
### **Requirements**
Ensure you have Python installed along with the required libraries.
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

### **Steps to Run**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/walmart-sales-prediction.git
   cd walmart-sales-prediction
   ```
2. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook Walmart_Sales_Prediction.ipynb
   ```
3. Review the generated **visualizations and model performance metrics**.

---

## Feature Importance (Linear Regression Coefficients)
<p align="center">
    <img src="https://github.com/user-attachments/assets/d6ff3076-db47-4d0d-8ab8-21889cc19bd1" width="600">
</p>

## Residual Distribution
<p align="center">
    <img src="https://github.com/user-attachments/assets/97a44ba8-e814-49b4-8ede-8ff94fc74c6b" width="600">
</p>

## Train Vs. Test RMSE
<p align="center">
    <img src="https://github.com/user-attachments/assets/746e15d6-a294-4573-bacd-ae7d7cd82d49" width="600">
</p>

## Model Performance Metrics

<p align="center">

| Metric | Train Value | Test Value |
|--------|------------|------------|
| **R-Squared (R²)** | 0.145 | 0.153 |
| **Root Mean Squared Error (RMSE)** | 521,024 | 522,253 |

</p>




---

## Contributors
- **Shreyas Khandale** https://github.com/sherurox
- **Rohan Upendra Patil**  https://github.com/rohanpatil2

---

### License
This project is licensed under the MIT License.

---

### Acknowledgments
- Thanks to Walmart for providing the dataset.
- Special thanks to the data science and machine learning community for valuable resources and insights.

---

## Final Notes
- This project provides a **baseline model** for Walmart sales prediction.
- Further **improvements and optimizations** are recommended for **better accuracy**.

