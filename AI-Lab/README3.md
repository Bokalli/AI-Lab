# Penguin Dataset Analysis and Classification

This project involves data cleaning, feature engineering, and classification on the **penguins dataset**. It uses *
*Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn** for various data preprocessing, visualization, and machine
learning tasks.

## Features and Steps

1. **Data Cleaning**:
    - Handling missing values by either dropping or imputing them.
    - Removing outliers based on the IQR method.

2. **Data Visualization**:
    - Boxplots to identify outliers for different features.
    - Correlation matrix to identify relationships between features.

3. **Feature Engineering**:
    - Creating new features such as average body mass and size categories based on body mass.

4. **Normalization**:
    - Min-Max scaling to normalize selected numerical columns.

5. **Machine Learning**:
    - Training a **Random Forest** classifier to assess feature importance.
    - Splitting the data into training and test sets for model validation.
    - Training a **Logistic Regression** model for classification.

## Prerequisites

- **Python 3.x**: Ensure that Python 3.x is installed.
- Required Libraries:
  ```bash
  pip install pandas matplotlib seaborn scikit-learn
