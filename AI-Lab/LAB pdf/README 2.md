**Customer Churn Prediction with Drift Detection**
Overview
This project implements a customer churn prediction model using logistic regression and an ensemble method (Bagging) with drift detection capabilities. The model is designed to adapt to changes in data distribution (concept drift) using the ADWIN algorithm.

**Language**
Python
Libraries Used
pandas: For data manipulation and analysis.
numpy: For numerical operations.
matplotlib: For creating static visualizations.
seaborn: For enhanced visualizations.
scikit-learn: For traditional machine learning metrics and models.
imblearn: For handling class imbalance through SMOTE and undersampling.
river: For online machine learning and drift detection.
Prerequisites
To run this code, you need to have the following installed:

*Python 3.x*
Jupyter Notebook (optional, but recommended for interactive development)
Required libraries can be installed using pip:
bash

CMD terminal
**pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn river**

*Dataset*

The dataset used in this project is assumed to be named customer_dataset.csv. It should contain at least the following columns:
churn: The target variable indicating whether a customer has churned (1 for churned, 0 otherwise).
Additional feature columns relevant to the prediction task (e.g., age, income, monthly_bill, etc.).

*How to Run*

Ensure that you have your dataset (customer_dataset.csv) in the same directory as your script or Jupyter Notebook.
Install the required libraries using the command mentioned above.
Open your Python environment or Jupyter Notebook.
Copy and paste the provided Python code into your script or notebook cell.
Run the code block to execute the model training and evaluation.

*Functionality*

The code loads the dataset, processes it, and trains a logistic regression model using a bagging approach.
It handles class imbalance using SMOTE for oversampling and random undersampling.
The model includes a drift detection mechanism using the ADWIN algorithm to monitor changes in data distribution.
When drift is detected, the model is retrained to adapt to new data distributions.
Finally, the model's performance is evaluated using classification metrics and AUC-ROC score.