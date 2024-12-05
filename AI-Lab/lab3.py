import pandas as pd  # Importing pandas for data manipulation
import penguins

# Prompt for the author's name
author_name = input("Enter your name:")  # Takes input from the user for the author's name

# Load the project dataset
data = pd.read_csv('penguins.csv')  # Reads the CSV file containing the penguin dataset into a DataFrame
print(f"\nAuthor:{author_name}")  # Prints the author’s name
print("Initial Data:")  # Prints a label to indicate the start of the dataset preview
print(data.head())  # Displays the first five rows of the dataset

# Step 2.1: Inspect for missing values
missing_value = penguins.isnull().sum()  # Checks for missing values and counts them per column

# Display the count of missing values for each column
mising_values[missing_values > 0]  # Displays columns with missing values (if any)

# Step 2.2: Handle missing values by dropping rows
penguins_cleaned_dropped = penguins.dropna()  # Drops rows with any missing values

# Display the shape of the original and cleaned dataset
print(f"Original dataset shape: {penguins.shape}")  # Prints the shape of the original dataset
print(
    f"Cleaned dataset shape (dropped): {penguins_cleaned_dropped.shape}")  # Prints the shape of the cleaned dataset after dropping rows

# Step 2.2: Handle missing values by imputing with the mean
penguins_imputed = penguins.fillna(
    penguins.mean(numeric_only=True))  # Fills missing values with the mean of the column for numeric columns

# Display the shape of the original and cleaned dataset
print(f"Original dataset shape:{penguins.shape}")  # Prints the shape of the original dataset
print(f"Cleaned dataset shape (imputed): {penguins_imputed.shape}")  # Prints the shape of the dataset after imputation

# With the mode
penguins_imputed_mode = penguins.fillna(
    penguins.mode().iloc[0])  # Imputes missing values with the mode (most frequent value)

# Display the shape of the original and cleaned dataset
print(f"Original dataset shape:{penguins.shape}")  # Prints the shape of the original dataset
print(
    f"Cleaned dataset shape (imputed with mode): {penguins_imputed_mode.shape}")  # Prints the shape after imputation with the mode

import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
import seaborn as sns  # Importing Seaborn for statistical plotting

# Step 3.1: Create box plots to identify outliers
plt.figure(figsize=(15, 5))  # Sets the figure size for the box plots

# Box plot for body mass
plt.subplot(1, 3, 1)  # Creates the first subplot for body mass
sns.boxplot(y=penguins['body_mass_g'])  # Creates a box plot for the 'body_mass_g' column
plt.title('Box Plot of Body Mass (g)')  # Title for the body mass plot

# Box plot for culmen Length
plt.subplot(1, 3, 2)  # Creates the second subplot for culmen length
sns.boxplot(y=penguins['culmen_length_mm'])  # Creates a box plot for 'culmen_length_mm' column
plt.title('Box Plot of Culmen Length (mm)')  # Title for the culmen length plot

# Box plot for culmen depth
plt.subplot(1, 3, 3)  # Creates the third subplot for culmen depth
sns.boxplot(y=penguins['culmen_depth_mm'])  # Creates a box plot for 'culmen_depth_mm' column
plt.title('Box Plot of Culmen Depth (mm)')  # Title for the culmen depth plot

plt.tight_layout()  # Adjusts the layout to avoid overlap of plots
plt.show()  # Displays the box plots


# Step 3.2: Remove outliers based on IQR for body_mass_g, culmen_length_mm, and culmen_depth_mm
def remove_outliers(df, column):  # Defines a function to remove outliers using the IQR method
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outlier detection
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outlier detection
    return df[
        (df[column] >= lower_bound) & (df[column] <= upper_bound)]  # Returns the filtered DataFrame without outliers


# Start with the original dataset
penguins_no_outliers = penguins.copy()  # Creates a copy of the original dataset

# Remove outliers for all three columns
for col in ['body_mass_g', 'culmen_length_mm', 'culmen_depth_mm']:  # Iterates over columns to remove outliers
    penguins_no_outliers = remove_outliers(penguins_no_outliers, col)  # Applies the remove_outliers function

# Display the shape of the original and cleaned dataset
print(f"Original dataset shape:{penguins.shape}")  # Prints the shape of the original dataset
print(
    f"Cleaned dataset shape (no outliers): {penguins_no_outliers.shape}")  # Prints the shape of the dataset after removing outliers

from sklearn.preprocessing import MinMaxScaler  # Imports MinMaxScaler from sklearn to normalize data

# Step 4.1: Normalize the numerical feature using min-max scaling
scaler = MinMaxScaler()  # Creates a MinMaxScaler object to scale features

# Apply MinMax scaling to selected columns
penguins_normalized = penguins_no_outliers.copy()  # Creates a copy of the DataFrame without outliers
penguins_normalized[['body_mass_g', 'culmen_length_mm', 'culmen_depth_mm']] = scaler.fit_transform(
    penguins_normalized[['body_mass_g', 'culmen_length_mm', 'culmen_depth_mm']]
    # Applies scaling to the selected columns
)

# Display the first few rows of the normalized dataset
print(penguins_normalized[['body_mass_g', 'culmen_length_mm',
                           'culmen_depth_mm']].head())  # Displays the first few rows of the normalized dataset

# Step 5.1: Create a new feature based on body mass
penguins_df['average_body_mass'] = penguins_df.groupby('species')['body_mass_g'].transform(
    'mean')  # Adds a new column for the average body mass per species

# Display the first few rows to verify the new column
print(penguins_df[['species', 'body_mass_g', 'average_body_mass']].head())  # Prints the updated DataFrame


# Step 5.2: Create a size category based on body mass
def categorize_size(mass):  # Defines a function to categorize size based on body mass
    if pd.isna(mass):  # Check for NaN values
        return 'Unknown'  # Assign a category for NaN values
    elif mass < 3500:  # Categorize based on mass ranges
        return 'Small'
    elif 3500 <= mass < 5000:
        return 'Medium'
    else:
        return 'Large'


penguins_df['size_category'] = penguins_df['body_mass_g'].apply(categorize_size)  # Applies the categorize_size function

# Display first row to verify the new column
print(penguins_df[
          ['body_mass_g', 'size_category']].head())  # Prints the updated DataFrame with the new 'size_category' column

# Step 6.1: Correlation analysis
numeric_columns = penguins_df.select_dtypes(
    include=['float64', 'int64'])  # Selects only numeric columns from the DataFrame
correlation_matrix = numeric_columns.corr()  # Computes the correlation matrix

# Display the correlation matrix
plt.figure(figsize=(8, 5))  # Sets the figure size for the heatmap
plt.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            fmt='.2f')  # Creates a heatmap to visualize the correlation matrix
plt.title('Correlation Matrix')  # Title for the heatmap
plt.show()  # Displays the heatmap

from sklearn.ensemble import RandomForestClassifier  # Imports RandomForestClassifier for classification
from sklearn.model_selection import \
    train_test_split  # Imports train_test_split to split data into training and testing sets

# Prepare the data for modeling (drop non-numeric and target columns)
features = penguins_df[['body_mass_g', 'culmen_length_mm', 'culmen_depth_mm']]  # Defines the feature columns
target = penguins_df['species']  # Defines the target column

# Train random forest model to assess feature importance
model = RandomForestClassifier(random_state=42)  # Creates a RandomForestClassifier model
model.fit(features, target)  # Fits the model to the data

# Get feature importance
importance = model.feature_importances_  # Gets the importance of each feature

# Create a DataFrame for visualization
importance_df = pd.DataFrame(
    {'Feature': features.columns, 'Importance': importance})  # Creates a DataFrame to store feature importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)  # Sorts the features by importance

# Display feature importance
print(importance_df)  # Prints the sorted feature importance

# Step 7.1: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,
                                                    random_state=42)  # Splits the data into 80% training and 20% test sets

# Display the shapes of the training and testing sets
print(f"Training feature set shape: {X_train.shape}")  # Prints the shape of the training feature set
print(f"Testing feature set shape {X_test.shape}")  # Prints the shape of the testing feature set

import numpy as np  # Imports numpy for handling arrays
from sklearn.linear_model import LogisticRegression  # Imports LogisticRegression for classification
from sklearn.model_selection import train_test_split  # Imports train_test_split
from sklearn.impute import SimpleImputer  # Imports SimpleImputer for handling missing values

# Example dataset creation (replace this with your actual dataset)
x = np.array([[1, 2], [np.nan, 3], [7, 6], [np.nan, np.nan], [4, 5]])  # Example dataset with missing values
y = np.array([0, 1, 0, 1, 0])  # Corresponding labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)  # Splits the data

# Create an imputer to fill NaN values with the mean
imputer = SimpleImputer(strategy='mean')  # Initializes the SimpleImputer to replace missing values with the mean

# Fit the imputer on the training data and transform both training and test data
X_train_imputed = imputer.fit_transform(X_train)  # Fits and transforms the training data
X_test_imputed = imputer.transform(X_test)  # Transforms the test data

# Fit the logistic regression model
logistic_model = LogisticRegression(max_iter=200)  # Initializes the LogisticRegression model
logistic_model.fit(X_train_imputed, y_train)  # Fits the logistic model to the imputed training data

# Make predictions on the test set
y_pred = logistic_model.predict(X_test_imputed)  # Predicts the test set labels

# Print predictions
print("Predictions:", y_pred)  # Prints the model's predictions
