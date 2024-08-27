
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # For handling missing values (optional)
import matplotlib.pyplot as plt


def load_and_preprocess_data(data_path):
  """
  Loads data from CSV, handles missing values (optional), and prepares features and target.

  Args:
      data_path (str): Path to the CSV file containing student performance data.

  Returns:
      tuple: A tuple containing the following elements:
          - X_train: Training features as a NumPy array.
          - X_test: Testing features as a NumPy array.
          - y_train: Training target variable as a NumPy array.
          - y_test: Testing target variable as a NumPy array.
  """

  data = pd.read_csv(data_path, sep=";")

  # Check for missing values (optional)
  if data.isnull().sum().any():
    imputer = SimpleImputer(strategy="mean")  # Replace missing values with mean (adjust strategy as needed)
    data = pd.DataFrame(imputer.fit_transform(data))

  # Define features and target variable
  features = ["G1", "G2", "studytime", "failures", "absences"]
  target = "G3"

  X = data[features]  # Select features as a DataFrame
  y = data[target]  # Select target variable as a Series

  # Convert to NumPy arrays if necessary
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

  return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  data_path = "student-mat.csv"  # Replace with actual path to your data

  X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

  linear = LinearRegression()
  linear.fit(X_train, y_train)
  r2_score = linear.score(X_test, y_test)

  bias = linear.intercept_  # Single value representing the effective bias

  print(f"R-squared score (coefficient of determination): {r2_score:.3f}")
  print("co:  \n", linear.coef_)
  print("intercept:  \n", linear.intercept_)

  '''predictions = linear.predict(X_test)

  for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])''' 

predictions = linear.predict(X_test)

# Option 1: Use .iloc for row selection (access by position)
for prediction, row, target_value in zip(predictions, X_test.values, y_test.values):
    print(prediction, list(row), target_value)


plt.scatter(predictions, y_test)  # Replace 'studytime' with the desired feature

  # Add labels and title
plt.xlabel('predicted grade')
plt.ylabel('Final Grade (G3)')
plt.title('Final Grade vs. Study Time (Test Set)')

  # Add a trendline (optional)
plt.plot(predictions, predictions, color='red')  # Replace 'studytime' with the desired feature

plt.show()