import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
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

    features_to_normalize = ["failures", "G1", "G2", "absences"]

    # Check for missing values (optional)
    if data.isnull().sum().any():
        imputer = SimpleImputer(strategy="mean")  # Replace missing values with mean (adjust strategy as needed)
        data = pd.DataFrame(imputer.fit_transform(data))

    # Define features and target variable
    features = ["G1", "G2", "studytime", "failures", "absences"]
    target = "G3"

    X = data[features]  # Select features as a DataFrame
    y = data[target]  # Select target variable as a Series
    features_to_normalize = ["failures", "G1", "G2", "absences"]

    # Convert to NumPy arrays if necessary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize selected features
    scaler = MinMaxScaler()
    X_train[features_to_normalize] = scaler.fit_transform(X_train[features_to_normalize])
    X_test[features_to_normalize] = scaler.transform(X_test[features_to_normalize])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_path = "prjct 1/student-mat.csv"  # Replace with actual path to your data"

    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # Train models with different regularization techniques
    linear_model = LinearRegression()
    ridge_model = Ridge(alpha=0.3)  # Adjust alpha as needed
    lasso_model = Lasso(alpha=0.2)  # Adjust alpha as needed

    models = [linear_model, ridge_model, lasso_model]
    model_names = ["Linear Regression", "Ridge Regression", "Lasso Regression"]

    for model, name in zip(models, model_names):
        model.fit(X_train, y_train)
        r2_score_test = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        print(f"\n{name}:")
        print(f"R-squared score (testing set): {r2_score_test:.3f}")
        print("Cross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())

        # Prediction and visualization (optional)
        predictions = model.predict(X_test)
        plt.scatter(predictions, y_test)  # Replace 'studytime' with the desired feature

        # Add labels and title
        plt.xlabel('Predicted Grade')
        plt.ylabel('Final Grade (G3)')
        plt.title(f'Final Grade vs. Predicted Grade ({name})')

        # Add a trendline (optional)
        plt.plot(predictions, predictions, color='red')  # Replace 'studytime' with the desired feature

        plt.show()
