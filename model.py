# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer  # For handling missing values (optional)
import matplotlib.pyplot as plt


# %%
if __name__ == "__main__":
    data_path = r"C:\Users\Ashwin\Downloads\Machine Learning\prjct 1\student-mat.csv"
data = pd.read_csv(data_path, sep=";")

# %%
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


# %%
# Check for missing values (optional)
if data.isnull().sum().any():
    imputer = SimpleImputer(strategy="mean")  # Replace missing values with mean (adjust strategy as needed)
    data = pd.DataFrame(imputer.fit_transform(data))

# %%
 # Define features and target variable
features = ["G1", "G2", "studytime", "failures", "absences"]
target = "G3"
X = data[features]  # Select features as a DataFrame
y = data[target]  # Select target variable as a Series
features_to_normalize = ["failures", "G1", "G2", "absences"]

# %%

# Assuming your dataset is already loaded as 'data'
features = ["G1", "G2", "studytime", "failures", "absences"]
X = data[features]

# 1. Show first few rows
print("Sample data:")
print(X.head(), "\n")

# 2. Basic statistics
print("Descriptive statistics:")
print(X.describe(), "\n")

# 3. Detailed value ranges and info
for feature in features:
    print(f"Feature: {feature}")
    print(f" - Type: {X[feature].dtype}")
    print(f" - Unique values: {X[feature].nunique()}")
    print(f" - Min: {X[feature].min()}")
    print(f" - Max: {X[feature].max()}")
    print(f" - Mean: {X[feature].mean():.2f}")
    print(f" - Median: {X[feature].median()}")
    print(f" - Std Dev: {X[feature].std():.2f}")
    print("-" * 40)


# %%
 # Convert to NumPy arrays if necessary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
def normalize_features(X_train, X_test, y_train, y_test, features_to_normalize):
    # Normalize selected features
    scaler = MinMaxScaler()
    X_train[features_to_normalize] = scaler.fit_transform(X_train[features_to_normalize])
    X_test[features_to_normalize] = scaler.transform(X_test[features_to_normalize])
    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = normalize_features(X_train, X_test, y_train, y_test, features_to_normalize)



# %%
# Initialize models with regularization
linear_model = LinearRegression()
ridge_model = Ridge(alpha=0.3)  # L2 regularization
lasso_model = Lasso(alpha=0.2)  # L1 regularization

# Group models and their names for easy iteration
models = [linear_model, ridge_model, lasso_model]
model_names = ["Linear Regression", "Ridge Regression", "Lasso Regression"]


# %%
for model, name in zip(models, model_names):
        model.fit(X_train, y_train)
        r2_score_test = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        print(f"\n{name}:")
        print(f"R-squared score (testing set): {r2_score_test:.3f}")
        print("Cross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())

# %%
# Prediction and visualization (optional)
predictions = model.predict(X_test)
plt.scatter(predictions, y_test)  # Replace 'studytime' with the desired feature

        # Add labels and title
plt.xlabel('Predicted Grade')
plt.ylabel('Final Grade (G3)')
plt.title(f'Final Grade vs. Predicted Grade ({name})')

        # Add a trendline (optional)
plt.plot(predictions, predictions, color='red')

# %%
plt.show()

# %%
    print("\n🎓 Predict your final grade (G3) using the Linear Regression model")
    try:
        # Collect only the features used for training
        G1 = int(input("Enter G1 (first period grade, 0–20): "))
        G2 = int(input("Enter G2 (second period grade, 0–20): "))
        studytime = int(input("Enter weekly study time (1–4): "))
        failures = int(input("Enter number of past class failures (0–3, else 4): "))
        absences = int(input("Enter number of absences (0–93): "))

        # Constraint checks
        assert 0 <= G1 <= 20
        assert 0 <= G2 <= 20
        assert 1 <= studytime <= 4
        failures = failures if failures < 4 else 4
        assert 0 <= absences <= 93

        # Prepare DataFrame for prediction
        input_data = pd.DataFrame([{
            "G1": G1,
            "G2": G2,
            "studytime": studytime,
            "failures": failures,
            "absences": absences
        }])

        # Normalize
        features_to_normalize = ["failures", "G1", "G2", "absences"]
        input_data[features_to_normalize] = scaler.transform(input_data[features_to_normalize])

        # Predict using linear regression model
        predicted_grade = linear_model.predict(input_data)[0]
        print(f"\n🎯 Predicted Final Grade (G3): {predicted_grade:.2f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


# %%
import joblib
joblib.dump(linear_model, 'grade_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save your MinMaxScaler too



