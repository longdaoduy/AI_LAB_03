import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Function to evaluate regression model performance
def evaluate_regression_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"--- {model_name} Performance ---")
    print(f"MAE:  {mae:,.2f} EUR")
    print(f"MSE:  {mse:,.2f} EUR^2")
    print(f"RMSE: {rmse:,.2f} EUR")
    print(f"R-squared: {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": f"{mae:,.2f}",
        "MSE": f"{mse:,.2f}",
        "RMSE": f"{rmse:,.2f}",
        "R2 Score": f"{r2:.4f}"
    }

# Load dataset
print("============ LOADING DATA ============")
dataset = "paris_housing_prices_dataset.csv"
df = pd.read_csv(dataset)

print(f"- Data Source: {dataset}")
print(f"- Total samples: {len(df)}")
print(f"- Total features: {len(df.columns) - 1}")
print(f"- Target variable: Price_EUR")
# Preprocess data
print("\n============ PREPROCESSING DATA ============")
df_processed = df.drop(columns=["Property_ID"])
print("- Dropped column: 'Property_ID' (No predictive value)")

# Encode categorical features
label_endcoders = {}
categorical_cols = ['Property_Type', 'Condition']

print("- Encoding categorical columns to numbers:")
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_endcoders[col] = le
    classes_mapped = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  + {col}: {classes_mapped}")
# Split data into features (X) and target (y)
X = df_processed.drop(columns=["Price_EUR"])
y = df_processed["Price_EUR"]

# Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n- Splitting data into Train and Test sets (80/20):")
print(f"  + Training set: {X_train.shape[0]} samples")
print(f"  + Testing set:  {X_test.shape[0]} samples")

# Train baseline decision tree model
print("\n============ TRAINING BASELINE MODEL ============")
baseline_tree = DecisionTreeRegressor(random_state=42)
baseline_tree.fit(X_train, y_train)

#Compute baseline model performance
results = []
res_baseline = evaluate_regression_model(baseline_tree, X_test, y_test, "Baseline Tree")
results.append(res_baseline)






# Display results
print("\n============ PERFORMANCE COMPARISON TABLE ============")
df_results = pd.DataFrame(results)
df_results.set_index('Model', inplace=True)
print(df_results)









