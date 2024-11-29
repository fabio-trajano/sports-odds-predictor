from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import pandas as pd

# File paths
PROCESSED_FEATURES_PATH = "data/processed/features.csv"
PROCESSED_TARGET_PATH = "data/processed/target.csv"
RF_MODEL_OUTPUT_PATH = "models/random_forest_model.pkl"
GB_MODEL_OUTPUT_PATH = "models/gradient_boosting_model.pkl"

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        model: Trained Random Forest model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        model: Trained Gradient Boosting model.
    """
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print performance metrics.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Print results
    print("Model Performance on Test Set:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def main():
    """
    Train and evaluate multiple models, then save them.
    """
    # Load processed data
    features = pd.read_csv(PROCESSED_FEATURES_PATH)
    target = pd.read_csv(PROCESSED_TARGET_PATH).squeeze()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train and evaluate Random Forest model
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    joblib.dump(rf_model, RF_MODEL_OUTPUT_PATH)
    print(f"Random Forest model saved to {RF_MODEL_OUTPUT_PATH}")

    # Train and evaluate Gradient Boosting model
    print("\nTraining Gradient Boosting model...")
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_test, y_test)
    joblib.dump(gb_model, GB_MODEL_OUTPUT_PATH)
    print(f"Gradient Boosting model saved to {GB_MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
