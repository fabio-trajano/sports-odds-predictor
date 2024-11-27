from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_model(features, target):
    """
    Train a Random Forest model to predict home win odds.

    Args:
        features (pd.DataFrame): Processed features.
        target (pd.Series): Target values (odds_home_win).

    Returns:
        model: Trained Random Forest model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained. Mean Absolute Error on test set: {mae:.2f}")

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
