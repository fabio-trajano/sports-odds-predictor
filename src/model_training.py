from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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
