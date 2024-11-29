import pandas as pd
from src.model_training import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import joblib

# File paths
PROCESSED_FEATURES_PATH = "data/processed/features.csv"
PROCESSED_TARGET_PATH = "data/processed/target.csv"
MODEL_OUTPUT_PATH = "models/sports_odds_model.pkl"

def main():
    """
    Train the model, evaluate it, and save the trained model.
    """
    # Load processed data
    features = pd.read_csv(PROCESSED_FEATURES_PATH)
    target = pd.read_csv(PROCESSED_TARGET_PATH).squeeze()

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Save the trained model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
