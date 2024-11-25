import pandas as pd
from src.model_training import train_model
import joblib

# File paths
PROCESSED_FEATURES_PATH = "data/processed/features.csv"
PROCESSED_TARGET_PATH = "data/processed/target.csv"
MODEL_OUTPUT_PATH = "models/sports_odds_model.pkl"

def main():
    """
    Train the machine learning model and save it.
    """
    # Load processed features and target
    features = pd.read_csv(PROCESSED_FEATURES_PATH)
    target = pd.read_csv(PROCESSED_TARGET_PATH).squeeze()  # Convert to Series

    # Train the model
    model = train_model(features, target)

    # Save the trained model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
