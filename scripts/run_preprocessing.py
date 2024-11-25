import pandas as pd
from src.data_preprocessing import preprocess_data

# File paths
RAW_DATA_PATH = "data/raw/match_odds_data.csv"
PROCESSED_FEATURES_PATH = "data/processed/features.csv"
PROCESSED_TARGET_PATH = "data/processed/target.csv"

def main():
    """
    Execute the preprocessing pipeline and save processed data.
    """
    # Preprocess the data
    processed_data = preprocess_data(RAW_DATA_PATH)

    # Save processed features and target
    processed_data["features"].to_csv(PROCESSED_FEATURES_PATH, index=False)
    processed_data["target"].to_csv(PROCESSED_TARGET_PATH, index=False)

    print(f"Processed features saved to {PROCESSED_FEATURES_PATH}")
    print(f"Processed target saved to {PROCESSED_TARGET_PATH}")

if __name__ == "__main__":
    main()
