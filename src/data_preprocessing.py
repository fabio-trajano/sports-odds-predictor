import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load raw data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)

def clean_data(data):
    """
    Perform basic data cleaning like removing nulls and correcting types.

    Args:
        data (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Drop rows with missing values
    data = data.dropna()

    # Ensure correct data types
    data["match_date"] = pd.to_datetime(data["match_date"])

    return data

def engineer_features(data):
    """
    Create new features from existing ones to improve the model's predictive power.

    Args:
        data (pd.DataFrame): Cleaned data.

    Returns:
        pd.DataFrame: Data with new features.
    """
    # Example feature: Rank difference
    data["rank_difference"] = data["home_team_rank"] - data["away_team_rank"]

    # Example feature: Combined goals average
    data["combined_goals_avg"] = (data["home_goals_scored_avg"] + data["away_goals_scored_avg"]) / 2

    return data

def scale_features(features):
    """
    Scale features using StandardScaler for normalization.

    Args:
        features (pd.DataFrame): Features to be scaled.

    Returns:
        pd.DataFrame: Scaled features.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features, columns=features.columns)

def preprocess_data(filepath):
    """
    Complete preprocessing pipeline for raw data.

    Args:
        filepath (str): Path to the raw data file.

    Returns:
        dict: Dictionary containing 'features' (X) and 'target' (y).
    """
    # Load data
    data = load_data(filepath)

    # Clean data
    data = clean_data(data)

    # Engineer features
    data = engineer_features(data)

    # Select features and target
    features = data[
        [
            "home_team_rank",
            "away_team_rank",
            "home_recent_wins",
            "away_recent_wins",
            "home_goals_scored_avg",
            "away_goals_scored_avg",
            "rank_difference",
            "combined_goals_avg",
        ]
    ]
    target = data["odds_home_win"]

    # Scale features
    features = scale_features(features)

    return {"features": features, "target": target}
