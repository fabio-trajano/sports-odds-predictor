import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# File paths
MODEL_OUTPUT_PATH = "models/sports_odds_model.pkl"
PROCESSED_FEATURES_PATH = "data/processed/features.csv"
PROCESSED_TARGET_PATH = "data/processed/target.csv"

def plot_feature_target_correlation(features, target):
    """
    Plot correlation between features and the target.

    Args:
        features (pd.DataFrame): Processed features.
        target (pd.Series): Target variable.
    """
    data = pd.concat([features, target.rename("target")], axis=1)
    correlation = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature-Target Correlation", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_importance():
    """
    Plot feature importance from the trained model.
    """
    # Load the trained model
    model = joblib.load(MODEL_OUTPUT_PATH)

    # Get feature importances
    importances = model.feature_importances_
    features = pd.read_csv(PROCESSED_FEATURES_PATH).columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_residuals(features, target):
    """
    Plot residuals to evaluate model performance.
    """
    # Load the trained model
    model = joblib.load(MODEL_OUTPUT_PATH)

    # Make predictions
    predictions = model.predict(features)

    # Calculate residuals
    residuals = target - predictions

    # Plot residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="blue", bins=30)
    plt.title("Residuals (Target - Predictions)", fontsize=16)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(features, target):
    """
    Plot predicted vs actual values.
    """
    # Load the trained model
    model = joblib.load(MODEL_OUTPUT_PATH)

    # Make predictions
    predictions = model.predict(features)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(target, predictions, alpha=0.5)
    plt.plot([target.min(), target.max()], [target.min(), target.max()], "r--", lw=2)
    plt.title("Predicted vs Actual", fontsize=16)
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.tight_layout()
    plt.show()

def main():
    """
    Visualize model-related metrics and insights.
    """
    # Load processed data
    features = pd.read_csv(PROCESSED_FEATURES_PATH)
    target = pd.read_csv(PROCESSED_TARGET_PATH).squeeze()

    # Visualizations
    plot_feature_target_correlation(features, target)
    plot_feature_importance()
    plot_residuals(features, target)
    plot_predictions_vs_actual(features, target)

if __name__ == "__main__":
    main()
