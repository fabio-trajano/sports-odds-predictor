import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# File paths
RF_MODEL_OUTPUT_PATH = "models/random_forest_model.pkl"
GB_MODEL_OUTPUT_PATH = "models/gradient_boosting_model.pkl"
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

def plot_feature_importance(model_path, model_name):
    """
    Plot feature importance from the trained model.

    Args:
        model_path (str): Path to the trained model file.
        model_name (str): Name of the model (e.g., Random Forest, Gradient Boosting).
    """
    # Load the trained model
    model = joblib.load(model_path)

    # Extract feature importances
    importances = model.feature_importances_
    features = pd.read_csv(PROCESSED_FEATURES_PATH).columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"{model_name} - Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_residuals(model_path, model_name, features, target):
    """
    Plot residuals (differences between actual and predicted values).

    Args:
        model_path (str): Path to the trained model file.
        model_name (str): Name of the model (e.g., Random Forest, Gradient Boosting).
        features (pd.DataFrame): Processed features.
        target (pd.Series): Target variable.
    """
    # Load the trained model
    model = joblib.load(model_path)

    # Generate predictions
    predictions = model.predict(features)

    # Calculate residuals
    residuals = target - predictions

    # Plot residuals histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="blue", bins=30)
    plt.title(f"{model_name} - Residuals (Target - Predictions)", fontsize=16)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(model_path, model_name, features, target):
    """
    Plot predicted vs actual values to evaluate model performance.

    Args:
        model_path (str): Path to the trained model file.
        model_name (str): Name of the model (e.g., Random Forest, Gradient Boosting).
        features (pd.DataFrame): Processed features.
        target (pd.Series): Target variable.
    """
    # Load the trained model
    model = joblib.load(model_path)

    # Generate predictions
    predictions = model.predict(features)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(target, predictions, alpha=0.5, label="Predicted vs Actual")
    plt.plot([target.min(), target.max()], [target.min(), target.max()], "r--", lw=2, label="Ideal: Predicted = Actual")
    plt.title(f"{model_name} - Predicted vs Actual", fontsize=16)
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Visualize model-related metrics and insights for both models.
    """
    # Load processed data
    features = pd.read_csv(PROCESSED_FEATURES_PATH)
    target = pd.read_csv(PROCESSED_TARGET_PATH).squeeze()

    # Plot feature-target correlation
    plot_feature_target_correlation(features, target)

    # Visualizations for Random Forest
    print("Visualizing Random Forest Model...")
    plot_feature_importance(RF_MODEL_OUTPUT_PATH, "Random Forest")
    plot_residuals(RF_MODEL_OUTPUT_PATH, "Random Forest", features, target)
    plot_predictions_vs_actual(RF_MODEL_OUTPUT_PATH, "Random Forest", features, target)

    # Visualizations for Gradient Boosting
    print("\nVisualizing Gradient Boosting Model...")
    plot_feature_importance(GB_MODEL_OUTPUT_PATH, "Gradient Boosting")
    plot_residuals(GB_MODEL_OUTPUT_PATH, "Gradient Boosting", features, target)
    plot_predictions_vs_actual(GB_MODEL_OUTPUT_PATH, "Gradient Boosting", features, target)

if __name__ == "__main__":
    main()
