from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from .serializer import PredictionInputSerializer

# Paths to models
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/random_forest_model.pkl")
GB_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/gradient_boosting_model.pkl")

# Load models if available
random_forest_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
gradient_boosting_model = joblib.load(GB_MODEL_PATH) if os.path.exists(GB_MODEL_PATH) else None

# Load test data for metrics
TEST_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../../data/processed/features.csv")
TEST_TARGET_PATH = os.path.join(os.path.dirname(__file__), "../../data/processed/target.csv")
test_features = pd.read_csv(TEST_FEATURES_PATH) if os.path.exists(TEST_FEATURES_PATH) else None
test_target = pd.read_csv(TEST_TARGET_PATH).squeeze() if os.path.exists(TEST_TARGET_PATH) else None


@api_view(["GET"])
def api_home(request):
    """
    Home page for the Sports Odds Predictor API.

    Returns a list of available endpoints and their descriptions.
    """
    data = {
        "message": "Welcome to the Sports Odds Predictor API!",
        "endpoints": {
            "/api/predict/random_forest/": "Predict odds using the Random Forest model (POST).",
            "/api/predict/gradient_boosting/": "Predict odds using the Gradient Boosting model (POST).",
            "/api/metrics/": "Get performance metrics for both models (GET)."
        },
        "example_input": {
            "home_team_rank": 2,
            "away_team_rank": 3,
            "home_recent_wins": 5,
            "away_recent_wins": 4,
            "home_goals_scored_avg": 2.7,
            "away_goals_scored_avg": 2.5
        }
    }
    return Response(data)


class RandomForestPredictionAPIView(APIView):
    """
    API to predict odds using Random Forest model.
    """
    def post(self, request):
        if random_forest_model is None:
            return Response(
                {"error": "Random Forest model is not available. Please train the model or upload it."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        serializer = PredictionInputSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            input_data = [[
                data["home_team_rank"],
                data["away_team_rank"],
                data["home_recent_wins"],
                data["away_recent_wins"],
                data["home_goals_scored_avg"],
                data["away_goals_scored_avg"],
                data["home_team_rank"] - data["away_team_rank"],  # rank_difference
                (data["home_goals_scored_avg"] + data["away_goals_scored_avg"]) / 2  # combined_goals_avg
            ]]
            prediction = random_forest_model.predict(input_data)[0]
            return Response({"prediction": prediction}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GradientBoostingPredictionAPIView(APIView):
    """
    API to predict odds using Gradient Boosting model.
    """
    def post(self, request):
        if gradient_boosting_model is None:
            return Response(
                {"error": "Gradient Boosting model is not available. Please train the model or upload it."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        serializer = PredictionInputSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            input_data = [[
                data["home_team_rank"],
                data["away_team_rank"],
                data["home_recent_wins"],
                data["away_recent_wins"],
                data["home_goals_scored_avg"],
                data["away_goals_scored_avg"],
                data["home_team_rank"] - data["away_team_rank"],  # rank_difference
                (data["home_goals_scored_avg"] + data["away_goals_scored_avg"]) / 2  # combined_goals_avg
            ]]
            prediction = gradient_boosting_model.predict(input_data)[0]
            return Response({"prediction": prediction}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MetricsAPIView(APIView):
    """
    API endpoint to return performance metrics for Random Forest and Gradient Boosting models.
    """
    def get(self, request):
        if test_features is None or test_target is None:
            return Response(
                {"error": "Test data is not available. Please ensure test features and targets exist."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        metrics = {}

        if random_forest_model:
            rf_predictions = random_forest_model.predict(test_features)
            rf_mae = mean_absolute_error(test_target, rf_predictions)
            rf_mse = mean_squared_error(test_target, rf_predictions)
            rf_rmse = np.sqrt(rf_mse)
            rf_r2 = r2_score(test_target, rf_predictions)

            metrics["Random Forest"] = {
                "Mean Absolute Error (MAE)": round(rf_mae, 2),
                "Mean Squared Error (MSE)": round(rf_mse, 2),
                "Root Mean Squared Error (RMSE)": round(rf_rmse, 2),
                "R² Score": round(rf_r2, 2),
            }

        if gradient_boosting_model:
            gb_predictions = gradient_boosting_model.predict(test_features)
            gb_mae = mean_absolute_error(test_target, gb_predictions)
            gb_mse = mean_squared_error(test_target, gb_predictions)
            gb_rmse = np.sqrt(gb_mse)
            gb_r2 = r2_score(test_target, gb_predictions)

            metrics["Gradient Boosting"] = {
                "Mean Absolute Error (MAE)": round(gb_mae, 2),
                "Mean Squared Error (MSE)": round(gb_mse, 2),
                "Root Mean Squared Error (RMSE)": round(gb_rmse, 2),
                "R² Score": round(gb_r2, 2),
            }

        if not metrics:
            return Response(
                {"error": "No models available to compute metrics. Please train the models."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(metrics, status=status.HTTP_200_OK)
