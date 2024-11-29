from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializer import PredictionInputSerializer
import joblib
import os

from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view

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
        },
        "example_input": {
            "home_team_rank": 2,
            "away_team_rank": 3,
            "home_recent_wins": 5,
            "away_recent_wins": 4,
            "home_goals_scored_avg": 2.7,
            "away_goals_scored_avg": 2.5
        },
    }
    return Response(data)


# Paths to models
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/random_forest_model.pkl")
GB_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/gradient_boosting_model.pkl")

# Load models
random_forest_model = joblib.load(RF_MODEL_PATH)
gradient_boosting_model = joblib.load(GB_MODEL_PATH)

class RandomForestPredictionAPIView(APIView):
    """
    API to predict odds using Random Forest model.

    Example Input (JSON):

    {
        "home_team_rank": 2,
        "away_team_rank": 3,
        "home_recent_wins": 5,
        "away_recent_wins": 4,
        "home_goals_scored_avg": 2.7,
        "away_goals_scored_avg": 2.5
    }

    Example Output (JSON):

    {
        "prediction": 0.75
    }
    """
    def post(self, request):
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

    Example Input (JSON):

    {
        "home_team_rank": 2,
        "away_team_rank": 3,
        "home_recent_wins": 5,
        "away_recent_wins": 4,
        "home_goals_scored_avg": 2.7,
        "away_goals_scored_avg": 2.5
    }

    Example Output (JSON):

    {
        "prediction": 0.75
    }
    """
    def post(self, request):
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
