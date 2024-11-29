from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializer import PredictionInputSerializer
import joblib
import os

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/sports_odds_model.pkl')

# Load the model
model = joblib.load(MODEL_PATH)

class PredictionAPIView(APIView):
    """
    API to predict odds for a sports match.

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
        # Validate the input data
        serializer = PredictionInputSerializer(data=request.data)
        if serializer.is_valid():
            # Extract the validated data
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
            # Make the prediction
            prediction = model.predict(input_data)[0]
            return Response({"prediction": prediction}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
