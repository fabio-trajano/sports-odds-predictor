import pandas as pd
import joblib

class Predictor:
    def __init__(self, model_path):
        """
        Initialize the Predictor class with the path to the trained model.
        Args:
            model_path (str): Path to the trained model file.
        """
        self.model = joblib.load(model_path)
        print("Model loaded successfully.")

    def get_user_input(self):
        """
        Prompt the user to input the necessary data for prediction.
        Returns:
            pd.DataFrame: DataFrame containing the input data.
        """
        print("Please provide the following match details:")

        # Collect inputs interactively
        home_team_rank = float(input("Home team rank: "))
        away_team_rank = float(input("Away team rank: "))
        home_recent_wins = float(input("Home recent wins: "))
        away_recent_wins = float(input("Away recent wins: "))
        home_goals_scored_avg = float(input("Home goals scored average: "))
        away_goals_scored_avg = float(input("Away goals scored average: "))

        # Calculate engineered features
        rank_difference = home_team_rank - away_team_rank
        combined_goals_avg = (home_goals_scored_avg + away_goals_scored_avg) / 2

        # Create a DataFrame for the input
        data = pd.DataFrame([{
            "home_team_rank": home_team_rank,
            "away_team_rank": away_team_rank,
            "home_recent_wins": home_recent_wins,
            "away_recent_wins": away_recent_wins,
            "home_goals_scored_avg": home_goals_scored_avg,
            "away_goals_scored_avg": away_goals_scored_avg,
            "rank_difference": rank_difference,
            "combined_goals_avg": combined_goals_avg,
        }])

        return data

    def predict(self, input_data):
        """
        Run the prediction using the trained model.
        Args:
            input_data (pd.DataFrame): DataFrame containing the processed input data.
        Returns:
            float: Predicted odds for a home win.
        """
        prediction = self.model.predict(input_data)
        return prediction[0]

    def run(self):
        """
        Full pipeline: Get user input, process data, and make predictions.
        """
        input_data = self.get_user_input()
        prediction = self.predict(input_data)
        print(f"\nPredicted odds for a home win: {prediction:.2f}")
