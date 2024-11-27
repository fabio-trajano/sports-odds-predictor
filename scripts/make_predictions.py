from src.predictions import Predictor

# Path to the trained model
MODEL_OUTPUT_PATH = "models/sports_odds_model.pkl"

def main():
    """
    Run the prediction pipeline interactively.
    """
    predictor = Predictor(MODEL_OUTPUT_PATH)
    predictor.run()

if __name__ == "__main__":
    main()
