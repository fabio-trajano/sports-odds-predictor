# Sports Odds Predictor

## Overview

Sports Odds Predictor is a machine learning project designed to predict the probability of a home team winning in sports matches using **Random Forest** and **Gradient Boosting** models . The project provides a pipeline for data processing, model training, evaluation, and predictive analysis.

## Features

### Data Processing
- Clean and preprocesd raw data into features and target.
- Rank difference and combined goals average calculations

### Machine Learning Model
- **Random Forest Regression** for odds prediction.
- **Gradient Boosting Regression** for odds prediction.
- Robust performance evaluation metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Square Error (RMSE)
  - R-squared (R²) score

### Prediction Capabilities
- Match data input trough CLI
- Flexible prediction output (console or CSV)
- Detailed odds analysis

### **API Features**
- **Two Prediction Endpoints:**
  - `/api/predict/random_forest/` for predictions using the Random Forest model.
  - `/api/predict/gradient_boosting/` for predictions using the Gradient Boosting model.
- **Home Page Endpoint:** `/api/` provides an overview of available endpoints and example input.
- Flexible prediction output in JSON format.

### Visualization Tools
- Feature distribution analysis
- Feature-target correlation visualization
- Model residuals plotting
- Feature importance graphics

## Prerequisites

- Python 3.8+
- Git
- Git LFS (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/sports-odds-predictor.git
cd sports-odds-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Git LFS:
```bash
git lfs install
```

## Usage

### Data Preprocessing
```bash
python scripts/run_preprocessing.py
```

### Model Training
```bash
python scripts/train_model.py
```

### Making Predictions
```bash
python scripts/make_predictions.py
```
- ### Or make predictions via Django API:

#### 1. Start the Django Server
```bash
cd api
python manage.py runserver
```

#### 2. Make Predictions Use one of the following endpoints to make predictions:
- ### Random Forest Model
  -  URL: `http://127.0.0.1:8000/api/predict/random_forest/`
  - Body Json:
```bash
{
    "home_team_rank": 2,
    "away_team_rank": 3,
    "home_recent_wins": 5,
    "away_recent_wins": 4,
    "home_goals_scored_avg": 2.7,
    "away_goals_scored_avg": 2.5
}
```
- ### Gradient Boosting Model
  -  URL: `http://127.0.0.1:8000/api/predict/gradient_boosting/`

### Visualization
```bash
python scripts/visualize_results.py
```



## License

Distributed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for more information.
