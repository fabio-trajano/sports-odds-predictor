# Sports Odds Predictor

## Overview

Sports Odds Predictor is a machine learning project designed to predict the probability of a home team winning in sports matches using Random Forest Algorithm. The project provides a pipeline for data processing, model training, evaluation, and predictive analysis.

## Features

### Data Processing
- Clean and preprocesd raw data into features and target.
- Rank difference and combined goals average calculations

### Machine Learning Model
- Random Forest regression for odds prediction
- Robust performance evaluation metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Square Error (RMSE)
  - R-squared (R²) score

### Prediction Capabilities
- Match data input trough CLI
- Flexible prediction output (console or CSV)
- Detailed odds analysis

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
PYTHONPATH=$(pwd) python3 scripts/run_preprocessing.py
```

### Model Training
```bash
PYTHONPATH=$(pwd) python3 scripts/train_model.py
```

### Making Predictions
```bash
PYTHONPATH=$(pwd) python3 scripts/make_predictions.py
```

### Visualization
```bash
PYTHONPATH=$(pwd) python3 scripts/visualize_results.py
```

## Project Structure
```
sports-odds-predictor/
│
├── data/               # Raw and processed data
├── models/             # Trained model files
├── scripts/            # Executable scripts
│   ├── run_preprocessing.py
│   ├── train_model.py
│   ├── make_predictions.py
│   └── visualize_results.py
│
├── src/                # Source code
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
```


## License

Distributed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for more information.
