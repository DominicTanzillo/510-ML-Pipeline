# Pizza Prediction - Random Acts of Pizza

Predicting whether pizza requests on Reddit will be successful.

## Setup

1. Download the dataset from Kaggle:
   - Go to: https://www.kaggle.com/datasets/kaggle/random-acts-of-pizza
   - Download `train.json` and place it in this directory

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python pizza_prediction.py
   ```

## Model

The script uses a **Random Forest Classifier** with the following features:
- Account age and activity metrics
- Subreddit participation
- Upvotes/downvotes
- Request engagement metrics

The model automatically:
- Splits data (80% train, 20% test)
- Scales features
- Trains and evaluates
- Shows accuracy, classification report, and feature importance
