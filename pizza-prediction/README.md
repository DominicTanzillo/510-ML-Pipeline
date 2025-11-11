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

## Results

### Model Performance
```
Accuracy: 0.7772 (77.72%)

Classification Report:
                precision    recall  f1-score   support

      No Pizza       0.81      0.93      0.86       609
Pizza Received       0.59      0.31      0.41       199

      accuracy                           0.78       808
     macro avg       0.70      0.62      0.64       808
  weighted avg       0.75      0.78      0.75       808

Confusion Matrix:
[[566  43]
 [137  62]]
```

### Top Features - Top 10
```
                                           feature  importance
           request_number_of_comments_at_retrieval    0.312860
       requester_upvotes_plus_downvotes_at_request    0.081688
         number_of_upvotes_of_request_at_retrieval    0.081643
          requester_account_age_in_days_at_request    0.077466
      requester_upvotes_minus_downvotes_at_request    0.076544
         requester_number_of_subreddits_at_request    0.060952
       number_of_downvotes_of_request_at_retrieval    0.059813
              requester_number_of_posts_at_request    0.059203
           requester_number_of_comments_at_request    0.059098
requester_days_since_first_post_on_raop_at_request    0.056691
```
