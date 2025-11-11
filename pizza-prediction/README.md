# Pizza Prediction - Random Acts of Pizza

<img width="947" height="259" alt="Screenshot 2025-11-11 at 12 46 00 PM" src="https://github.com/user-attachments/assets/673da6ef-b547-463e-ae54-a5849deebbec" />

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

   **Option A: Command Line**
   ```bash
   python pizza_prediction.py
   ```

   **Option B: Interactive Dashboard**
   ```bash
   streamlit run app.py
   ```
   This live dashboard let you:
   - Watch the model train in real-time
   - Adjust parameters with sliders
   - See interactive visualizations
   - View feature importance, confusion matrix, ROC curve

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

<img width="791" height="446" alt="Screenshot 2025-11-11 at 12 45 41 PM" src="https://github.com/user-attachments/assets/d6cc16d9-c941-444b-b966-36a52e9e5d97" />

### Top Features - Top 10

<img width="809" height="420" alt="Screenshot 2025-11-11 at 12 45 47 PM" src="https://github.com/user-attachments/assets/0562c3d5-45ae-46fe-8121-75fdd42577fc" />

