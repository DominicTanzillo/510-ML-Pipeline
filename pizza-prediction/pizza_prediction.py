import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import kagglehub
from pathlib import Path

# Load data
print("Loading data...")
# Download the dataset from Kaggle using kagglehub
dataset_path = kagglehub.dataset_download("kaggle/random-acts-of-pizza")
print(f"Dataset downloaded to: {dataset_path}")

# Load the train.json file
train_json_path = Path(dataset_path) / "train.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Select numeric features
numeric_features = [
    'requester_account_age_in_days_at_request',
    'requester_days_since_first_post_on_raop_at_request',
    'requester_number_of_comments_at_request',
    'requester_number_of_comments_in_raop_at_request',
    'requester_number_of_posts_at_request',
    'requester_number_of_posts_on_raop_at_request',
    'requester_number_of_subreddits_at_request',
    'requester_upvotes_minus_downvotes_at_request',
    'requester_upvotes_plus_downvotes_at_request',
    'number_of_upvotes_of_request_at_retrieval',
    'number_of_downvotes_of_request_at_retrieval',
    'request_number_of_comments_at_retrieval'
]

# Prepare features and target
X = df[numeric_features].fillna(0)
y = df['requester_received_pizza'].astype(int)

print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Percentage received pizza: {y.mean()*100:.2f}%")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Pizza', 'Pizza Received']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))
