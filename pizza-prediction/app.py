import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import kagglehub
from pathlib import Path
import time

st.set_page_config(page_title="Pizza Prediction Dashboard", layout="wide", page_icon="🍕")

# Title
st.title("🍕 Pizza Prediction - Live Training Dashboard")
st.markdown("---")

# Sidebar for model parameters
st.sidebar.header("🎛️ Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)
max_depth = st.sidebar.slider("Max Depth", 3, 20, 10)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42)

start_training = st.sidebar.button("🚀 Start Training", type="primary")

# Load data function
@st.cache_data
def load_data():
    with st.spinner("📥 Downloading dataset from Kaggle..."):
        dataset_path = kagglehub.dataset_download("kaggle/random-acts-of-pizza")
        train_json_path = Path(dataset_path) / "train.json"

        with open(train_json_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
    return df

# Load the data
try:
    df = load_data()
    st.sidebar.success(f"✅ Data loaded: {df.shape[0]} samples")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Feature selection
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

# Prepare data
X = df[numeric_features].fillna(0)
y = df['requester_received_pizza'].astype(int)

# Display data info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Samples", len(df))
with col2:
    st.metric("✅ Pizza Received", y.sum())
with col3:
    st.metric("❌ No Pizza", len(y) - y.sum())
with col4:
    st.metric("📈 Success Rate", f"{y.mean()*100:.1f}%")

st.markdown("---")

if start_training:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create placeholders for live updates
    progress_bar = st.progress(0)
    status_text = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        metrics_placeholder = st.empty()
    with col2:
        tree_progress_placeholder = st.empty()

    st.markdown("---")

    # Training section
    st.subheader("🎯 Model Performance")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        accuracy_placeholder = st.empty()
        report_placeholder = st.empty()

    with perf_col2:
        confusion_placeholder = st.empty()

    st.markdown("---")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        feature_imp_placeholder = st.empty()

    with viz_col2:
        roc_placeholder = st.empty()

    # Train model with progress tracking
    status_text.text("🏋️ Training Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        warm_start=True  # Allow incremental training
    )

    # Train incrementally to show progress
    trees_per_batch = max(5, n_estimators // 20)
    current_trees = 0

    for i in range(1, n_estimators + 1, trees_per_batch):
        current_trees = min(i + trees_per_batch - 1, n_estimators)
        model.n_estimators = current_trees
        model.fit(X_train_scaled, y_train)

        # Update progress
        progress = current_trees / n_estimators
        progress_bar.progress(progress)
        status_text.text(f"🌲 Building trees: {current_trees}/{n_estimators}")

        # Show intermediate metrics
        y_pred_train = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_pred_train)

        with tree_progress_placeholder:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[current_trees],
                y=[train_acc],
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Training Accuracy'
            ))
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Number of Trees",
                yaxis_title="Training Accuracy",
                yaxis_range=[0.5, 1.0],
                xaxis_range=[0, n_estimators]
            )
            st.plotly_chart(fig, use_container_width=True, key=f"progress_{current_trees}")

        time.sleep(0.1)  # Small delay for visual effect

    # Final predictions
    status_text.text("📊 Generating predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['No Pizza', 'Pizza Received'], output_dict=True)

    status_text.text("✅ Training Complete!")
    progress_bar.progress(1.0)

    # Display accuracy
    with accuracy_placeholder:
        st.success(f"### 🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Classification report
        st.text("Classification Report:")
        report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

    # Confusion matrix heatmap
    with confusion_placeholder:
        fig_cm = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Pizza', 'Pizza Received'],
            y=['No Pizza', 'Pizza Received'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    with feature_imp_placeholder:
        feature_importance = pd.DataFrame({
            'feature': numeric_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_imp = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features",
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    # ROC Curve
    with roc_placeholder:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain')
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Success message
    st.balloons()
    st.success("🎉 Training completed successfully!")

else:
    st.info("👈 Adjust parameters in the sidebar and click 'Start Training' to begin!")

    # Show sample data
    st.subheader("📋 Sample Data Preview")
    st.dataframe(df[numeric_features + ['requester_received_pizza']].head(10), use_container_width=True)

    # Show feature distributions
    st.subheader("📊 Target Distribution")
    fig_dist = px.pie(
        values=[len(y) - y.sum(), y.sum()],
        names=['No Pizza', 'Pizza Received'],
        title='Pizza Request Outcomes',
        color_discrete_sequence=['#ff6b6b', '#51cf66']
    )
    st.plotly_chart(fig_dist, use_container_width=True)
