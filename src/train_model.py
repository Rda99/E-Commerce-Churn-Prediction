# --- train_model.py ---
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
print(os.getcwd())

from preprocess import load_data,  preprocess_data, split_data

def train_model(data_path, model_path):
    """Train a Gradient Boosting model."""
    # Ensure the directory for the model exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Load and preprocess data
    df = load_data(data_path)
    features, target = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(features, target)

    # Train model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model("Data/E Commerce Dataset.xlsx", "models/churn_model.pkl")
