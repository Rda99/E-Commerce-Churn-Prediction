# --- train_model.py ---
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from scripts.preprocess import load_data, preprocess_data, split_data

def train_model(data_path, model_path):
    """Train a Gradient Boosting model."""
    # Load and preprocess data
    df = load_data(data_path)
    features, target = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(features, target)

    # Train model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model("data/ecommerce_churn_data.csv", "models/churn_model.pkl")
