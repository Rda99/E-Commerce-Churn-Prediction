# --- evaluate_model.py ---
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score

def evaluate_model(model_path, X_test, y_test):
    """Evaluate the trained model."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Metrics
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))