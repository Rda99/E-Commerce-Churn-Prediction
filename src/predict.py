# --- predict.py ---
import joblib
import numpy as np

def predict(model_path, input_data):
    """Make predictions with the trained model."""
    model = joblib.load(model_path)
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    sample_data = np.array([[30, 120, 500, 1]])  # Example feature input
    prediction = predict("models/churn_model.pkl", sample_data)
    print("Prediction:", prediction)