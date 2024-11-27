# --- preprocess.py ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data."""
    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    # Encoding categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Feature scaling
    scaler = StandardScaler()
    features = df.drop('Churn', axis=1)
    target = df['Churn']
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target

def split_data(features, target):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test