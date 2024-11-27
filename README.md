# E-Commerce Churn Prediction Project

## Overview
This project implements an end-to-end machine learning solution for predicting customer churn in an ecommerce setting using Gradient Boosting (XGBoost).

## Project Structure
- `data/`: Raw customer data
- `src/`: Source code modules
- `models/`: Trained machine learning models
- `notebooks/`: Jupyter notebooks for exploration
- `main.py`: Main execution script

## Setup and Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to preprocess data, engineer features, train the model, and evaluate performance:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python src/train_model.py
   ```

3. Evaluate the model:
   ```
   python src/evaluate_model.py
   ```

4. Make predictions:
   ```
   python src/predict.py

## Key Components
- Data Preprocessing
- Feature Engineering
- XGBoost Model Training
- Model Evaluation

## Performance Metrics
Refer to the console output after running `main.py` for detailed classification report.

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License.
