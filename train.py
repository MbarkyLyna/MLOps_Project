import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Start MLflow run
with mlflow.start_run(run_name="LinearRegression-Demo"):
    # Generate data
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions & metrics
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Log everything
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("n_samples", 100)
    
    # Log the model itself
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Logged model with R² = {r2:.4f}")
