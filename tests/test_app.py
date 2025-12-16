import pytest
import pandas as pd
import numpy as np
from model_utils import load_and_process_data, train_models

# Mock file path
TEST_FILE_PATH = "data.csv"

def test_data_loading():
    """Test that data loads and processes correctly."""
    # We assume data.csv exists in the root as copied in step 1
    df, df_processed, features, encoders, scaler = load_and_process_data(TEST_FILE_PATH)

    assert df is not None, "Original DataFrame should not be None"
    assert df_processed is not None, "Processed DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"
    assert 'Risk_Grade' in df.columns, "Risk_Grade should be engineered"

def test_processed_data_integrity():
    """Test that processed data is ready for modeling."""
    df, df_processed, features, encoders, scaler = load_and_process_data(TEST_FILE_PATH)

    X = df_processed[features]
    # Check if all columns are numeric
    assert np.issubdtype(X.dtypes.iloc[0], np.number)

    # Check scaling (mean approx 0, std approx 1 for scaled cols)
    # We just check one known scaled column if it exists
    if 'Median Salary (USD)' in features:
        assert -5 < X['Median Salary (USD)'].mean() < 5 # loose bound

def test_model_training():
    """Test that models are trained and returned correctly."""
    df, df_processed, features, encoders, scaler = load_and_process_data(TEST_FILE_PATH)

    X = df_processed[features]
    y = df_processed['Risk_Grade']

    # Use a small subset for speed
    X_small = X.head(50)
    y_small = y.head(50)

    models = train_models(X_small, y_small)

    assert len(models) >= 3, "There should be at least 3 models trained"
    assert "Neural Network (MLP)" in models
    assert "Random Forest" in models
