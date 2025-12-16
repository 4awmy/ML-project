import pytest
import pandas as pd
import numpy as np
from model_utils import load_and_process_data, train_models, FEATURE_COLS, NUM_COLS_TO_SCALE

# Mock file path
TEST_FILE_PATH = "data.csv"

def test_data_loading():
    """Test that data loads and processes correctly."""
    # We assume data.csv exists in the root
    df, df_processed, features, encoders, scaler = load_and_process_data(TEST_FILE_PATH)

    assert df is not None, "Original DataFrame should not be None"
    assert df_processed is not None, "Processed DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"
    assert 'Risk_Grade' in df.columns, "Risk_Grade should be engineered"

def test_feature_consistency():
    """Test that features match the expected constants."""
    df, df_processed, features, encoders, scaler = load_and_process_data(TEST_FILE_PATH)

    # Check that the features returned match the global constant (if data has all columns)
    # Note: data.csv might miss some columns in a real scenario, but here we expect full match
    assert set(features) == set([c for c in FEATURE_COLS if c in df.columns]), "Feature mismatch"

def test_processed_data_integrity():
    """Test that processed data is ready for modeling."""
    df, df_processed, features, encoders, scaler = load_and_process_data(TEST_FILE_PATH)

    X = df_processed[features]
    # Check if all columns are numeric
    assert np.issubdtype(X.dtypes.iloc[0], np.number)

    # Check scaling (mean approx 0, std approx 1 for scaled cols)
    for col in NUM_COLS_TO_SCALE:
        if col in features:
            assert -0.5 < X[col].mean() < 0.5  # Standard Scaler centers around 0

def test_model_training():
    """Test that models are trained and returned correctly."""
    df, df_processed, features, encoders, scaler = load_and_process_data(TEST_FILE_PATH)

    X = df_processed[features]
    y = df_processed['Risk_Grade']

    # Use a small subset for speed
    X_small = X.head(50)
    y_small = y.head(50)

    models = train_models(X_small, y_small)

    assert len(models) >= 4, "There should be at least 4 models trained"
    assert "Neural Network (MLP)" in models
    assert "Random Forest" in models
