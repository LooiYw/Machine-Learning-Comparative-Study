import time
import xgboost as xgb
from shared_pipeline import prepare_traditional_features, evaluate_model

def run_xgboost():
    print("=== MODEL 3: XGBOOST ===")
    X_train, X_test, y_train, y_test, test_df = prepare_traditional_features()
    
    # Initialize model with parallel processing and histogram binning for speed
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        random_state=42,
        tree_method='hist',  # Crucial for fast processing on large datasets
        n_jobs=-1            # Uses all available CPU cores
    )
    
    # Train
    print("\nTraining XGBoost (Parallel Processing Enabled)...")
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Evaluate
    evaluate_model("XGBoost", xgb_model, X_test, y_test, test_df, k=20)

if __name__ == "__main__":
    run_xgboost()