import time
from sklearn.ensemble import RandomForestRegressor
from shared_pipeline import prepare_traditional_features, evaluate_model

def run_random_forest():
    print("=== MODEL 2: RANDOM FOREST ===")
    X_train, X_test, y_train, y_test, test_df = prepare_traditional_features()
    
    # Initialize model with parallel processing (n_jobs=-1 uses all CPU cores)
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Train
    print("\nTraining Random Forest (Parallel Processing Enabled)...")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    end_time = time.time() - start_time
    print(f"Training completed in {end_time:.2f} seconds")
    
    # Evaluate
    evaluate_model("Random Forest", rf_model, X_test, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_random_forest()