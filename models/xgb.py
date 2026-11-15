import time
import xgboost as xgb
from preprocessing import prepare_raw_features, prepare_cleaned_features
from shared_pipeline import evaluate_model

def run_xgb():
    print("=== MODEL 3: XGBOOST ===")
    
    
    # PART 1: EVALUATE ON RAW (CORRUPTED) DATA
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, test_df_raw = prepare_raw_features(sample_fraction=0.1)
    
    xgb_model_raw = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
    print("\nTraining XGBoost on RAW data (Parallel Processing)...")
    start_time = time.time()
    xgb_model_raw.fit(X_train_raw, y_train_raw)
    training_time_raw = time.time() - start_time
    print(f"Training completed in {training_time_raw:.2f} seconds")
    
    evaluate_model("XGBoost (Raw)", xgb_model_raw, X_test_raw, y_test_raw, test_df_raw, training_time_raw)

    # PART 2: EVALUATE ON CLEANED DATA
    X_train_clean, X_test_clean, y_train_clean, y_test_clean, test_df_clean = prepare_cleaned_features(sample_fraction=0.1)
    
    xgb_model_clean = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
    print("\nTraining XGBoost on CLEANED data (Parallel Processing)...")
    start_time = time.time()
    xgb_model_clean.fit(X_train_clean, y_train_clean)
    training_time_clean = time.time() - start_time
    print(f"Training completed in {training_time_clean:.2f} seconds")
    
    evaluate_model("XGBoost (Clean)", xgb_model_clean, X_test_clean, y_test_clean, test_df_clean, training_time_clean)

if __name__ == "__main__":
    run_xgb()