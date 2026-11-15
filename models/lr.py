import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from preprocessing import prepare_raw_features, prepare_cleaned_features
from shared_pipeline import evaluate_model

def run_lr():
    print("=== MODEL 4: LINEAR REGRESSION ===")
    
    
    # PART 1: EVALUATE ON RAW (CORRUPTED) DATA
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, test_df_raw = prepare_raw_features(sample_fraction=0.1)
    
    # Scale data for Linear Regression to balance feature weights
    scaler_raw = StandardScaler()
    X_train_raw_scaled = scaler_raw.fit_transform(X_train_raw)
    X_test_raw_scaled = scaler_raw.transform(X_test_raw)
    
    lr_model_raw = LinearRegression(n_jobs=-1)
    
    print("\nTraining Linear Regression on RAW data...")
    start_time = time.time()
    lr_model_raw.fit(X_train_raw_scaled, y_train_raw)
    training_time_raw = time.time() - start_time
    print(f"Training completed in {training_time_raw:.4f} seconds")
    
    evaluate_model("Linear Regression (Raw)", lr_model_raw, X_test_raw_scaled, y_test_raw, test_df_raw, training_time_raw)


    # PART 2: EVALUATE ON CLEANED DATA
    X_train_clean, X_test_clean, y_train_clean, y_test_clean, test_df_clean = prepare_cleaned_features(sample_fraction=0.1)
    
    # Scale cleaned data
    scaler_clean = StandardScaler()
    X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean)
    X_test_clean_scaled = scaler_clean.transform(X_test_clean)
    
    lr_model_clean = LinearRegression(n_jobs=-1)
    
    print("\nTraining Linear Regression on CLEANED data...")
    start_time = time.time()
    lr_model_clean.fit(X_train_clean_scaled, y_train_clean)
    training_time_clean = time.time() - start_time
    print(f"Training completed in {training_time_clean:.4f} seconds")
    
    evaluate_model("Linear Regression (Clean)", lr_model_clean, X_test_clean_scaled, y_test_clean, test_df_clean, training_time_clean)

if __name__ == "__main__":
    run_lr()