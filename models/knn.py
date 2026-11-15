import time
from sklearn.neighbors import KNeighborsRegressor
from preprocessing import prepare_raw_features, prepare_cleaned_features
from shared_pipeline import evaluate_model

def run_knn():
    # 1. RAW DATA
    print("\n=== MODEL: KNN (RAW CORRUPTED DATA) ===")
    X_train, X_test, y_train, y_test, test_df = prepare_raw_features()
    model = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)  # Fully Parallel
    
    start_time = time.time()
    print("Training KNN (Raw)...")
    model.fit(X_train, y_train)
    end_time = time.time() - start_time
    evaluate_model("KNN_CORRUPTED", model, X_test, y_test, test_df, end_time, k=20)

    # 2. CLEANED DATA
    print("\n=== MODEL: KNN (CLEANED ROBUST DATA) ===")
    X_train, X_test, y_train, y_test, test_df = prepare_cleaned_features()
    model = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)  # Fully Parallel
    
    start_time = time.time()
    print("Training KNN (Cleaned)...")
    model.fit(X_train, y_train)
    end_time = time.time() - start_time
    evaluate_model("KNN_CLEANED", model, X_test, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_knn()