import time
from sklearn.neighbors import KNeighborsRegressor
from preprocessing import prepare_xr_features
from shared_pipeline import evaluate_model

def run_knn():
    for is_clean in [False, True]:
        condition = "CLEANED" if is_clean else "CORRUPTED"
        print(f"\n=== MODEL: KNN ({condition} DATA) ===")
        
        X_train, X_test, y_train, y_test, test_df = prepare_xr_features(clean_data=is_clean)
        knn_model = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)
        
        start_time = time.time()
        print("Training KNN...")
        knn_model.fit(X_train, y_train)
        end_time = time.time() - start_time
        print(f"Training completed in {end_time:.2f} seconds")
        
        model_name = f"KNN_{condition}"
        evaluate_model(model_name, knn_model, X_test, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_knn()