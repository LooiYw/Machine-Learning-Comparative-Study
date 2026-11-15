import time
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from preprocessing import prepare_xr_features
from shared_pipeline import evaluate_model

def run_svd():
    for is_clean in [False, True]:
        condition = "CLEANED" if is_clean else "CORRUPTED"
        print(f"\n=== MODEL: SVD ({condition} DATA) ===")
        
        X_train, X_test, y_train, y_test, test_df = prepare_xr_features(clean_data=is_clean)
        
        svd_model = Pipeline([
            ('svd', TruncatedSVD(n_components=20, random_state=42)),
            ('regressor', Ridge(random_state=42))
        ])
        
        start_time = time.time()
        print("Training SVD Pipeline...")
        svd_model.fit(X_train, y_train)
        end_time = time.time() - start_time
        print(f"Training completed in {end_time:.2f} seconds")
        
        model_name = f"SVD_{condition}"
        evaluate_model(model_name, svd_model, X_test, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_svd()