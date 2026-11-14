import time
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xr_preprocessing import prepare_xr_features
from shared_pipeline import evaluate_model

def run_nmf():
    for is_clean in [False, True]:
        condition = "CLEANED" if is_clean else "CORRUPTED"
        print(f"\n=== MODEL: NMF ({condition} DATA) ===")
        
        X_train, X_test, y_train, y_test, test_df = prepare_xr_features(clean_data=is_clean)
        
        nmf_model = Pipeline([
            ('nmf', NMF(n_components=20, init='nndsvda', random_state=42, max_iter=500)),
            ('regressor', Ridge(random_state=42))
        ])
        
        start_time = time.time()
        print("Training NMF Pipeline...")
        nmf_model.fit(X_train, y_train)
        end_time = time.time() - start_time
        print(f"Training completed in {end_time:.2f} seconds")
        
        model_name = f"NMF_{condition}"
        evaluate_model(model_name, nmf_model, X_test, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_nmf()