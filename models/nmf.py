import time
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from preprocessing import prepare_raw_features, prepare_cleaned_features
from shared_pipeline import evaluate_model

def run_nmf():
    # 1. RAW DATA
    print("\n=== MODEL: NMF (RAW CORRUPTED DATA) ===")
    X_train, X_test, y_train, y_test, test_df = prepare_raw_features()
    model = Pipeline([('nmf', NMF(n_components=20, init='nndsvda', random_state=42, max_iter=500)), ('regressor', Ridge(random_state=42))])
    
    start_time = time.time()
    print("Training NMF (Raw)...")
    model.fit(X_train, y_train)
    end_time = time.time() - start_time
    evaluate_model("NMF_CORRUPTED", model, X_test, y_test, test_df, end_time, k=20)

    # 2. CLEANED DATA
    print("\n=== MODEL: NMF (CLEANED ROBUST DATA) ===")
    X_train, X_test, y_train, y_test, test_df = prepare_cleaned_features()
    model = Pipeline([('nmf', NMF(n_components=20, init='nndsvda', random_state=42, max_iter=500)), ('regressor', Ridge(random_state=42))])
    
    start_time = time.time()
    print("Training NMF (Cleaned)...")
    model.fit(X_train, y_train)
    end_time = time.time() - start_time
    evaluate_model("NMF_CLEANED", model, X_test, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_nmf()