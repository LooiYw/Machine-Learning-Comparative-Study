import time
from sklearn.svm import LinearSVR
from xr_preprocessing import prepare_svm_features
from shared_pipeline import evaluate_model

def run_svm():
    print("=== MODEL: SVM (LinearSVR) ===")
    # Keep fraction around 0.02 - 0.05 (640k - 1.6M rows) for fast SVM training
    X_train, X_test, y_train, y_test, test_df = prepare_svm_features(sample_fraction=0.02)
    
    svm_model = LinearSVR(
        C=1.0,
        epsilon=0.1,
        dual=False,       # Faster when n_samples > n_features
        max_iter=2000,
        random_state=42
    )
    
    print("\nTraining LinearSVR...")
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Uses your teammate's exact evaluation & CSV logger
    evaluate_model("SVM (LinearSVR)", svm_model, X_test, y_test, test_df, k=20)

if __name__ == "__main__":
    run_svm()