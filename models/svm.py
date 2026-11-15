import time
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from preprocessing import prepare_raw_features, prepare_cleaned_features
from shared_pipeline import evaluate_model

def run_svm():
    # 1. RAW DATA
    print("\n=== MODEL: Linear SVM (RAW CORRUPTED DATA) ===")
    X_train, X_test, y_train, y_test, test_df = prepare_raw_features()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearSVR(random_state=42, dual=False, loss='squared_epsilon_insensitive', max_iter=2000)
    
    start_time = time.time()
    print("Training SVM (Raw)...")
    model.fit(X_train_scaled, y_train)
    end_time = time.time() - start_time
    evaluate_model("SVM_CORRUPTED", model, X_test_scaled, y_test, test_df, end_time, k=20)

    # 2. CLEANED DATA
    print("\n=== MODEL: Linear SVM (CLEANED ROBUST DATA) ===")
    X_train, X_test, y_train, y_test, test_df = prepare_cleaned_features()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearSVR(random_state=42, dual=False, loss='squared_epsilon_insensitive', max_iter=2000)
    
    start_time = time.time()
    print("Training SVM (Cleaned)...")
    model.fit(X_train_scaled, y_train)
    end_time = time.time() - start_time
    evaluate_model("SVM_CLEANED", model, X_test_scaled, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_svm()