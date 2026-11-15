import time
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from preprocessing import prepare_xr_features
from shared_pipeline import evaluate_model

def run_svm():
    for is_clean in [False, True]:
        condition = "CLEANED" if is_clean else "CORRUPTED"
        print(f"\n=== MODEL: Linear SVM ({condition} DATA) ===")
        
        X_train, X_test, y_train, y_test, test_df = prepare_xr_features(clean_data=is_clean)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # FIX IS HERE: Added loss='squared_epsilon_insensitive'
        svm_model = LinearSVR(
            random_state=42, 
            dual=False, 
            loss='squared_epsilon_insensitive', 
            max_iter=2000
        )
        
        start_time = time.time()
        print("Training Linear SVM...")
        svm_model.fit(X_train_scaled, y_train)
        end_time = time.time() - start_time
        print(f"Training completed in {end_time:.2f} seconds")
        
        model_name = f"SVM_{condition}"
        evaluate_model(model_name, svm_model, X_test_scaled, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_svm()