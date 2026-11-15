import time
from sklearn.tree import DecisionTreeRegressor
from shared_pipeline import prepare_traditional_features, evaluate_model

def run_decision_tree():
    print("=== MODEL 1: DECISION TREE ===")
    X_train, X_test, y_train, y_test, test_df = prepare_traditional_features()
    
    # Initialize model
    dt_model = DecisionTreeRegressor(random_state=42)
    
    # Train
    print("\nTraining Decision Tree...")
    start_time = time.time()
    dt_model.fit(X_train, y_train)
    end_time = time.time() - start_time
    print(f"Training completed in {end_time:.2f} seconds")
    
    # Evaluate
    evaluate_model("Decision Tree", dt_model, X_test, y_test, test_df, end_time, k=20)

if __name__ == "__main__":
    run_decision_tree()