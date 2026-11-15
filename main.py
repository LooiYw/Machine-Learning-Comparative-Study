import os
from models.knn import run_knn
from models.svm import run_svm
from models.svd import run_svd
from models.nmf import run_nmf

# When you finish your other 3 models, uncomment their imports:
# from models.decision_tree import run_dt
# from models.random_forest import run_rf
# from models.xgboost import run_xgb

def main():
    print("=== Starting Traditional Machine Learning ===")
    
    # --- NEW: Delete old results file so you always have the latest ones ---
    results_file = 'results/model_evaluation_results.csv'
    if os.path.exists(results_file):
        os.remove(results_file)
        print(f"[*] Deleted old results file. Starting fresh!\n")
    
    run_knn()
    run_svm()
    run_svd()
    run_nmf()

    # When you are ready for the other 3, uncomment them here:
    # run_dt()
    # run_rf()
    # run_xgb()
    
    print("\n=== All models have finished running! ===")
    print("Check results/model_evaluation_results.csv for your final comparison.")

if __name__ == "__main__":
    main()