import os
from models.knn import run_knn
from models.svm import run_svm
from models.svd import run_svd
from models.nmf import run_nmf
from models.dt import run_dt
from models.rf import run_rf
from models.xgb import run_xgb
from models.lr import run_lr

def main():
    print("=== Starting Traditional Machine Learning ===")
    
    # Delete old results file so you always have the latest ones
    results_file = 'results/model_evaluation_results.csv'
    if os.path.exists(results_file):
        os.remove(results_file)
        print(f"[*] Deleted old results file. Starting fresh!\n")
    
    run_knn()
    run_svm()
    run_svd()
    run_nmf()
    run_dt()
    run_lr()
    run_xgb()
    run_rf()

    print("\n=== All models have finished running! ===")
    print("Check results/model_evaluation_results.csv for your final comparison.")

if __name__ == "__main__":
    main()