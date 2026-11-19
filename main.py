import os
import pandas as pd

from preprocessing import load_ratings_preprocessed, load_ratings_scaled
# from models.knn import run_knn
# from models.svm import run_svm
# from models.svd import run_svd
# from models.nmf import run_nmf
from models.lr import run_lr
from models.dt import run_dt
from models.rf import run_rf
from models.xgboost import run_xgb


def main():
    print("=== Starting Traditional Machine Learning ===")

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'model_evaluation_results.csv')

    if os.path.exists(results_file):
        os.remove(results_file)
        print("[*] Deleted old results file. Starting fresh!\n")

    # Load both datasets once
    datasets = [
        ("Preprocessed (Unscaled)", load_ratings_preprocessed()),
        ("Scaled", load_ratings_scaled()),
    ]

    models = [
        #("KNN", run_knn),
        #("SVM", run_svm),
        #("SVD", run_svd),
        #("NMF", run_nmf),
        ("Ridge / Linear Regression", run_lr),
        ("Decision Tree", run_dt),
        ("Random Forest", run_rf),
        ("XGBoost", run_xgb),
    ]

    all_results = []

    for dataset_label, df in datasets:
        for model_name, run_fn in models:
            _, metrics = run_fn(df, dataset_label)
            all_results.append({
                'model': model_name,
                'dataset': dataset_label,
                **metrics,
            })

            # Save after each model finishes so progress isn't lost if a later model stops
            pd.DataFrame(all_results).to_csv(results_file, index=False)

    print("\n=== All models have finished running! ===")
    print(f"Check {results_file} for your final comparison.")


if __name__ == "__main__":
    main()