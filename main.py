from models.knn import run_knn
from models.svm import run_svm
from models.svd import run_svd
from models.nmf import run_nmf

if __name__ == "__main__":
    print("Starting XR ML models (Raw vs Clean)...")
    run_knn()
    run_svm()
    run_svd()
    run_nmf()
    print("Finished! Check the 'results' and 'models' folders.")