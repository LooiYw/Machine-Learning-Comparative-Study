import time
import numpy as np
from joblib import Parallel, delayed
from surprise import Dataset, Reader, SVD
from xr_preprocessing import get_base_split
from shared_pipeline import evaluate_model

class SurpriseWrapper:
    """Wraps Surprise models to support .predict(X_test) and parallel inference."""
    def __init__(self, algo):
        self.algo = algo

    def _predict_chunk(self, uids, iids):
        return [self.algo.predict(u, i).est for u, i in zip(uids, iids)]

    def predict(self, X_test):
        # X_test is expected to have 'userId' and 'movieId'
        uids = X_test['userId'].values
        iids = X_test['movieId'].values
        
        # Parallelize prediction across all CPU cores
        n_jobs = 8
        uid_chunks = np.array_split(uids, n_jobs)
        iid_chunks = np.array_split(iids, n_jobs)
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._predict_chunk)(u_c, i_c) 
            for u_c, i_c in zip(uid_chunks, iid_chunks)
        )
        return np.concatenate(results)

def run_svd():
    print("=== MODEL: SVD (Matrix Factorization) ===")
    train_df, test_df = get_base_split(sample_fraction=0.1)
    
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    
    svd_algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    
    print("\nTraining SVD...")
    start_time = time.time()
    svd_algo.fit(trainset)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    wrapped_model = SurpriseWrapper(svd_algo)
    evaluate_model("SVD", wrapped_model, test_df, test_df['rating'].values, test_df, k=20)

if __name__ == "__main__":
    run_svd()