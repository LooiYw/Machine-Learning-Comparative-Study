import time
from surprise import Dataset, Reader, NMF
from xr_preprocessing import get_base_split
from shared_pipeline import evaluate_model
from models.svd import SurpriseWrapper  # Reuses the parallel wrapper

def run_nmf():
    print("=== MODEL: NMF (Non-Negative Matrix Factorization) ===")
    train_df, test_df = get_base_split(sample_fraction=0.1)
    
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    
    nmf_algo = NMF(n_factors=30, n_epochs=25, biased=True, random_state=42)
    
    print("\nTraining NMF...")
    start_time = time.time()
    nmf_algo.fit(trainset)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    wrapped_model = SurpriseWrapper(nmf_algo)
    evaluate_model("NMF", wrapped_model, test_df, test_df['rating'].values, test_df, k=20)

if __name__ == "__main__":
    run_nmf()