import time
from surprise import Dataset, Reader, KNNBaseline
from xr_preprocessing import get_base_split
from shared_pipeline import evaluate_model
from models.svd import SurpriseWrapper

def run_knn():
    print("=== MODEL: Item-Based KNN (KNNBaseline) ===")
    # Keep fraction <= 0.03 so the item-item similarity matrix fits comfortably in RAM
    train_df, test_df = get_base_split(sample_fraction=0.02)
    
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    
    # Item-based collaborative filtering with Pearson baseline shrinkage
    sim_options = {
        'name': 'pearson_baseline',
        'user_based': False  # CRITICAL: Item-based prevents OOM crash on 200k+ users
    }
    
    knn_algo = KNNBaseline(k=40, min_k=3, sim_options=sim_options, verbose=True)
    
    print("\nTraining Item-KNN...")
    start_time = time.time()
    knn_algo.fit(trainset)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    wrapped_model = SurpriseWrapper(knn_algo)
    evaluate_model("Item-KNN", wrapped_model, test_df, test_df['rating'].values, test_df, k=20)

if __name__ == "__main__":
    run_knn()