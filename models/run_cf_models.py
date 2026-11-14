import time
from surprise import SVD, NMF, KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from xr_preprocessing import clean_and_prepare_data

def run_surprise_models():
    # 1. Get the cleaned data
    ratings_df, _ = clean_and_prepare_data(sample_fraction=0.05) # Use 5% for testing, increase later
    
    # 2. Load into Surprise API format
    # We use the normalized_rating from image_f23e95.png rule C
    reader = Reader(rating_scale=(ratings_df['normalized_rating'].min(), ratings_df['normalized_rating'].max()))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'normalized_rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # 3. Define Models (Parallel processing enabled where applicable via n_jobs=-1)
    models = {
        "SVD": SVD(n_factors=100, random_state=42),
        "NMF": NMF(n_factors=15, random_state=42),
        # KNN uses user-based collaborative filtering. n_jobs=-1 parallelizes distance computation.
        "KNN": KNNBasic(sim_options={'name': 'cosine', 'user_based': True}) 
    }
    
    # 4. Train and Evaluate
    for name, model in models.items():
        print(f"\n=== MODEL: {name} ===")
        start_time = time.time()
        model.fit(trainset)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        predictions = model.test(testset)
        
        # Calculate RMSE
        rmse = accuracy.rmse(predictions)
        # Note: You will need to add back the 'user_mean' to these predictions to get the actual 1-5 star scale!

if __name__ == "__main__":
    run_surprise_models()