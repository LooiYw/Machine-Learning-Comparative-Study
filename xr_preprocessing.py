import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from shared_pipeline import load_movies, load_ratings, load_tags
from preprocessing import build_tag_features

# ==========================================
# 1. DATA CLEANING (ROBUSTNESS TESTING)
# ==========================================
def apply_data_cleaning(ratings, min_ratings=5, max_ratings_per_day=500):
    print("\n[+] Applying Preprocessing Pipeline (Cleaning Data)...")
    
    # Filter suspicious users (bot/shilling attacks)
    user_counts = ratings['userId'].value_counts()
    suspicious_users = user_counts[user_counts > max_ratings_per_day].index
    ratings = ratings[~ratings['userId'].isin(suspicious_users)]

    user_std = ratings.groupby('userId')['rating'].std()
    zero_variance_users = user_std[user_std == 0].index
    ratings = ratings[~ratings['userId'].isin(zero_variance_users)]

    # Drop sparse data
    active_users = user_counts[user_counts >= min_ratings].index
    ratings = ratings[ratings['userId'].isin(active_users)]
    
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= min_ratings].index
    ratings = ratings[ratings['movieId'].isin(popular_movies)]

    # Normalize bias
    user_means = ratings.groupby('userId')['rating'].transform('mean')
    ratings['rating_normalized'] = ratings['rating'] - user_means
    
    return ratings

# ==========================================
# 2. FEATURE PREPARATION
# ==========================================
def prepare_xr_features(sample_fraction=0.1, top_n_tags=50, clean_data=False):
    # clean_data=False represents "Before Cleaning / Corrupted Data"
    # clean_data=True represents "After Cleaning / Robust Data"
    
    ratings = load_ratings().sample(frac=sample_fraction, random_state=42)
    movies = load_movies()
    tags = load_tags()

    if clean_data:
        ratings = apply_data_cleaning(ratings)
        target_column = 'rating_normalized'
    else:
        print("\n[-] Using Corrupted/Before Cleaning Data...")
        target_column = 'rating'

    user_stats = ratings.groupby('userId')['rating'].agg(['mean', 'count']).rename(
        columns={'mean': 'user_avg', 'count': 'user_activity'}
    )
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).rename(
        columns={'mean': 'movie_avg', 'count': 'movie_popularity'}
    )
    
    movie_tag_features, user_tag_features = build_tag_features(tags, top_n_tags=top_n_tags)
    
    df = ratings.join(user_stats, on='userId').join(movie_stats, on='movieId').join(user_tag_features, on='userId')
    
    movies['genres'] = movies['genres'].fillna('None')
    genres_encoded = movies['genres'].str.get_dummies(sep='|')
    movies_features = pd.concat([movies[['movieId']], genres_encoded], axis=1)
    
    df = df.merge(movies_features, on='movieId', how='left').join(movie_tag_features, on='movieId').fillna(0)
    
    drop_cols = ['rating', 'timestamp', 'userId', 'movieId']
    if 'rating_normalized' in df.columns:
        drop_cols.append('rating_normalized')
        
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols), train_df[target_column], test_df[target_column], test_df

# ==========================================
# 3. EVALUATION
# ==========================================
def evaluate_xr_model(model_name, model, X_test, y_test, test_df, training_time, k=20, threshold=3.5):
    print(f"\nEvaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    eval_df = test_df[['userId', 'movieId', 'rating']].copy()
    eval_df['predicted_rating'] = y_pred
    
    # 1. RMSE & MAE
    rmse = np.sqrt(mean_squared_error(eval_df['rating'], eval_df['predicted_rating']))
    mae = mean_absolute_error(eval_df['rating'], eval_df['predicted_rating'])
    
    # 2. Classification Accuracy
    eval_df['actual_relevant'] = (eval_df['rating'] >= threshold).astype(int)
    eval_df['predicted_relevant'] = (eval_df['predicted_rating'] >= threshold).astype(int)
    accuracy = accuracy_score(eval_df['actual_relevant'], eval_df['predicted_relevant'])

    # 3. Precision@K, Recall@K, F1-Score@K
    def calculate_top_k(group):
        total_relevant = group['actual_relevant'].sum()
        if total_relevant == 0:
            return pd.Series({'precision': np.nan, 'recall': np.nan})
        
        top_k_items = group.nlargest(k, 'predicted_rating')
        hits = top_k_items['actual_relevant'].sum()
        
        precision = hits / k
        recall = hits / total_relevant
        return pd.Series({'precision': precision, 'recall': recall})

    user_metrics = eval_df.groupby('userId').apply(calculate_top_k).dropna()
    precision_k = user_metrics['precision'].mean()
    recall_k = user_metrics['recall'].mean()
    
    if (precision_k + recall_k) > 0:
        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
    else:
        f1_k = 0.0

    # Save to CSV exactly matching the image columns
    os.makedirs('results', exist_ok=True)
    file_path = 'results/xr_model_results.csv'
    results_df = pd.DataFrame({
        'Model Name': [model_name], 
        'RMSE': [round(rmse, 4)], 
        'MAE': [round(mae, 4)],
        'Classification Accuracy': [round(accuracy, 4)], 
        f'Precision@{k}': [round(precision_k, 4)], 
        f'Recall@{k}': [round(recall_k, 4)], 
        f'F1-Score@{k}': [round(f1_k, 4)]
    })
    results_df.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    safe_name = model_name.replace(" ", "_").lower()
    with open(f'models/{safe_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print(f"[+] Saved {model_name} metrics to results/xr_model_results.csv")