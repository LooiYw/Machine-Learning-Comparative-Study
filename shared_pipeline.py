import pandas as pd
import numpy as np
import time
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from preprocessing import (
    load_movies,
    load_ratings,
    load_tags
)

# ==========================================
# 1. MODEL PREPROCESSING (Feature Engineering)
# ==========================================
def build_tag_features(tags, top_n_tags=50):
    """
    Builds tag-based features from tags.csv (columns: userId, movieId, tag, timestamp).

    Returns two DataFrames:
      - movie_tag_features (index=movieId):
            movie_tag_count   : total number of tags applied to the movie
            movie_unique_tags : number of distinct tags applied to the movie
            tag_<name>        : 1 if the movie has received that tag (top_n_tags most frequent tags)
      - user_tag_features (index=userId):
            user_tag_count    : how many tags the user has written (engagement signal)
    """
    tags = tags.dropna(subset=['tag']).copy()
    tags['tag'] = tags['tag'].astype(str).str.lower().str.strip()
    tags = tags[tags['tag'] != '']

    # --- Movie-level counts ---
    movie_tag_stats = tags.groupby('movieId')['tag'].agg(
        movie_tag_count='count',
        movie_unique_tags='nunique'
    )

    # --- Top-N tags as binary columns per movie ---
    top_tags = tags['tag'].value_counts().head(top_n_tags).index
    top_df = tags[tags['tag'].isin(top_tags)]
    tag_matrix = (
        top_df.groupby(['movieId', 'tag']).size().unstack(fill_value=0)
    )
    tag_matrix = (tag_matrix > 0).astype(np.int8)
    # Prefix avoids collisions with genre column names (e.g. genre "Comedy" vs tag "comedy")
    tag_matrix.columns = [f"tag_{c}" for c in tag_matrix.columns]

    movie_tag_features = movie_tag_stats.join(tag_matrix, how='left').fillna(0)

    # --- User-level tag activity ---
    user_tag_features = tags.groupby('userId').size().to_frame('user_tag_count')

    return movie_tag_features, user_tag_features


def prepare_traditional_features(sample_fraction=0.1, top_n_tags=50):
    """
    Tree models cannot handle raw IDs. We must create statistical features.
    Note: sample_fraction is set to 10% by default to prevent memory crashes on 32M rows.
    Change to 1.0 when running on a high-RAM server.

    top_n_tags: number of most frequent tags (from tags.csv) to one-hot encode per movie.
    """
    print("\n--- Starting Feature Engineering ---")
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()
    
    # Sample data to manage memory during development testing
    ratings = ratings.sample(frac=sample_fraction, random_state=42)
    
    # Create User Features (Average rating, activity level)
    user_stats = ratings.groupby('userId')['rating'].agg(['mean', 'count']).rename(
        columns={'mean': 'user_avg', 'count': 'user_activity'}
    )
    
    # Create Movie Features (Average rating, popularity)
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).rename(
        columns={'mean': 'movie_avg', 'count': 'movie_popularity'}
    )
    
    # Create Tag Features (from tags.csv)
    print("Building tag features...")
    movie_tag_features, user_tag_features = build_tag_features(tags, top_n_tags=top_n_tags)
    del tags  # free memory
    
    # Merge stats into main dataframe
    df = (
        ratings
        .join(user_stats, on='userId')
        .join(movie_stats, on='movieId')
        .join(user_tag_features, on='userId')
    )
    
    # One-Hot Encode Genres
    movies['genres'] = movies['genres'].fillna('None')
    genres_encoded = movies['genres'].str.get_dummies(sep='|')
    movies_features = pd.concat([movies[['movieId']], genres_encoded], axis=1)
    
    # Final Merge (genres + movie-level tag features)
    df = df.merge(movies_features, on='movieId', how='left')
    df = df.join(movie_tag_features, on='movieId')
    df = df.fillna(0)  # Users/movies with no tags (or any other gaps) get 0
    
    # Define Features (X) and Target (y)
    # Drop timestamp and string labels; keep IDs only for the test set evaluation later
    drop_cols = ['rating', 'timestamp', 'userId', 'movieId']
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['rating']
    
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['rating']
    
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, test_df

# ==========================================
# 2. EVALUATION & RESULT SAVING
# ==========================================


def save_results_to_csv(model_name, rmse, mae, accuracy, avg_precision, avg_recall, f1_score, k,
                        training_time_seconds):

    os.makedirs('results', exist_ok=True)
    file_path = 'results/model_evaluation_results.csv'

    results_dict = {
        'Model Name': [model_name],
        'RMSE': [round(rmse, 4)],
        'MAE': [round(mae, 4)],
        'Classification Accuracy': [round(accuracy, 4)],
        f'Precision@{k}': [round(avg_precision, 4)],
        f'Recall@{k}': [round(avg_recall, 4)],
        f'F1-Score@{k}': [round(f1_score, 4)],
        'Training Time (s)': [round(training_time_seconds, 4)],
    }

    results_df = pd.DataFrame(results_dict)

    file_exists = os.path.isfile(file_path)

    results_df.to_csv(
        file_path,
        mode='a' ,
        header=not file_exists,
        index=False,
        lineterminator='\n'
    )

    print(f"[+] Results successfully saved to {file_path}")



def evaluate_model(model_name, model, X_test, y_test, test_df, training_time_seconds, k=20, threshold=3.5):
    """
    Calculates metrics, saves results to CSV, AND saves the trained model to a .pkl file.
    """
    print(f"\nEvaluating {model_name}...")
    start_time = time.time()
    
    # Predict ratings
    y_pred = model.predict(X_test)
    
    # Append predictions back to the test dataframe
    eval_df = test_df[['userId', 'movieId', 'rating']].copy()
    eval_df['predicted_rating'] = y_pred
    
    # Classification logic
    eval_df['actual_relevant'] = (eval_df['rating'] >= threshold).astype(int)
    eval_df['predicted_relevant'] = (eval_df['predicted_rating'] >= threshold).astype(int)

    # 1. Error Metrics
    rmse = np.sqrt(mean_squared_error(eval_df['rating'], eval_df['predicted_rating']))
    mae = mean_absolute_error(eval_df['rating'], eval_df['predicted_rating'])
    
    # 2. Classification Accuracy
    accuracy = accuracy_score(eval_df['actual_relevant'], eval_df['predicted_relevant'])

    # 3. Top-N Metrics
    precisions, recalls = [], []
    user_groups = eval_df.groupby('userId')
    for user_id, group in user_groups:
        if len(group) < k:
            continue
        top_k = group.nlargest(k, 'predicted_rating')
        n_relevant_and_recommended = top_k['actual_relevant'].sum()
        n_total_relevant = group['actual_relevant'].sum()
        
        precisions.append(n_relevant_and_recommended / k)
        recalls.append(n_relevant_and_recommended / n_total_relevant if n_total_relevant != 0 else 0)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) != 0 else 0

    print(f"Evaluation finished in {time.time() - start_time:.2f} seconds")
    
    # Save the results to CSV

    save_results_to_csv(model_name, rmse, mae, accuracy,avg_precision, avg_recall, f1_score, k,training_time_seconds)
    
    # SAVE THE TRADITIONAL MODEL (.pkl format)
    os.makedirs('saved_models', exist_ok=True)
    safe_name = model_name.replace(" ", "_").lower()
    model_path = f'saved_models/{safe_name}_model.pkl'
    
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
        
    print(f"[+] Model successfully saved to {model_path}")