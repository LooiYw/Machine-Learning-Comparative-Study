import pandas as pd
import numpy as np
import time as time
from sklearn.model_selection import train_test_split


# ==========================================
# 1. DATA LOADING METHODS
# ==========================================

# These methods load the raw CSV files using PyArrow for faster reading.
def load_ratings(): 
    return pd.read_csv("data/ratings.csv", engine="pyarrow")

def load_movies(): 
    return pd.read_csv("data/movies.csv", engine="pyarrow")

def load_tags():
    start_time = time.time()
    df = pd.read_csv("data/tags.csv", engine="pyarrow")
    print(f"Loaded tags in {time.time() - start_time:.2f} seconds")
    return df

#endregion



# ==========================================
# 2. DATA CLEANING PIPELINE
# ==========================================

# This method removes noisy data (bots, inactive users) to make the model more robust.
def apply_data_cleaning(ratings, min_ratings=5, max_ratings_per_day=500):
    print("\n[+] Applying Preprocessing Pipeline (Cleaning Data)...")
    
    # Remove bots or scrapers (users rating hundreds of movies in a single day)
    user_counts = ratings['userId'].value_counts()
    ratings = ratings[~ratings['userId'].isin(user_counts[user_counts > max_ratings_per_day].index)]

    # Remove users who give the exact same rating to every movie (Standard Deviation = 0)
    user_std = ratings.groupby('userId')['rating'].std()
    ratings = ratings[~ratings['userId'].isin(user_std[user_std == 0].index)]

    # Remove inactive users who haven't rated enough movies to establish a preference
    active_users = user_counts[user_counts >= min_ratings].index
    ratings = ratings[ratings['userId'].isin(active_users)]
    
    # Remove unpopular movies that don't have enough ratings to be recommended reliably
    movie_counts = ratings['movieId'].value_counts()
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_ratings].index)]

    # Normalize the ratings by subtracting the user's average rating
    # This helps adjust for "harsh" vs "generous" reviewers
    user_means = ratings.groupby('userId')['rating'].transform('mean')
    ratings['rating_normalized'] = ratings['rating'] - user_means
    
    return ratings


# ==========================================
# 3. FEATURE ENGINEERING METHODS
# ==========================================

# This method extracts meaningful patterns from the text tags left by users.
def build_tag_features(tags, top_n_tags=50):
    # Clean the text data by removing empties, lowercasing, and stripping spaces
    tags = tags.dropna(subset=['tag']).copy()
    tags['tag'] = tags['tag'].astype(str).str.lower().str.strip()
    tags = tags[tags['tag'] != '']

    # Count how many total tags and unique tags a movie has
    movie_tag_stats = tags.groupby('movieId')['tag'].agg(movie_tag_count='count', movie_unique_tags='nunique')
    
    # Find the most commonly used tags across the whole dataset
    top_tags = tags['tag'].value_counts().head(top_n_tags).index
    
    # Create a binary matrix (0 or 1) indicating if a movie has a specific top tag
    tag_matrix = tags[tags['tag'].isin(top_tags)].groupby(['movieId', 'tag']).size().unstack(fill_value=0)
    tag_matrix = (tag_matrix > 0).astype(np.int8)
    tag_matrix.columns = [f"tag_{c}" for c in tag_matrix.columns]

    # Combine movie tag stats with the binary matrix
    movie_tag_features = movie_tag_stats.join(tag_matrix, how='left').fillna(0)
    
    # Count how many tags a user has written (shows user engagement)
    user_tag_features = tags.groupby('userId').size().to_frame('user_tag_count')
    
    return movie_tag_features, user_tag_features


# This is the core engine that joins all tables (Ratings, Movies, Tags) together.
def _build_dataset(ratings, movies, tags, top_n_tags, target_column):
    # Calculate average ratings and total activity for users and movies
    user_stats = ratings.groupby('userId')['rating'].agg(user_avg='mean', user_activity='count')
    movie_stats = ratings.groupby('movieId')['rating'].agg(movie_avg='mean', movie_popularity='count')
    
    # Get tag features
    movie_tag_features, user_tag_features = build_tag_features(tags, top_n_tags)
    
    # Join everything into one large dataframe
    df = ratings.join(user_stats, on='userId').join(movie_stats, on='movieId').join(user_tag_features, on='userId')
    
    # Break down the genres (e.g., 'Action|Comedy' becomes two separate columns)
    movies['genres'] = movies['genres'].fillna('None')
    genres_encoded = movies['genres'].str.get_dummies(sep='|')
    movies_features = pd.concat([movies[['movieId']], genres_encoded], axis=1)
    
    # Merge genres and tags into the main dataframe, filling missing values with 0
    df = df.merge(movies_features, on='movieId', how='left').join(movie_tag_features, on='movieId').fillna(0)
    
    # Identify which columns are IDs or labels that shouldn't be used for training
    drop_cols = ['rating', 'timestamp', 'userId', 'movieId']
    if 'rating_normalized' in df.columns:
        drop_cols.append('rating_normalized')
        
    # Split into 80% training data and 20% testing data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Return X_train, X_test, y_train, y_test, and the raw test_df
    return train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols), train_df[target_column], test_df[target_column], test_df


# ==========================================
# 4. FINAL EXPORT PIPELINES
# ==========================================

# Pipeline 1: Builds the dataset using RAW (corrupted) data
def prepare_raw_features(sample_fraction=0.1, top_n_tags=50):
    print("\n[-] Processing Raw/Corrupted Dataset...")
    ratings = load_ratings().sample(frac=sample_fraction, random_state=42)
    return _build_dataset(ratings, load_movies(), load_tags(), top_n_tags, target_column='rating')


# Pipeline 2: Builds the dataset using CLEANED (robust) data
def prepare_cleaned_features(sample_fraction=0.1, top_n_tags=50):
    print("\n[+] Processing Cleaned/Robust Dataset...")
    ratings = load_ratings().sample(frac=sample_fraction, random_state=42)
    ratings = apply_data_cleaning(ratings)
    return _build_dataset(ratings, load_movies(), load_tags(), top_n_tags, target_column='rating_normalized')

