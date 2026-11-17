import pandas as pd
import numpy as np
import os
import time as time
from sklearn.model_selection import train_test_split

#region Load Dataset
def load_ratings():
    start_time = time.time()
    df = pd.read_csv("data/ratings.csv", engine="pyarrow")
    print(f"Loaded ratings in {time.time() - start_time:.2f} seconds")
    return df


def load_movies():
    start_time = time.time()
    df = pd.read_csv("data/movies.csv", engine="pyarrow")
    print(f"Loaded movies in {time.time() - start_time:.2f} seconds")
    return df


def load_tags():
    start_time = time.time()
    df = pd.read_csv("data/tags.csv", engine="pyarrow")
    print(f"Loaded tags in {time.time() - start_time:.2f} seconds")
    return df
#endregion 

def apply_data_cleaning(ratings, min_ratings=5, max_ratings_per_day=500):
    print("\n" + "=" * 65)
    print("DATA CLEANING SUMMARY")
    print("=" * 65)

    ratings = ratings.copy()
    original_count = len(ratings)

    print(f"Initial records: {original_count:,}")

    cleaning_summary = []

    # Helper function to track cleaning results
    def log_cleaning(step, before, after):
        removed = before - after
        percentage = (removed / before * 100) if before > 0 else 0

        cleaning_summary.append({
            "Cleaning Operation": step,
            "Records Removed": removed,
            "Records Remaining": after,
            "Percentage Removed": round(percentage, 2)
        })

        print(f"\n{step}")
        print(f"  Records removed   : {removed:,}")
        print(f"  Records remaining : {after:,}")
        print(f"  Percentage removed: {percentage:.2f}%")
    
    # --------------------------------------------------
    # 1. DETECT SUSPICIOUS USERS
    # Combines rapid rating velocity and zero rating variance
    # --------------------------------------------------
    before = len(ratings)

    max_rating_interval = 1       # Seconds
    min_rapid_percentage = 0.80   # At least 80%
    min_rapid_intervals = 3       # Minimum number of rapid intervals

    # Sort ratings chronologically for each user
    ratings = ratings.sort_values(
        ['userId', 'timestamp']
    ).copy()

    # Calculate time difference between consecutive ratings
    ratings['time_difference'] = (
        ratings.groupby('userId')['timestamp'].diff()
    )

    # Calculate rating interval statistics for each user
    user_intervals = ratings.groupby('userId').agg(
        total_intervals=('time_difference', 'count'),
        rapid_intervals=(
            'time_difference',
            lambda x: (x <= max_rating_interval).sum()
        ),
        rating_std=('rating', 'std')
    )

    # Calculate percentage of rapid consecutive ratings
    user_intervals['rapid_percentage'] = (
        user_intervals['rapid_intervals'] /
        user_intervals['total_intervals']
    )

    # Identify users satisfying ALL suspicious behaviour conditions
    suspicious_users = user_intervals[
        (user_intervals['rapid_intervals'] >= min_rapid_intervals) &
        (user_intervals['rapid_percentage'] >= min_rapid_percentage) &
        (user_intervals['rating_std'].fillna(0) == 0)
    ].index

    print("\n[3] Suspicious Users (Rapid Ratings + Constant Ratings)")
    print(f"  Maximum rating interval : {max_rating_interval} second(s)")
    print(f"  Minimum rapid percentage: {min_rapid_percentage * 100:.0f}%")
    print(f"  Minimum rapid intervals : {min_rapid_intervals}")
    print(f"  Zero rating variance    : Required")
    print(f"  Users identified        : {len(suspicious_users):,}")

    # Remove suspicious users
    ratings = ratings[
        ~ratings['userId'].isin(suspicious_users)
    ].copy()

    ratings = ratings.drop(columns=['time_difference'])

    log_cleaning("Suspicious Users", before, len(ratings))


    # --------------------------------------------------
    # 6. REMOVE UNPOPULAR MOVIES
    # Fewer than 5 ratings
    # --------------------------------------------------
    before = len(ratings)

    movie_counts = ratings['movieId'].value_counts()

    unpopular_movies = movie_counts[
        movie_counts < min_ratings
    ].index

    print("\n[6] Unpopular Movies")
    print(f"  Movies identified: {len(unpopular_movies):,}")

    ratings = ratings[
        ratings['movieId'].isin(
            movie_counts[movie_counts >= min_ratings].index
        )
    ].copy()

    log_cleaning("Unpopular Movies", before, len(ratings))

    # --------------------------------------------------
    # 8. FINAL SUMMARY
    # --------------------------------------------------
    final_count = len(ratings)
    total_removed = original_count - final_count

    print("\n" + "=" * 65)
    print("FINAL DATA CLEANING RESULTS")
    print("=" * 65)

    print(f"Original records : {original_count:,}")
    print(f"Final records    : {final_count:,}")
    print(f"Total removed    : {total_removed:,}")

    retention_rate = 0
    if original_count > 0:
        retention_rate = final_count / original_count * 100
        
    print(f"Retention rate   : {retention_rate:.2f}%")

    print("\nDetailed Cleaning Breakdown:")
    print(pd.DataFrame(cleaning_summary).to_string(index=False))

    print("=" * 65)
    
    # --------------------------------------------------
    # 9. SAVE CLEANED DATASET
    # --------------------------------------------------
    os.makedirs('data', exist_ok=True)

    output_path = 'data/cleaned_ratings.csv'

    ratings.to_csv(output_path, index=False)

    print("\n[+] Cleaned dataset saved successfully!")
    print(f"    File location : {output_path}")
    print(f"    Total records : {len(ratings):,}")
    print(f"    Total columns : {len(ratings.columns)}")

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

