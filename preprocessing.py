import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

#region Load dataset functions
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

def load_links():
    start_time = time.time()
    df = pd.read_csv("data/links.csv", engine="pyarrow")
    print(f"Loaded links in {time.time() - start_time:.2f} seconds")
    return df

def load_tags():
    start_time = time.time()
    df = pd.read_csv("data/tags.csv", engine="pyarrow")
    print(f"Loaded tags in {time.time() - start_time:.2f} seconds")
    return df

#endregion

def inspect_dataset(dataframe, file_name):
    start_time = time.time()
    print(file_name, " Dataset Dimensions")
    rows, columns = dataframe.shape
    print("=" * 20)
    print("Number of rows: ", rows)
    print("Number of columns: ", columns)
    print("=" * 20)
    print("\nData Type")
    print(dataframe.dtypes)
    print("=" * 20)
    print("\nMissing Values")
    missing_values = pd.DataFrame({"Missing Count": dataframe.isnull().sum()})
    print(missing_values)
    print("=" * 20)
    print("\nFirst 3 Records:")
    print(dataframe.head())
    print("=" * 20)
    end_time = time.time() - start_time
    print("Total Inspection Time for", file_name, f": {end_time:.2f} seconds")

def apply_data_cleaning(ratings, min_ratings=5, max_ratings_per_day=500):
    """
    Cleans the dataset by removing suspicious bot activity, dropping sparse 
    users/items to prevent overfitting, and mean-centering ratings to normalize bias.
    """
    print("\n[+] Running Data Cleaning Process...")
    initial_size = len(ratings)

    # Filter Out Suspicious Users
    user_counts = ratings['userId'].value_counts()
    suspicious_users = user_counts[user_counts > max_ratings_per_day].index
    ratings = ratings[~ratings['userId'].isin(suspicious_users)]

    user_std = ratings.groupby('userId')['rating'].std()
    zero_variance_users = user_std[user_std == 0].index
    ratings = ratings[~ratings['userId'].isin(zero_variance_users)]

    # Drop Sparse Items and Users
    active_users = user_counts[user_counts >= min_ratings].index
    ratings = ratings[ratings['userId'].isin(active_users)]
    
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= min_ratings].index
    ratings = ratings[ratings['movieId'].isin(popular_movies)]

    # Normalize User Rating Bias
    user_means = ratings.groupby('userId')['rating'].transform('mean')
    ratings['rating_normalized'] = ratings['rating'] - user_means
    
    print(f"[-] Dropped {initial_size - len(ratings)} noisy/sparse records.")
    return ratings

def build_tag_features(tags, top_n_tags=50):
    """
    Builds tag-based features from tags.csv
    """
    tags = tags.dropna(subset=['tag']).copy()
    tags['tag'] = tags['tag'].astype(str).str.lower().str.strip()
    tags = tags[tags['tag'] != '']

    movie_tag_stats = tags.groupby('movieId')['tag'].agg(
        movie_tag_count='count',
        movie_unique_tags='nunique'
    )

    top_tags = tags['tag'].value_counts().head(top_n_tags).index
    top_df = tags[tags['tag'].isin(top_tags)]
    tag_matrix = top_df.groupby(['movieId', 'tag']).size().unstack(fill_value=0)
    tag_matrix = (tag_matrix > 0).astype(np.int8)
    tag_matrix.columns = [f"tag_{c}" for c in tag_matrix.columns]

    movie_tag_features = movie_tag_stats.join(tag_matrix, how='left').fillna(0)
    user_tag_features = tags.groupby('userId').size().to_frame('user_tag_count')

    return movie_tag_features, user_tag_features

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


def prepare_traditional_features(sample_fraction=0.1, top_n_tags=50, clean_data=False):
    """
    Generates statistical features for traditional ML models. 
    Accepts a boolean flag to toggle between raw data and cleaned data for robustness testing.
    """
    print(f"\n--- Starting Feature Engineering (Clean Data: {clean_data}) ---")
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()
    
    print(movies[movies["movieId"] == 1])
    print(tags[tags["movieId"] == 1])

    #region Inspect datasets
    # inspect_dataset(ratings, "Ratings") #32000204 rows
    # inspect_dataset(movies, "Movies Metadata")
    # inspect_dataset(keywords, "Keywords")
    # inspect_dataset(tags, "Tags")
    #endregion
    
    

if __name__ == "__main__":
    # Just a quick test to ensure loading works
    ratings_df = load_ratings()
    inspect_dataset(ratings_df, "Ratings")