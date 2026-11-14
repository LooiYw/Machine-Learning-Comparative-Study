import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from preprocessing import load_ratings, load_movies, load_tags

def get_base_split(sample_fraction=0.05):
    """
    Loads ratings, applies sampling, extracts timestamp features, 
    and performs a consistent 80/20 train-test split.
    """
    print(f"\n--- Loading & Splitting Data (Fraction: {sample_fraction}) ---")
    ratings = load_ratings()
    
    if sample_fraction < 1.0:
        ratings = ratings.sample(frac=sample_fraction, random_state=42)
        
    # 1. USE TIMESTAMP: Convert Unix timestamp to meaningful temporal features
    dt = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['rating_year'] = dt.dt.year
    ratings['rating_month'] = dt.dt.month
    ratings['rating_dayofweek'] = dt.dt.dayofweek
    # Days since the earliest rating in the sample
    ratings['days_since_start'] = (ratings['timestamp'] - ratings['timestamp'].min()) / 86400.0
    
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    return train_df.copy(), test_df.copy()


def prepare_svm_features(sample_fraction=0.02, max_tag_features=100):
    """
    Builds sparse feature matrices for SVM using:
    - Leakage-free User & Movie stats
    - Timestamp features
    - Movie genres
    - Aggregated user-generated tags from tags.csv (TF-IDF)
    """
    train_df, test_df = get_base_split(sample_fraction=sample_fraction)
    movies = load_movies()
    tags = load_tags()
    
    # 2. LEAKAGE-FREE STATS (Computed ONLY on train_df)
    global_mean = train_df['rating'].mean()
    user_stats = train_df.groupby('userId')['rating'].agg(['mean', 'count']).rename(
        columns={'mean': 'user_avg', 'count': 'user_activity'}
    )
    movie_stats = train_df.groupby('movieId')['rating'].agg(['mean', 'count']).rename(
        columns={'mean': 'movie_avg', 'count': 'movie_popularity'}
    )
    
    # 3. USE TAGS.CSV: Group all tags per movie into a single document
    tags['tag'] = tags['tag'].fillna('').astype(str)
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    # One-hot encode genres
    movies['genres'] = movies['genres'].fillna('None')
    genres_encoded = movies['genres'].str.get_dummies(sep='|')
    movies_meta = pd.concat([movies[['movieId']], genres_encoded], axis=1)
    movies_meta = movies_meta.merge(movie_tags, on='movieId', how='left')
    movies_meta['tag'] = movies_meta['tag'].fillna('')
    
    # Merge features into train and test
    def enrich(df):
        df = df.join(user_stats, on='userId').join(movie_stats, on='movieId')
        df['user_avg'] = df['user_avg'].fillna(global_mean)
        df['movie_avg'] = df['movie_avg'].fillna(global_mean)
        df['user_activity'] = df['user_activity'].fillna(0)
        df['movie_popularity'] = df['movie_popularity'].fillna(0)
        df = df.merge(movies_meta, on='movieId', how='left')
        df['tag'] = df['tag'].fillna('')
        return df

    train_enriched = enrich(train_df)
    test_enriched = enrich(test_df)
    
    # TF-IDF on tags (Fit on train ONLY, transform test)
    print("Vectorizing tags with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=max_tag_features, stop_words='english')
    X_train_tags = tfidf.fit_transform(train_enriched['tag'])
    X_test_tags = tfidf.transform(test_enriched['tag'])
    
    # Numeric & Genre columns
    drop_cols = ['userId', 'movieId', 'rating', 'timestamp', 'tag']
    num_cols = [c for c in train_enriched.columns if c not in drop_cols]
    
    # Scale numeric features (Critical for SVM convergence)
    scaler = StandardScaler()
    X_train_num = csr_matrix(scaler.fit_transform(train_enriched[num_cols].fillna(0)))
    X_test_num = csr_matrix(scaler.transform(test_enriched[num_cols].fillna(0)))
    
    # Combine sparse numeric/genre matrix with sparse TF-IDF matrix
    X_train = hstack([X_train_num, X_train_tags]).tocsr()
    X_test = hstack([X_test_num, X_test_tags]).tocsr()
    
    y_train = train_enriched['rating'].values
    y_test = test_enriched['rating'].values
    
    return X_train, X_test, y_train, y_test, test_enriched