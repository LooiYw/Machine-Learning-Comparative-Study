import gc
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
PREPROCESSED_PATH = os.path.join(PROCESSED_DIR, 'ratings_preprocessed.parquet')
SCALED_PATH = os.path.join(PROCESSED_DIR, 'ratings_scaled.parquet')

VALID_MIN, VALID_MAX = 1, 5     # MovieLens 1M ratings are whole stars, 1-5

# Feature scaling, as decided in dataset_analysis.py (section E)
LOG_STANDARDIZE_COLS = ['user_activity', 'movie_popularity']   # skewed counts
MINMAX_COLS = ['age', 'release_year']                          # bounded numerics


# =====================================================================
# Load dataset (MovieLens 1M: '::'-separated .dat files, no header)
#   ratings.dat : UserID::MovieID::Rating::Timestamp
#   movies.dat  : MovieID::Title::Genres            (latin-1 encoded)
#   users.dat   : UserID::Gender::Age::Occupation::Zip-code
# '::' is a multi-character separator, so pandas' python engine is needed.
# =====================================================================

def _read_dat(filename, names):
    start = time.time()
    df = pd.read_csv(
        os.path.join(RAW_DIR, filename),
        sep='::', engine='python', header=None,
        names=names, encoding='latin-1',
    )
    print(f"Loaded {filename}: {df.shape} in {time.time() - start:.2f}s")
    return df


def load_ratings_raw():
    return _read_dat('ratings.dat', ['userId', 'movieId', 'rating', 'timestamp'])


def load_movies():
    return _read_dat('movies.dat', ['movieId', 'title', 'genres'])


def load_users():
    return _read_dat('users.dat', ['userId', 'gender', 'age', 'occupation', 'zip'])


def downcast_ratings(ratings):
    """int32 ids, float32 rating, timestamp dropped."""
    ratings = ratings.drop(columns=['timestamp'], errors='ignore')
    ratings['userId'] = ratings['userId'].astype(np.int32)
    ratings['movieId'] = ratings['movieId'].astype(np.int32)
    ratings['rating'] = ratings['rating'].astype(np.float32)
    return ratings


# =====================================================================
# Feature builders (ML-1M has no tags, so side features come from
# users.dat and from the movie titles / genres)
# =====================================================================

def build_user_features(users):
    index = pd.Index(users['userId'].to_numpy(), name='userId')
    features = pd.DataFrame(index=index)
    features['is_male'] = (users['gender'].to_numpy() == 'M').astype(np.uint8)
    # ML-1M age is already a bucket code: 1, 18, 25, 35, 45, 50, 56
    features['age'] = users['age'].to_numpy().astype(np.float32)

    occ = pd.get_dummies(users['occupation'], prefix='occ', dtype=np.uint8)
    occ.index = index
    # zips can look like "55455-1234"; the first digit is a coarse US region
    zip_region = users['zip'].astype(str).str.strip().str[0]
    zip_region = pd.get_dummies(zip_region, prefix='zip', dtype=np.uint8)
    zip_region.index = index

    features = pd.concat([features, occ, zip_region], axis=1)
    return features[~features.index.duplicated()].sort_index()


def build_movie_features(movies):
    """release_year (parsed from the title) + one-hot genres, indexed by movieId."""
    index = pd.Index(movies['movieId'].to_numpy(), name='movieId')

    year = movies['title'].str.extract(r'\((\d{4})\)\s*$')[0].astype(float)
    year = year.fillna(year.median())
    features = pd.DataFrame({'release_year': year.to_numpy().astype(np.float32)}, index=index)

    genres = movies['genres'].fillna('None').str.get_dummies(sep='|').astype(np.uint8)
    genres.index = index

    features = pd.concat([features, genres], axis=1)
    return features[~features.index.duplicated()].sort_index()


def lookup(table, keys):
    """Row lookup by id with 0 for missing ids; keeps uint8/float32 dtypes."""
    return table.reindex(keys.to_numpy(), fill_value=0)


# =====================================================================
# Cleaning (the data-quality checks from dataset_analysis.py)
# =====================================================================

def clean_ratings(ratings):
    """
    Removes full-row duplicates, duplicate user-movie pairs (keeps the first)
    and ratings outside 1-5 / not whole stars. On the standard ML-1M files
    every one of these is expected to remove 0 rows; the counts are printed
    so you can confirm that.
    """
    print("\n========== CLEANING ==========")
    start = len(ratings)

    n = len(ratings)
    ratings = ratings.drop_duplicates()
    print(f"Full-row duplicates removed       : {n - len(ratings):,}")

    n = len(ratings)
    ratings = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='first')
    print(f"Duplicate user-movie pairs removed: {n - len(ratings):,}")

    n = len(ratings)
    valid = ratings['rating'].between(VALID_MIN, VALID_MAX) & (ratings['rating'] % 1 == 0)
    ratings = ratings[valid]
    print(f"Invalid ratings removed           : {n - len(ratings):,}")

    print(f"Records: {start:,} -> {len(ratings):,}")
    return ratings.reset_index(drop=True)


# =====================================================================
# Feature scaling (fit on TRAIN rows only, applied to all rows)
# =====================================================================

def scale_features(df):
    """
    Same scheme as dataset_analysis.py section E:
      - user_activity, movie_popularity: log1p, then StandardScaler
      - age, release_year: MinMaxScaler to [0, 1]
    Not scaled: rating (target), IDs, user_avg / movie_avg (already on the
    rating scale), and the 0/1 one-hot columns. Scalers are fitted on the
    train rows only so no test/val information leaks in. NaNs (movies unseen
    in train) are left as NaN.
    """
    print("\n========== FEATURE SCALING (fit on train only) ==========")
    df = df.copy()
    is_train = (df['split'] == 'train').to_numpy()
    rows = []

    for col in LOG_STANDARDIZE_COLS:
        values = np.log1p(df[[col]].to_numpy(dtype=np.float64))
        scaler = StandardScaler().fit(values[is_train])
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = scaler.transform(values).ravel().astype(np.float32)
        rows.append((col, 'log1p + StandardScaler', before_min, before_max))

    for col in MINMAX_COLS:
        values = df[[col]].to_numpy(dtype=np.float64)
        scaler = MinMaxScaler().fit(values[is_train])
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = scaler.transform(values).ravel().astype(np.float32)
        rows.append((col, 'MinMaxScaler', before_min, before_max))

    summary = pd.DataFrame(rows, columns=['feature', 'method', 'before_min', 'before_max'])
    summary['train_mean_after'] = [df.loc[is_train, c].mean() for c in summary['feature']]
    summary['train_std_after'] = [df.loc[is_train, c].std() for c in summary['feature']]
    print(summary.round(4).to_string(index=False))
    return df


# =====================================================================
# Core assembly: ratings + side tables -> one feature dataset
# =====================================================================

def assemble_dataset(ratings, movies, users):
    """Split 80/10/10, add train-only stats and side features, return one frame."""
    total = len(ratings)

    train, remaining = train_test_split(ratings, test_size=0.2, random_state=42)
    test, val = train_test_split(remaining, test_size=0.5, random_state=42)
    del remaining
    gc.collect()

    print("\n========== DATA SPLIT ==========")
    for name, part in (('Training', train), ('Testing', test), ('Validation', val)):
        print(f"{name:<11}: {len(part):,} ({len(part) / total * 100:.1f}%)")

    # user / movie stats from train only (no leakage)
    user_stats = train.groupby('userId')['rating'].agg(
        user_avg='mean', user_activity='count'
    ).astype(np.float32)
    movie_stats = train.groupby('movieId')['rating'].agg(
        movie_avg='mean', movie_popularity='count'
    ).astype(np.float32)

    print("\n========== STATISTICS ==========")
    print(f"Users with stats : {len(user_stats):,}")
    print(f"Movies with stats: {len(movie_stats):,}")

    user_features = build_user_features(users)
    movie_features = build_movie_features(movies)
    print("\n========== FEATURES ==========")
    print(f"User features : {user_features.shape}")
    print(f"Movie features: {movie_features.shape} (release_year + genres)")

    def combine(split_ratings, split_name):
        df = split_ratings.join(user_stats, on='userId')
        df = df.join(movie_stats, on='movieId')  # NaN for ids unseen in train
        blocks = [
            lookup(user_features, split_ratings['userId']),
            lookup(movie_features, split_ratings['movieId']),
        ]
        for b in blocks:
            b.index = split_ratings.index
        df = pd.concat([df] + blocks, axis=1)
        df.insert(3, 'split', split_name)
        return df

    parts = [combine(train, 'train'), combine(test, 'test'), combine(val, 'val')]
    del train, test, val
    gc.collect()

    final = pd.concat(parts).sort_index()   # back to original rating order
    del parts
    gc.collect()
    final['split'] = final['split'].astype('category')
    return final


def save_dataset(df, path):
    df.to_parquet(path, index=False)
    print(f"\nSaved: {path}")
    print(f"Shape   : {df.shape}")
    print(f"Splits  : {df['split'].value_counts().to_dict()}")


# =====================================================================
# The two output datasets
# =====================================================================

def build_preprocessed_dataset(ratings, movies, users):
    """Raw ratings, no cleaning, no scaling -> ratings_preprocessed.parquet"""
    print("\n########## PREPROCESSED DATASET ##########")
    start = time.time()
    final = assemble_dataset(downcast_ratings(ratings.copy()), movies, users)
    save_dataset(final, PREPROCESSED_PATH)
    print(f"Elapsed : {time.time() - start:.2f}s")
    return final


def build_scaled_dataset(ratings, movies, users):
    """Cleaned + feature-scaled ratings -> ratings_scaled.parquet"""
    print("\n########## SCALED DATASET ##########")
    start = time.time()
    cleaned = clean_ratings(downcast_ratings(ratings.copy()))
    final = assemble_dataset(cleaned, movies, users)
    final = scale_features(final)
    save_dataset(final, SCALED_PATH)
    print(f"Elapsed : {time.time() - start:.2f}s")
    print(final.head(10))
    return final


# =====================================================================
#region Dataloaders for the processed files
# =====================================================================

def _load_processed(path):
    start = time.time()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run this script to build it first.")
    df = pd.read_parquet(path)
    print(f"Loaded {os.path.basename(path)}: {df.shape} in {time.time() - start:.2f}s")
    return df


def load_ratings_preprocessed():
    """Loads data/processed/ratings_preprocessed.parquet"""
    return _load_processed(PREPROCESSED_PATH)


def load_ratings_scaled():
    """Loads data/processed/ratings_scaled.parquet"""
    return _load_processed(SCALED_PATH)
#endregion

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ratings = load_ratings_raw()
    movies = load_movies()
    users = load_users()

    build_preprocessed_dataset(ratings, movies, users)
    build_scaled_dataset(ratings, movies, users)


if __name__ == "__main__":
    main()