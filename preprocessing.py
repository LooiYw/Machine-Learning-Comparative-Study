import gc
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'uncleaned_ratings.parquet')


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
# Full pipeline: raw files -> one final dataset
# =====================================================================

def build_uncleaned_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    total_start = time.time()

    # ---------------- 1. LOAD ----------------
    ratings = downcast_ratings(load_ratings_raw())
    movies = load_movies()
    users = load_users()
    total = len(ratings)

    # ---------------- 2. SPLIT 80 / 10 / 10 ----------------
    train, remaining = train_test_split(ratings, test_size=0.2, random_state=42)
    del ratings
    test, val = train_test_split(remaining, test_size=0.5, random_state=42)
    del remaining
    gc.collect()

    print("\n========== DATA SPLIT ==========")
    for name, part in (('Training', train), ('Testing', test), ('Validation', val)):
        print(f"{name:<11}: {len(part):,} ({len(part) / total * 100:.1f}%)")

    # ---------------- 3. USER / MOVIE STATS (train only, no leakage) ----------------
    user_stats = train.groupby('userId')['rating'].agg(
        user_avg='mean', user_activity='count'
    ).astype(np.float32)
    movie_stats = train.groupby('movieId')['rating'].agg(
        movie_avg='mean', movie_popularity='count'
    ).astype(np.float32)

    print("\n========== STATISTICS ==========")
    print(f"Users with stats : {len(user_stats):,}")
    print(f"Movies with stats: {len(movie_stats):,}")

    # ---------------- 4. SIDE FEATURES ----------------
    user_features = build_user_features(users)
    movie_features = build_movie_features(movies)
    print("\n========== FEATURES ==========")
    print(f"User features : {user_features.shape}")
    print(f"Movie features: {movie_features.shape} (release_year + genres)")

    # ---------------- 5. JOIN EVERYTHING ----------------
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

    # ---------------- 6. SAVE THE FINAL DATASET ONLY ----------------
    final.to_parquet(OUTPUT_PATH, index=False)

    print("\n========== FINAL DATASET SAVED ==========")
    print(f"File    : {OUTPUT_PATH}")
    print(f"Shape   : {final.shape}")
    print(f"Splits  : {final['split'].value_counts().to_dict()}")
    print(f"Elapsed : {time.time() - total_start:.2f}s")
    print(final.head(10))

    return final


def main():
    build_uncleaned_dataset()


if __name__ == "__main__":
    main()