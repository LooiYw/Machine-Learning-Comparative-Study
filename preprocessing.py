import pandas as pd
import numpy as np
import os
import time as time
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:                      # no-op fallback
        def __init__(self, *a, **k): pass
        def set_description(self, *a, **k): pass
        def update(self, *a, **k): pass
        def close(self): pass
        @staticmethod
        def write(msg): print(msg)

#region Load Dataset
def load_ratings_raw():
    start_time = time.time()
    df = pd.read_csv("data/raw/ratings.csv", engine="pyarrow")
    print(f"Loaded ratings in {time.time() - start_time:.2f} seconds")
    return df


def load_movies():
    start_time = time.time()
    df = pd.read_csv("data/raw/movies.csv", engine="pyarrow")
    print(f"Loaded movies in {time.time() - start_time:.2f} seconds")
    return df


def load_tags():
    start_time = time.time()
    df = pd.read_csv("data/raw/tags.csv", engine="pyarrow")
    print(f"Loaded tags in {time.time() - start_time:.2f} seconds")
    return df


#endregion 

import gc
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = 'data/processed'


# =====================================================================
# KEEP YOUR EXISTING CODE HERE (it was not in the part you pasted):
#   - load_ratings_raw(), load_tags(), load_movies()
#   - the EDA helpers (inspect_dataset, dataset_overview, etc.)
# =====================================================================


def _downcast_ratings(ratings):
    """
    Shrinks the ratings table: int32 ids, float32 rating, no timestamp
    (it is dropped before modelling anyway). Cuts memory by more than half
    and every later join/copy gets cheaper too.
    """
    ratings = ratings.drop(columns=['timestamp'], errors='ignore')
    ratings['userId'] = ratings['userId'].astype(np.int32)
    ratings['movieId'] = ratings['movieId'].astype(np.int32)
    ratings['rating'] = ratings['rating'].astype(np.float32)
    return ratings


# =====================================================================
# STAGE 1 — clean ratings, save to disk
# =====================================================================

def apply_data_cleaning(ratings, thresholds=(10, 20)):
    """Unchanged logic; only the extra full-table copy on entry is gone."""
    print("\n" + "=" * 65)
    print("DATA CLEANING SUMMARY")
    print("=" * 65)

    original_count = len(ratings)
    print(f"Initial records: {original_count:,}")

    cleaning_summary = []

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

    os.makedirs('data', exist_ok=True)

    before = len(ratings)
    movie_counts = ratings['movieId'].value_counts()
    results = {}

    for n in thresholds:
        unpopular_movies = movie_counts[movie_counts < n].index

        print(f"\nUnpopular Movies (< {n} ratings)")
        print(f"  Movies identified: {len(unpopular_movies):,}")

        cleaned = ratings[
            ratings['movieId'].isin(movie_counts[movie_counts >= n].index)
        ].copy()

        log_cleaning(f"Unpopular Movies (< {n} ratings)", before, len(cleaned))
        results[n] = cleaned

    print("\n" + "=" * 65)
    print("FINAL DATA CLEANING RESULTS")
    print("=" * 65)
    print(f"Original records : {original_count:,}")

    for n, cleaned in results.items():
        final_count = len(cleaned)
        total_removed = original_count - final_count
        retention_rate = (final_count / original_count * 100) if original_count > 0 else 0
        print(f"\nThreshold: < {n} ratings removed")
        print(f"  Final records  : {final_count:,}")
        print(f"  Total removed  : {total_removed:,}")
        print(f"  Retention rate : {retention_rate:.2f}%")

    print("\nDetailed Cleaning Breakdown:")
    print(pd.DataFrame(cleaning_summary).to_string(index=False))
    print("=" * 65)

    return results


def clean_and_save_ratings(thresholds=(10, 20)):
    """STAGE 1. Run alone, then comment it out in main()."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ratings = _downcast_ratings(load_ratings_raw())
    gc.collect()
    cleaning_results = apply_data_cleaning(ratings, thresholds=thresholds)
    del ratings

    # save and release one threshold at a time
    for n in list(cleaning_results):
        cleaned = cleaning_results.pop(n)
        out_path = os.path.join(PROCESSED_DIR, f'ratings_cleaned_{n}.parquet')
        cleaned.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}  (rows={len(cleaned):,})")
        del cleaned
        gc.collect()


# =====================================================================
# STAGE 2 — build tag features from raw tags, save the FEATURES to disk
# =====================================================================

def build_tag_features(tags, top_n_tags=None, pbar=None):
    def step(label):
        if pbar is not None:
            pbar.set_description(label)
            pbar.update(1)

    step("Cleaning tag text")
    t = pa.Table.from_pandas(tags[['userId', 'movieId', 'tag']], preserve_index=False)
    t = t.filter(pc.is_valid(t['tag']))
    cleaned = pc.utf8_trim_whitespace(pc.utf8_lower(t['tag']))
    t = t.set_column(t.schema.get_field_index('tag'), 'tag', cleaned)
    t = t.filter(pc.not_equal(t['tag'], ''))
    tqdm.write(f"Original Tags Shape: {(t.num_rows, t.num_columns)}")

    step("Counting tags per movie and user")
    movie_tag_stats = (
        t.group_by('movieId')
        .aggregate([('tag', 'count'), ('tag', 'count_distinct')])
        .to_pandas()
        .rename(columns={'tag_count': 'movie_tag_count',
                         'tag_count_distinct': 'movie_unique_tags'})
        .set_index('movieId').sort_index()
        .astype(np.float32)
    )
    user_tag_features = (
        t.group_by('userId').aggregate([('tag', 'count')])
        .to_pandas()
        .rename(columns={'tag_count': 'user_tag_count'})
        .set_index('userId').sort_index()
        .astype(np.float32)
    )

    step("Selecting tags")
    if top_n_tags is not None:
        vc = pc.value_counts(t['tag'].combine_chunks())
        counts = pd.DataFrame({'tag': vc.field('values').to_pylist(),
                               'n': vc.field('counts').to_numpy()})
        top_tags = counts.nlargest(top_n_tags, 'n')['tag'].tolist()
        t = t.filter(pc.is_in(t['tag'], value_set=pa.array(top_tags)))

        # --- histogram: number of movies per kept tag ---
    movies_per_tag = (
        t.group_by('tag')
        .aggregate([('movieId', 'count_distinct')])
        .to_pandas()
        .rename(columns={'movieId_count_distinct': 'n_movies'})
    )
    counts = movies_per_tag['n_movies'].to_numpy()

    tqdm.write(f"\nMovies per tag: min={counts.min():,}, median={int(np.median(counts)):,}, "
               f"max={counts.max():,}  ({len(counts):,} tags)")

    import matplotlib
    matplotlib.use('Agg')            # save to file; no window to block the script
    import matplotlib.pyplot as plt

    # tag popularity is heavily skewed, so use log-spaced bins and a log y-axis
    bins = np.unique(np.logspace(0, np.log10(counts.max()), 40).astype(int))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(counts, bins=bins, edgecolor='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of movies the tag appears on (log scale)')
    ax.set_ylabel('Number of tags (log scale)')
    ax.set_title(f'Movies per tag ({len(counts):,} tags)')
    fig.tight_layout()

    hist_path = os.path.join(PROCESSED_DIR, 'movies_per_tag_hist.png')
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    tqdm.write(f"Saved histogram: {hist_path}\n")

    step("Building tag matrix")
    tag_col = pc.dictionary_encode(t['tag'].combine_chunks())
    tag_codes = tag_col.indices.to_numpy()
    tag_names = tag_col.dictionary.to_pylist()
    movie_ids, movie_codes = np.unique(t['movieId'].to_numpy(), return_inverse=True)
    tqdm.write(f"Distinct tags kept: {len(tag_names):,}")

    matrix = np.zeros((len(movie_ids), len(tag_names)), dtype=np.uint8)
    matrix[movie_codes, tag_codes] = 1
    tag_matrix = pd.DataFrame(
        matrix,
        index=pd.Index(movie_ids, name='movieId'),
        columns=[f"tag_{c}" for c in tag_names],
    )
    del matrix

    step("Joining movie features")
    movie_tag_features = movie_tag_stats.join(tag_matrix, how='left')
    tag_cols = tag_matrix.columns
    movie_tag_features[tag_cols] = movie_tag_features[tag_cols].fillna(0).astype(np.uint8)

    return movie_tag_features, user_tag_features


def clean_and_save_tags(top_n_tags=None):
    """STAGE 2, with a progress bar. Run alone, then comment it out."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with tqdm(total=8, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        pbar.set_description("Loading raw tags")
        tags = load_tags()
        pbar.update(1)

        movie_tag_features, user_tag_features = build_tag_features(
            tags, top_n_tags=top_n_tags, pbar=pbar     # advances 5 steps
        )
        del tags

        movie_path = os.path.join(PROCESSED_DIR, 'movie_tag_features.parquet')
        user_path = os.path.join(PROCESSED_DIR, 'user_tag_features.parquet')

        pbar.set_description("Saving movie_tag_features.parquet")
        movie_tag_features.to_parquet(movie_path)
        pbar.update(1)

        pbar.set_description("Saving user_tag_features.parquet")
        user_tag_features.to_parquet(user_path)
        pbar.update(1)

    print(f"Saved: {movie_path}  (shape={movie_tag_features.shape})")
    print(f"Saved: {user_path}  (shape={user_tag_features.shape})")


# =====================================================================
# STAGE 2b — movies + tag features -> movies_and_tags.csv
# =====================================================================

def build_movies_and_tags():
    """
    Joins every movie (movieId, title, genres) with its tag features from
    Stage 2 and saves data/processed/movies_and_tags.csv.
    Movies with no tags get 0 in every tag column. Needs Stage 2 only.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    movies = load_movies()
    movie_tag_features = pd.read_parquet(
        os.path.join(PROCESSED_DIR, 'movie_tag_features.parquet')
    )
    print(f"Loaded movies: {movies.shape}")
    print(f"Loaded movie_tag_features.parquet: {movie_tag_features.shape}")

    movies_and_tags = movies.join(movie_tag_features, on='movieId', how='left')

    tag_cols = movie_tag_features.columns
    movies_and_tags[tag_cols] = movies_and_tags[tag_cols].fillna(0)

    out_path = os.path.join(PROCESSED_DIR, 'movies_and_tags.csv')
    pacsv.write_csv(pa.Table.from_pandas(movies_and_tags, preserve_index=False), out_path)

    print("\n========== MOVIES + TAGS SAVED ==========")
    print(f"File location : {out_path}")
    print(f"Total records : {len(movies_and_tags):,}")
    print(f"Total columns : {len(movies_and_tags.columns)}")
    print(f"Movies with at least one tag: "
          f"{(movies_and_tags['movie_tag_count'] > 0).sum():,}")

    return movies_and_tags


# =====================================================================
# STAGE 3 — build the feature dataset from the Stage 1 / Stage 2 output
# =====================================================================

def _lookup(table, keys):
    """
    Row lookup of `table` (indexed by id) for each id in `keys`, with 0
    for ids that are missing. Unlike a left join, fill_value keeps the
    uint8 / float32 dtypes instead of upcasting everything to float64.
    """
    return table.reindex(keys.to_numpy(), fill_value=0)


def _build_dataset(ratings_loader, movies, movie_tag_features, user_tag_features,
                   target_column='rating', dataset_name='', save_merged=False):
    """
    ratings_loader: zero-argument function returning the ratings table.
    Taking a loader (not the table) lets this function free the full
    ratings table as soon as it is split, instead of the caller keeping it.
    """
    # ==================================================
    # A. SPLIT RATINGS INTO TRAIN / TEST / VALIDATION
    # ==================================================
    ratings = _downcast_ratings(ratings_loader())
    total = len(ratings)

    train_ratings, remaining_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42
    )
    del ratings
    test_ratings, val_ratings = train_test_split(
        remaining_ratings, test_size=0.5, random_state=42
    )
    del remaining_ratings
    gc.collect()

    print("\n========== DATA SPLIT ==========")
    print(f"Training records  :{len(train_ratings):,} ({len(train_ratings)/total*100:.1f}%)")
    print(f"Testing records   :{len(test_ratings):,} ({len(test_ratings)/total*100:.1f}%)")
    print(f"Validation records:{len(val_ratings):,} ({len(val_ratings)/total*100:.1f}%)")

    # ==================================================
    # B. USER AND MOVIE STATISTICS (train only, no leakage)
    # ==================================================
    user_stats = train_ratings.groupby('userId')['rating'].agg(
        user_avg='mean', user_activity='count'
    ).astype(np.float32)

    print("\n========== USER STATISTICS ==========")
    print(f"Number of users: {len(user_stats):,}")
    print(user_stats.head(10))
    print(user_stats.describe())

    movie_stats = train_ratings.groupby('movieId')['rating'].agg(
        movie_avg='mean', movie_popularity='count'
    ).astype(np.float32)

    print("\n========== MOVIE STATISTICS ==========")
    print(f"Number of movies: {len(movie_stats):,}")
    print(movie_stats.head(10))
    print(movie_stats.describe())

    # ==================================================
    # C. TAG FEATURES (already built and saved in Stage 2)
    # ==================================================
    print("\n========== MOVIE TAG FEATURES ==========")
    print(f"Shape: {movie_tag_features.shape}")
    print(movie_tag_features.head(10))

    print("\n========== USER TAG FEATURES ==========")
    print(f"Shape: {user_tag_features.shape}")
    print(user_tag_features.head(10))

    # ==================================================
    # D. GENRE FEATURES (0/1 -> uint8), indexed by movieId
    # ==================================================
    movies = movies.copy()
    movies['genres'] = movies['genres'].fillna('None')
    genres_encoded = movies['genres'].str.get_dummies(sep='|').astype(np.uint8)
    genres_encoded.index = movies['movieId'].to_numpy()
    genres_encoded = genres_encoded[~genres_encoded.index.duplicated()]

    print("\n========== GENRE FEATURES ==========")
    print(f"Shape: {genres_encoded.shape}")
    print(genres_encoded.head(10))

    # ==================================================
    # E. JOIN ALL FEATURES
    # ==================================================
    def combine_features(split_ratings):
        df = split_ratings.join(user_stats, on='userId')
        df = df.join(movie_stats, on='movieId')  # NaN for unseen ids, as before

        blocks = [
            _lookup(user_tag_features, split_ratings['userId']),
            _lookup(genres_encoded, split_ratings['movieId']),
            _lookup(movie_tag_features, split_ratings['movieId']),
        ]
        for b in blocks:
            b.index = split_ratings.index
        return pd.concat([df] + blocks, axis=1)

    train_df = combine_features(train_ratings)
    test_df = combine_features(test_ratings)
    val_df = combine_features(val_ratings)
    del train_ratings, test_ratings, val_ratings
    gc.collect()

    print("\n========== COMBINED DATASETS ==========")
    print(f"Training shape:   {train_df.shape}")
    print(f"Testing shape:    {test_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(train_df.head(10))

    # ==================================================
    # SAVE (each split on its own; no big concat unless asked for)
    # ==================================================
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    suffix = f'_{dataset_name}' if dataset_name else ''

    for name, frame in (('train', train_df), ('test', test_df), ('val', val_df)):
        path = os.path.join(PROCESSED_DIR, f'{name}_dataset{suffix}.parquet')
        frame.to_parquet(path, index=False)
        print(f"Saved: {path}  (rows={len(frame):,}, cols={len(frame.columns)})")

    if save_merged:
        # peak-memory step: only turn on if you really need the single file
        merged = pd.concat([train_df, test_df, val_df]).sort_index()
        merged_path = os.path.join(PROCESSED_DIR, f'merged_dataset{suffix}.parquet')
        merged.to_parquet(merged_path, index=False)
        print(f"Saved: {merged_path}  (rows={len(merged):,})")
        del merged

    # ==================================================
    # F. PREPARE MODEL INPUTS AND TARGETS
    # ==================================================
    drop_cols = ['rating', 'userId', 'movieId']

    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()
    y_val = val_df[target_column].copy()

    X_train = train_df.drop(columns=drop_cols)
    del train_df                      # not returned, so free it now
    X_test = test_df.drop(columns=drop_cols)
    X_val = val_df.drop(columns=drop_cols)
    gc.collect()

    # test_df / val_df are returned because they still hold userId / movieId
    return (X_train, X_test, X_val, y_train, y_test, y_val, test_df, val_df)


def _load_tag_features():
    """Loads the Stage 2 output (movie + user tag features)."""
    movie_tag_features = pd.read_parquet(
        os.path.join(PROCESSED_DIR, 'movie_tag_features.parquet')
    )
    user_tag_features = pd.read_parquet(
        os.path.join(PROCESSED_DIR, 'user_tag_features.parquet')
    )
    print(f"Loaded movie_tag_features.parquet: {movie_tag_features.shape}")
    print(f"Loaded user_tag_features.parquet: {user_tag_features.shape}")
    return movie_tag_features, user_tag_features


def build_dataset_from_processed(threshold=10, save_merged=False):
    """
    STAGE 3 (cleaned). Needs Stage 1 + Stage 2.
    threshold: 10 or 20, picks ratings_cleaned_<n>.parquet.
    """
    movie_tag_features, user_tag_features = _load_tag_features()
    movies = load_movies()

    def loader():
        r = pd.read_parquet(
            os.path.join(PROCESSED_DIR, f'ratings_cleaned_{threshold}.parquet')
        )
        print(f"\nLoaded ratings_cleaned_{threshold}.parquet: {len(r):,} rows")
        return r

    return _build_dataset(
        loader, movies, movie_tag_features, user_tag_features,
        target_column='rating',
        dataset_name=str(threshold),
        save_merged=save_merged,
    )


def build_dataset_uncleaned(save_merged=False):
    """STAGE 3 (uncleaned). Raw ratings, no filtering. Needs Stage 2."""
    movie_tag_features, user_tag_features = _load_tag_features()
    movies = load_movies()

    def loader():
        r = load_ratings_raw()
        print(f"\nLoaded raw ratings (uncleaned): {len(r):,} rows")
        return r

    return _build_dataset(
        loader, movies, movie_tag_features, user_tag_features,
        target_column='rating',
        dataset_name='uncleaned',
        save_merged=save_merged,
    )


# =====================================================================
# MAIN — comment out everything except the stage you're running.
# Run ONE Stage 3 dataset per Python process for the lowest memory use.
# =====================================================================

def main():

    # ---- STAGE 1: clean ratings, save to data/processed/ ----
    #clean_and_save_ratings()

    # ---- STAGE 2: build + save movie/user tag features ----
    #clean_and_save_tags()

    # ---- STAGE 2b: movies + tag features -> movies_and_tags.csv ----
    build_movies_and_tags()                      # needs Stage 2 only

    # ---- STAGE 3: build the feature datasets ----
    # build_dataset_uncleaned()                  # needs Stage 2 only
    # build_dataset_from_processed(threshold=10)   # needs Stage 1 + 2
    # build_dataset_from_processed(threshold=20)   # needs Stage 1 + 2


if __name__ == "__main__":
    main()