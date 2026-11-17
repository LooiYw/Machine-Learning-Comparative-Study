import os
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Set to True if want plot windows to pop up (figures are always saved)
SHOW_PLOTS = False
OUTPUT_DIR = "images"
VALID_MIN, VALID_MAX = 0.5, 5.0 # Imdb no accept ppl rate 0 stars
Z_THRESHOLD = 3 

if not SHOW_PLOTS:
    matplotlib.use("Agg")

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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def finish_plot(filename):
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    print(f"Saved plot: {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def describe_series(series, name):
    stats = pd.Series({
        "count": series.count(),
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
    })
    print(f"\n{name}")
    print(f"  count  : {int(stats['count']):,}")
    for key in ["min", "max", "mean", "median", "std"]:
        print(f"  {key:<7}: {stats[key]:,.4f}")
    return stats


# ----------------------------------------------------------------------
# A. Dataset Overview
# ----------------------------------------------------------------------
def dataset_overview(ratings, movies, tags):
    section("A. DATASET OVERVIEW")

    print(f"Total unique users              : {ratings['userId'].nunique():,}")
    print(f"Total unique movies (ratings)   : {ratings['movieId'].nunique():,}")
    print(f"Total unique movies (movies.csv): {movies['movieId'].nunique():,}")
    print(f"Total ratings                   : {len(ratings):,}")
    print(f"Total unique tags               : {tags['tag'].nunique():,}")

    print("\nDataset dimensions (rows x columns)")
    for name, df in [("Ratings", ratings), ("Movies", movies), ("Tags", tags)]:
        rows, cols = df.shape
        print(f"  {name:<8}: {rows:,} x {cols}")


# ----------------------------------------------------------------------
# B. Data Distribution
# ----------------------------------------------------------------------
def data_distribution(ratings):
    section("B. DATA DISTRIBUTION")

    ratings_per_user = ratings.groupby("userId").size()
    ratings_per_movie = ratings.groupby("movieId").size()

    # Histogram: rating values (0.5 - 5.0), one bin per rating value
    bins = np.arange(VALID_MIN - 0.25, VALID_MAX + 0.75, 0.5)
    plt.figure(figsize=(9, 5))
    plt.hist(ratings["rating"].dropna(), bins=bins, edgecolor="black")
    plt.xticks(np.arange(VALID_MIN, VALID_MAX + 0.5, 0.5))
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    finish_plot("hist_rating_distribution.png")

    # Histogram: ratings per user (log y-axis because of heavy skew)
    plt.figure(figsize=(9, 5))
    plt.hist(ratings_per_user, bins=100, edgecolor="black")
    plt.yscale("log")
    plt.title("Distribution of Ratings per User")
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Number of users (log scale)")
    finish_plot("hist_ratings_per_user.png")

    # Histogram: ratings per movie (log y-axis because of heavy skew)
    plt.figure(figsize=(9, 5))
    plt.hist(ratings_per_movie, bins=100, edgecolor="black")
    plt.yscale("log")
    plt.title("Distribution of Ratings per Movie")
    plt.xlabel("Number of ratings per movie")
    plt.ylabel("Number of movies (log scale)")
    finish_plot("hist_ratings_per_movie.png")

    # Descriptive statistics
    print("\nDescriptive statistics")
    stats = pd.DataFrame({
        "Rating value": describe_series(ratings["rating"], "Rating values"),
        "Ratings per user": describe_series(ratings_per_user, "Ratings per user"),
        "Ratings per movie": describe_series(ratings_per_movie, "Ratings per movie"),
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats.round(4).to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))

    return ratings_per_user, ratings_per_movie


# ----------------------------------------------------------------------
# C. Outlier Detection
# ----------------------------------------------------------------------
def zscore_outliers(counts, label, id_name):
    std = counts.std()
    if std == 0 or np.isnan(std):
        print(f"\n{label}: standard deviation is 0, z-score not applicable.")
        return pd.DataFrame()

    z = (counts - counts.mean()) / std
    outliers = pd.DataFrame({id_name: counts.index, "rating_count": counts.values,
                             "z_score": z.values})
    outliers = outliers[outliers["z_score"] > Z_THRESHOLD] \
        .sort_values("z_score", ascending=False)

    print(f"\n{label}: {len(outliers):,} of {len(counts):,} "
          f"({len(outliers) / len(counts) * 100:.2f}%) have z-score > {Z_THRESHOLD}")
    print("Top 10:")
    print(outliers.head(10).to_string(index=False))
    return outliers


def outlier_detection(ratings_per_user, ratings_per_movie):
    section("C. OUTLIER DETECTION")

    # Boxplot: ratings per user
    plt.figure(figsize=(9, 5))
    plt.boxplot(ratings_per_user, vert=False, showfliers=True)
    plt.xscale("log")
    plt.title("Boxplot of Ratings per User")
    plt.xlabel("Number of ratings (log scale)")
    plt.yticks([])
    finish_plot("boxplot_ratings_per_user.png")

    # Boxplot: ratings per movie
    plt.figure(figsize=(9, 5))
    plt.boxplot(ratings_per_movie, vert=False, showfliers=True)
    plt.xscale("log")
    plt.title("Boxplot of Ratings per Movie")
    plt.xlabel("Number of ratings (log scale)")
    plt.yticks([])
    finish_plot("boxplot_ratings_per_movie.png")

    # Z-score analysis for unusually high rating counts
    user_outliers = zscore_outliers(ratings_per_user, "Users", "userId")
    movie_outliers = zscore_outliers(ratings_per_movie, "Movies", "movieId")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_outliers.to_csv(os.path.join(OUTPUT_DIR, "user_outliers_zscore.csv"), index=False)
    movie_outliers.to_csv(os.path.join(OUTPUT_DIR, "movie_outliers_zscore.csv"), index=False)


# ----------------------------------------------------------------------
# D. Data Quality
# ----------------------------------------------------------------------
def data_quality(ratings):
    section("D. DATA QUALITY")

    print("Missing values")
    print(pd.DataFrame({"Missing Count": ratings.isnull().sum()}))

    print("\nDuplicate records")
    full_dups = ratings.duplicated().sum()
    pair_dups = ratings.duplicated(subset=["userId", "movieId"]).sum()
    print(f"Fully duplicated rows            : {full_dups:,}")
    print(f"Duplicate (userId, movieId) pairs: {pair_dups:,}")

    print(f"\nInvalid rating values (outside {VALID_MIN}-{VALID_MAX})")
    invalid = ratings[(ratings["rating"] < VALID_MIN) | (ratings["rating"] > VALID_MAX)]
    print(f"Invalid rating count: {len(invalid):,}")
    if len(invalid) > 0:
        print(invalid.head())

    # Ratings should also sit on the 0.5 step grid
    off_grid = ratings[(ratings["rating"] * 2) % 1 != 0]
    print(f"Ratings not on a 0.5 step: {len(off_grid):,}")

    print("\nData types")
    print(ratings.dtypes)


# ----------------------------------------------------------------------
# Original inspection function
# ----------------------------------------------------------------------
def inspect_dataset(dataframe, file_name):
    start_time = time.time()
    print(file_name, " Dataset Dimensions")
    rows, columns = dataframe.shape
    print("=" * 35)
    print("Number of rows: ", rows)
    print("Number of columns: ", columns)
    print("=" * 35)
    print("\nData Type")
    print(dataframe.dtypes)
    print("=" * 35)
    print("\nMissing Values")
    missing_values = pd.DataFrame({"Missing Count": dataframe.isnull().sum()})
    print(missing_values)
    print("=" * 35)
    print("\nFirst 3 Records:")
    print(dataframe.head(3))
    print("=" * 35)
    end_time = time.time() - start_time
    print("Total Inspection Time for", file_name, f": {end_time:.2f} seconds")
    
    
import os
import pandas as pd


# DISCLAIMER: This take from preprocessing to test and modify
def apply_data_cleaning(ratings, thresholds=(10, 20)):
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

    # # --------------------------------------------------
    # # 1. DETECT SUSPICIOUS USERS
    # # Combines rapid rating velocity and zero rating variance
    # # --------------------------------------------------
    # before = len(ratings)

    # max_rating_interval = 1.5       # Seconds
    # min_rapid_percentage = 0.80   # At least 80%
    # min_rapid_intervals = 3       # Minimum number of rapid intervals

    # # Sort ratings chronologically for each user
    # ratings = ratings.sort_values(
    #     ['userId', 'timestamp']
    # ).copy()

    # # Calculate time difference between consecutive ratings
    # ratings['time_difference'] = (
    #     ratings.groupby('userId')['timestamp'].diff()
    # )

    # # Calculate rating interval statistics for each user
    # user_intervals = ratings.groupby('userId').agg(
    #     total_intervals=('time_difference', 'count'),
    #     rapid_intervals=(
    #         'time_difference',
    #         lambda x: ((x > 0) & (x <= max_rating_interval)).sum()
    #     ),
    #     rating_std=('rating', 'std')
    # )

    # # Calculate percentage of rapid consecutive ratings
    # user_intervals['rapid_percentage'] = (
    #     user_intervals['rapid_intervals'] /
    #     user_intervals['total_intervals']
    # )

    # # Identify users satisfying ALL suspicious behaviour conditions
    # suspicious_users = user_intervals[
    #     (user_intervals['rapid_intervals'] >= min_rapid_intervals) &
    #     (user_intervals['rapid_percentage'] >= min_rapid_percentage) &
    #     (user_intervals['rating_std'].fillna(0) == 0)
    # ].index

    # print("\n[1] Suspicious Users (Rapid Ratings + Constant Ratings)")
    # print(f"  Maximum rating interval : {max_rating_interval} second(s)")
    # print(f"  Minimum rapid percentage: {min_rapid_percentage * 100:.0f}%")
    # print(f"  Minimum rapid intervals : {min_rapid_intervals}")
    # print(f"  Zero rating variance    : Required")
    # print(f"  Users identified        : {len(suspicious_users):,}")

    # # Remove suspicious users
    # ratings = ratings[
    #     ~ratings['userId'].isin(suspicious_users)
    # ].copy()

    # ratings = ratings.drop(columns=['time_difference'])

    # log_cleaning("Suspicious Users", before, len(ratings))

    # --------------------------------------------------
    # 2. REMOVE UNPOPULAR MOVIES (one pass per threshold)
    # Removes movies with fewer than N ratings for N in thresholds
    # (default: <10 and <20), saved as ratings_10.csv / ratings_20.csv
    # --------------------------------------------------
    os.makedirs('data', exist_ok=True)

    before = len(ratings)

    movie_counts = ratings['movieId'].value_counts()
    results = {}

    for n in thresholds:
        unpopular_movies = movie_counts[movie_counts < n].index

        print(f"\nUnpopular Movies (< {n} ratings)")
        print(f"  Movies identified: {len(unpopular_movies):,}")

        cleaned = ratings[
            ratings['movieId'].isin(
                movie_counts[movie_counts >= n].index
            )
        ].copy()

        log_cleaning(f"Unpopular Movies (< {n} ratings)", before, len(cleaned))

        # ----------------------------------------------
        # 3. SAVE CLEANED DATASET FOR THIS THRESHOLD
        # ----------------------------------------------
        output_path = f'data/cleaned_ratings_{n}.csv'
        cleaned.to_csv(output_path, index=False)

        print(f"\n[+] Cleaned dataset (< {n} removed) saved successfully!")
        print(f"    File location : {output_path}")
        print(f"    Total records : {len(cleaned):,}")
        print(f"    Total columns : {len(cleaned.columns)}")

        results[n] = cleaned

    # --------------------------------------------------
    # 4. FINAL SUMMARY
    # --------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL DATA CLEANING RESULTS")
    print("=" * 65)

    print(f"Original records : {original_count:,}")

    for n, cleaned in results.items():
        final_count = len(cleaned)
        total_removed = original_count - final_count
        retention_rate = (
            final_count / original_count * 100 if original_count > 0 else 0
        )

        print(f"\nThreshold: < {n} ratings removed")
        print(f"  Final records  : {final_count:,}")
        print(f"  Total removed  : {total_removed:,}")
        print(f"  Retention rate : {retention_rate:.2f}%")

    print("\nDetailed Cleaning Breakdown:")
    print(pd.DataFrame(cleaning_summary).to_string(index=False))

    print("=" * 65)

    # Returns {10: DataFrame, 20: DataFrame}
    return results




def main():
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()

    # inspect_dataset(ratings, "Ratings")

    # dataset_overview(ratings, movies, tags)                       # A
    # ratings_per_user, ratings_per_movie = data_distribution(ratings)  # B
    # outlier_detection(ratings_per_user, ratings_per_movie)        # C
    # data_quality(ratings)                                         # D
    apply_data_cleaning(ratings)


if __name__ == "__main__":
    main()