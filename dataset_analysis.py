import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set to True if you want plot windows to pop up (figures are always saved)
SHOW_PLOTS = False
RAW_DIR = "data/raw"
OUTPUT_DIR = "images"
RESULTS_FILE = os.path.join("results", "model_evaluation_results.csv")
VALID_MIN, VALID_MAX = 1, 5     # MovieLens 1M ratings are whole stars, 1-5
Z_THRESHOLD = 3

if not SHOW_PLOTS:
    matplotlib.use("Agg")


# ----------------------------------------------------------------------
# Load dataset (MovieLens 1M: '::'-separated .dat files, no header)
#   ratings.dat : UserID::MovieID::Rating::Timestamp
#   movies.dat  : MovieID::Title::Genres            (latin-1 encoded)
#   users.dat   : UserID::Gender::Age::Occupation::Zip-code
# ----------------------------------------------------------------------
def _read_dat(filename, names):
    start = time.time()
    df = pd.read_csv(
        os.path.join(RAW_DIR, filename),
        sep="::", engine="python", header=None,
        names=names, encoding="latin-1",
    )
    print(f"Loaded {filename}: {df.shape} in {time.time() - start:.2f}s")
    return df


def load_ratings_raw():
    return _read_dat("ratings.dat", ["userId", "movieId", "rating", "timestamp"])


def load_movies():
    return _read_dat("movies.dat", ["movieId", "title", "genres"])


def load_users():
    return _read_dat("users.dat", ["userId", "gender", "age", "occupation", "zip"])


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
def dataset_overview(ratings, movies, users):
    section("A. DATASET OVERVIEW")

    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    print(f"Total unique users (ratings)    : {n_users:,}")
    print(f"Total unique users (users.dat)  : {users['userId'].nunique():,}")
    print(f"Total unique movies (ratings)   : {n_movies:,}")
    print(f"Total unique movies (movies.dat): {movies['movieId'].nunique():,}")
    print(f"Total ratings                   : {len(ratings):,}")
    print(f"Matrix sparsity                 : "
          f"{(1 - len(ratings) / (n_users * n_movies)) * 100:.2f}%")

    print("\nDataset dimensions (rows x columns)")
    for name, df in [("Ratings", ratings), ("Movies", movies), ("Users", users)]:
        rows, cols = df.shape
        print(f"  {name:<8}: {rows:,} x {cols}")


# ----------------------------------------------------------------------
# B. Data Distribution
# ----------------------------------------------------------------------
def data_distribution(ratings):
    section("B. DATA DISTRIBUTION")

    ratings_per_user = ratings.groupby("userId").size()
    ratings_per_movie = ratings.groupby("movieId").size()

    # Histogram: rating values (1-5), one bin per star
    bins = np.arange(VALID_MIN - 0.5, VALID_MAX + 1.5, 1)
    plt.figure(figsize=(9, 5))
    plt.hist(ratings["rating"].dropna(), bins=bins, edgecolor="black")
    plt.xticks(np.arange(VALID_MIN, VALID_MAX + 1, 1))
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

    for counts, title, fname in [
        (ratings_per_user, "Boxplot of Ratings per User", "boxplot_ratings_per_user.png"),
        (ratings_per_movie, "Boxplot of Ratings per Movie", "boxplot_ratings_per_movie.png"),
    ]:
        plt.figure(figsize=(9, 5))
        plt.boxplot(counts, vert=False, showfliers=True)
        plt.xscale("log")
        plt.title(title)
        plt.xlabel("Number of ratings (log scale)")
        plt.yticks([])
        finish_plot(fname)

    user_outliers = zscore_outliers(ratings_per_user, "Users", "userId")
    movie_outliers = zscore_outliers(ratings_per_movie, "Movies", "movieId")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_outliers.to_csv(os.path.join(OUTPUT_DIR, "user_outliers_zscore.csv"), index=False)
    movie_outliers.to_csv(os.path.join(OUTPUT_DIR, "movie_outliers_zscore.csv"), index=False)


# ----------------------------------------------------------------------
# D. Data Quality
# ----------------------------------------------------------------------
def data_quality(ratings, movies, users, ratings_per_user, ratings_per_movie):
    section("D. DATA QUALITY")
    tables = {"Ratings": ratings, "Movies": movies, "Users": users}

    # 1. Missing values: all columns, all tables
    print("1. Missing values (all columns)")
    for name, df in tables.items():
        print(f"\n{name}")
        print(pd.DataFrame({"Missing Count": df.isnull().sum()}))

    # 2. Full-row duplicates
    print("\n2. Full-row duplicates")
    for name, df in tables.items():
        print(f"  {name:<8}: {df.duplicated().sum():,}")

    # 3. Duplicate user-movie pairs
    print("\n3. Duplicate user-movie pairs")
    pair_dups = ratings.duplicated(subset=["userId", "movieId"]).sum()
    print(f"  Duplicate (userId, movieId) pairs: {pair_dups:,}")

    # 4. Invalid rating values (1-5 range)
    print(f"\n4. Invalid rating values (outside {VALID_MIN}-{VALID_MAX})")
    invalid = ratings[(ratings["rating"] < VALID_MIN) | (ratings["rating"] > VALID_MAX)]
    print(f"  Invalid rating count    : {len(invalid):,}")
    if len(invalid) > 0:
        print(invalid.head())
    non_integer = ratings[ratings["rating"] % 1 != 0]
    print(f"  Non-whole-star ratings  : {len(non_integer):,}")
    print(f"  Distinct rating values  : {sorted(ratings['rating'].unique().tolist())}")

    # 5. User interaction counts
    print("\n5. User interaction counts")
    print(f"  Users                  : {len(ratings_per_user):,}")
    print(f"  Min / median / max     : {ratings_per_user.min():,} / "
          f"{int(ratings_per_user.median()):,} / {ratings_per_user.max():,}")
    for k in (10, 20, 50):
        print(f"  Users with < {k:<3} ratings: {(ratings_per_user < k).sum():,}")

    # 6. Movie interaction counts
    print("\n6. Movie interaction counts")
    print(f"  Rated movies           : {len(ratings_per_movie):,}")
    print(f"  Min / median / max     : {ratings_per_movie.min():,} / "
          f"{int(ratings_per_movie.median()):,} / {ratings_per_movie.max():,}")
    for k in (5, 10, 20):
        print(f"  Movies with < {k:<3} ratings: {(ratings_per_movie < k).sum():,}")
    unrated = len(set(movies['movieId']) - set(ratings['movieId']))
    print(f"  Movies in movies.dat with no ratings: {unrated:,}")

    # 7. Data types
    print("\n7. Data types")
    for name, df in tables.items():
        print(f"\n{name}")
        print(df.dtypes)


# ----------------------------------------------------------------------
# E. Feature Scaling
# ----------------------------------------------------------------------
def feature_scaling(ratings, movies, users, ratings_per_user, ratings_per_movie):
    """
    Scales only the features where it matters:
      - interaction counts (heavily right-skewed): log1p, then standardise
      - age bucket code and release_year (bounded numeric): min-max to [0, 1]
    Not scaled: rating (the target), IDs, and 0/1 one-hot columns
    (gender, occupation, zip region, genres) which are already on 0-1.
    In the modelling pipeline, fit the scalers on the TRAIN split only.
    """
    section("E. FEATURE SCALING")

    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    year = year.fillna(year.median())

    candidates = {
        "user_activity (ratings per user)": ratings_per_user.astype(float),
        "movie_popularity (ratings per movie)": ratings_per_movie.astype(float),
        "age (bucket code)": users["age"].astype(float),
        "release_year": year,
    }

    rows = []
    scaled = {}
    for name, series in candidates.items():
        values = series.to_numpy().reshape(-1, 1)
        skew = float(series.skew())
        if "ratings per" in name:
            method = "log1p + StandardScaler"
            out = StandardScaler().fit_transform(np.log1p(values))
        else:
            method = "MinMaxScaler"
            out = MinMaxScaler().fit_transform(values)
        scaled[name] = out.ravel()
        rows.append({
            "feature": name, "skew": round(skew, 3), "method": method,
            "before_min": series.min(), "before_max": series.max(),
            "after_min": out.min(), "after_max": out.max(),
            "after_mean": out.mean(), "after_std": out.std(),
        })

    summary = pd.DataFrame(rows)
    print(summary.round(4).to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.round(4).to_csv(os.path.join(OUTPUT_DIR, "feature_scaling_summary.csv"), index=False)

    # Before / after plot for the two skewed count features
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, name in enumerate(["user_activity (ratings per user)",
                                "movie_popularity (ratings per movie)"]):
        axes[row, 0].hist(candidates[name], bins=60, edgecolor="black")
        axes[row, 0].set_title(f"{name}: before")
        axes[row, 1].hist(scaled[name], bins=60, edgecolor="black")
        axes[row, 1].set_title(f"{name}: after log1p + standardise")
    finish_plot("feature_scaling_before_after.png")


# ----------------------------------------------------------------------
# F. Model Comparison (reads results/model_evaluation_results.csv)
# ----------------------------------------------------------------------
QUALITY_METRICS = ["RMSE", "MAE", "R2", "Classification Accuracy",
                   "Precision@20", "Recall@20", "F1@20"]


def _load_model_results(split="test"):
    """Loads the results CSV and returns one row per model for the given split.

    save_results() appends on every run, so if a model appears more than once
    only its most recent row is kept.
    """
    if not os.path.exists(RESULTS_FILE):
        print(f"{RESULTS_FILE} not found - run the model scripts first.")
        return None

    df = pd.read_csv(RESULTS_FILE)
    df = df[df["Split"] == split]
    if df.empty:
        print(f"No '{split}' rows found in {RESULTS_FILE}.")
        return None

    df = df.drop_duplicates(subset=["Algorithm", "Dataset"], keep="last")
    df["Model"] = df["Algorithm"].astype(str) + "\n(" + df["Dataset"].astype(str) + ")"
    return df.set_index("Model")


def _label_bars(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=7, xytext=(0, 2 if h >= 0 else -2),
                    textcoords="offset points")


def model_comparison():
    """
    Which split each plotted value comes from:
      - performance metrics (RMSE, MAE, R2, ...) : test
      - training time                            : train
      - predict time                             : test
      - predict memory                           : test
    """
    section("F. MODEL COMPARISON")
    df = _load_model_results("test")          # performance, predict time, predict memory
    train_df = _load_model_results("train")   # training time
    if df is None:
        return

    models = df.index.tolist()
    split = "test"

    # Training time comes from the train rows only. Models that never wrote a train
    # row (e.g. knn.py only saves val/test) get no bar in that panel.
    if train_df is None:
        train_time = pd.Series(np.nan, index=models)
    else:
        train_time = train_df["Training Time (s)"].reindex(models)
    missing = train_time[train_time.isna()].index.tolist()
    if missing:
        print("No 'train' rows for: " + ", ".join(m.replace("\n", " ") for m in missing)
              + " - skipped in the training-time plot.")
    n_models = len(models)

    # --- Plot 1: performance metrics (grouped by metric, one bar per model) ---
    metrics = [m for m in QUALITY_METRICS if m in df.columns]
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(models):
        bars = ax.bar(x + i * width - 0.4 + width / 2, df.loc[model, metrics].astype(float),
                      width, label=model.replace("\n", " "), edgecolor="black")
        _label_bars(ax, bars)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Performance Comparison ({split} split)\n"
                 "Lower is better for RMSE / MAE; higher is better for the rest")
    ax.legend(fontsize=8, ncol=2)
    finish_plot("model_comparison_performance.png")

    # --- Plot 2: cost (training time, predict time, predict memory) ---
    cost_panels = [
        ("Training time (s) - train split", train_time),
        ("Predict time (s) - test split", df["Predict Time (s)"]),
        ("Predict memory (MB) - test split", df["Predict Memory (MB)"]),
    ]

    fig, axes = plt.subplots(1, len(cost_panels), figsize=(5.5 * len(cost_panels), 6))
    for ax, (title, values) in zip(axes, cost_panels):
        bars = ax.bar(models, values.astype(float).to_numpy(), edgecolor="black")
        _label_bars(ax, bars, fmt="{:,.2f}")
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Model Cost Comparison")
    finish_plot("model_comparison_cost.png")

    summary = df[[m for m in QUALITY_METRICS if m in df.columns]].copy()
    summary["Training Time (s)"] = train_time
    summary["Predict Time (s)"] = df["Predict Time (s)"]
    summary["Predict Memory (MB)"] = df["Predict Memory (MB)"]
    print(summary.round(4).to_string())


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
    print(pd.DataFrame({"Missing Count": dataframe.isnull().sum()}))
    print("=" * 35)
    print("\nFirst 3 Records:")
    print(dataframe.head(3))
    print("=" * 35)
    print("Total Inspection Time for", file_name, f": {time.time() - start_time:.2f} seconds")


# ----------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------
def main():
    # ratings = load_ratings_raw()
    # movies = load_movies()
    # users = load_users()

    # dataset_overview(ratings, movies, users)                                    # A
    # ratings_per_user, ratings_per_movie = data_distribution(ratings)            # B
    # outlier_detection(ratings_per_user, ratings_per_movie)                      # C
    # data_quality(ratings, movies, users, ratings_per_user, ratings_per_movie)   # D
    # feature_scaling(ratings, movies, users, ratings_per_user, ratings_per_movie)  # E
    model_comparison()                                                          # F


if __name__ == "__main__":
    main()