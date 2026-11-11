import pandas as pd
import time
#also do pip install pyarrow for parallel programming

#Data Loading
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
    
#See Dataset Characteristics
def inspect_dataset(dataframe, file_name):
    start_time = time.time()
     
    print(file_name, " Dataset Dimensions")
    
    rows,columns = dataframe.shape
    
    print("=" * 20)
    print("Number of rows: ", rows)
    print("Number of columns: ", columns)
    print("=" * 20)

    #Data types
    print("\nData Type")
    print(dataframe.dtypes)
    print("=" * 20)

    #Missing values
    print("\nMissing Values")
    missing_values = pd.DataFrame({
        "Missing Count": dataframe.isnull().sum()
    })
    print(missing_values)
    print("=" * 20)

    # #Duplicate records
    # print(f"Duplicate rows: {dataframe.duplicated().sum():,}")
    # print("=" * 20)
    
    print("\nFirst 3 Records:")
    print(dataframe.head())
    print("=" * 20)

    print("\nLast 3 Records:")
    print(dataframe.tail())
    print("=" * 20)
    
    end_time = time.time() - start_time
    print("Total Inspection Time for", file_name, f": {end_time:.2f} seconds")
    
    
def main():
    # Load datasets
    ratings = load_ratings()
    movies = load_movies()
    links = load_links()
    tags = load_tags()

    #region Inspect datasets
    # inspect_dataset(ratings, "Ratings") #32000204 rows
    # inspect_dataset(movies, "Movies Metadata")
    # inspect_dataset(keywords, "Keywords")
    inspect_dataset(tags, "Tags")
    #endregion

if __name__ == "__main__":
    main()