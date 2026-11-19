import os
import time
import json
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Columns that are ignored during model training
NON_FEATURE_COLS = ['userId', 'movieId', 'rating', 'split']

# Path where the final evaluation results are saved
RESULTS_FILE = os.path.join('results', 'model_evaluation_results.csv')

# Separates features (X) and target ratings (y), and retains metadata for ranking metrics
def get_splits(df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    splits = {}
    meta = {}
    for name in ('train', 'val', 'test'):
        part = df[df['split'] == name]
        splits[name] = (part[feature_cols].to_numpy(dtype=np.float32),
                        part['rating'].to_numpy(dtype=np.float32))
        # Keep user/movie/rating for Top-K ranking calculations
        meta[name] = part[['userId', 'movieId', 'rating']].copy()
    return splits, meta

# Calculates Precision@K, Recall@K, and F1@K for recommendation ranking
def compute_ranking_metrics(eval_meta, y_pred, k=20, threshold=4.0):
    df = eval_meta.copy()
    df['pred'] = y_pred
    precisions, recalls = [], []

    for _, group in df.groupby('userId'):
        relevant = group[group['rating'] >= threshold]
        if len(relevant) == 0:
            continue
        top_k = group.nlargest(k, 'pred')
        hits = (top_k['rating'] >= threshold).sum()
        precisions.append(hits / min(k, len(group)))
        recalls.append(hits / len(relevant))

    p_k = float(np.mean(precisions)) if precisions else 0.0
    r_k = float(np.mean(recalls)) if recalls else 0.0
    
    # F1@K is the harmonic mean of Precision and Recall
    f1_k = (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0
    
    return p_k, r_k, f1_k

# Predicts ratings, calculates all error/classification/ranking metrics, and measures memory
def evaluate(split_name, model, X, y, eval_meta):
    tracemalloc.start()
    start = time.time()
    
    # Ensure predicted ratings stay within the valid 1 to 5 star range
    pred = np.clip(model.predict(X), 1, 5)
    
    predict_time = time.time() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Error Metrics
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    
    # Classification Metric (round predictions to nearest whole star to check for exact match)
    acc = float(accuracy_score(y, np.round(pred)))
    
    # Ranking Metrics
    p20, r20, f1_20 = compute_ranking_metrics(eval_meta, pred, k=20)
    
    # Round all values before saving
    result = {
        'Split': split_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'Classification Accuracy': round(acc, 4),
        'Precision@20': round(p20, 4),
        'Recall@20': round(r20, 4),
        'F1@20': round(f1_20, 4),
        'Predict Time (s)': round(predict_time, 2),
        'Predict Memory (MB)': round(peak_mem / 1e6, 2)
    }
    
    print(f"  [{split_name:<4}] RMSE={result['RMSE']} | MAE={result['MAE']} | R2={result['R2']} | "
          f"Acc={result['Classification Accuracy']} | P@20={result['Precision@20']} | "
          f"R@20={result['Recall@20']} | F1@20={result['F1@20']} | "
          f"Pred Time={result['Predict Time (s)']}s | Pred Mem={result['Predict Memory (MB)']}MB")
    
    return result

# Formats the evaluation metrics and appends them to the results CSV file
def save_results(rows, params):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    out = pd.DataFrame(rows)
    
    # Extract algorithm and dataset names from the model label string
    out['Algorithm'] = out['model'].str.split('_').str[0]
    out['Dataset'] = out['model'].str.split('_', n=1).str[1].str.lower()
    
    # Automatically rename the training time/memory columns coming from your model files
    # This prevents you from having to modify the 4 model files again!
    out.rename(columns={
        'train_time_s': 'Training Time (s)',
        'train_mem_mb': 'Training Memory (MB)'
    }, inplace=True)
    
    # Round the training columns as well
    out['Training Time (s)'] = out['Training Time (s)'].round(2)
    out['Training Memory (MB)'] = out['Training Memory (MB)'].round(2)
    
    # Convert the dictionary of parameters into a JSON string for easy logging
    out['Parameters'] = json.dumps(params, default=str)
    
    # Organize all requested columns in a fixed, readable order
    cols = ['Algorithm', 'Dataset', 'Split', 'RMSE', 'MAE', 'R2', 'Classification Accuracy', 
            'Precision@20', 'Recall@20', 'F1@20', 'Training Time (s)', 'Training Memory (MB)', 
            'Predict Time (s)', 'Predict Memory (MB)', 'Parameters']
    out = out[cols]
    
    # Create a new header only if the file does not exist yet
    write_header = not os.path.exists(RESULTS_FILE)
    out.to_csv(RESULTS_FILE, mode='a', header=write_header, index=False)