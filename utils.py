import os
import time
import json
import threading
from contextlib import contextmanager
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Columns that are ignored during model training
NON_FEATURE_COLS = ['userId', 'movieId', 'rating', 'split']

# Path where the final evaluation results are saved
RESULTS_FILE = os.path.join('results', 'model_evaluation_results.csv')
TUNING_RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')
TUNING_RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')

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

# Calculates all error/classification/ranking metrics from predictions and builds the result row
def _build_result(split_name, y, pred, eval_meta, predict_time, peak_mem_mb, time_digits=2):
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
        'Predict Time (s)': round(predict_time, time_digits),
        'Predict Memory (MB)': round(peak_mem_mb, 2)
    }
    
    print(f"  [{split_name:<4}] RMSE={result['RMSE']} | MAE={result['MAE']} | R2={result['R2']} | "
          f"Acc={result['Classification Accuracy']} | P@20={result['Precision@20']} | "
          f"R@20={result['Recall@20']} | F1@20={result['F1@20']} | "
          f"Pred Time={result['Predict Time (s)']}s | Pred Mem={result['Predict Memory (MB)']}MB")
    
    return result

# Predicts ratings, calculates all error/classification/ranking metrics, and measures
# time and memory with track_resources, the same method for every model (sklearn, numpy
# and PyTorch alike). Pass device (a torch device) only for GPU models.
def evaluate(split_name, model, X, y, eval_meta, device=None):
    with track_resources(device) as res:
        # Ensure predicted ratings stay within the valid 1 to 5 star range
        pred = np.clip(model.predict(X), 1, 5)

    return _build_result(split_name, y, pred, eval_meta,
                         res['time_s'], res['mem_mb'], time_digits=4)


# =====================================================================
# Time / memory measurement (one method for ALL models)
# =====================================================================
# tracemalloc only tracks Python-heap allocations, so it misses memory that
# numpy / scikit-learn / PyTorch allocate natively. track_resources measures
# what the process actually used, the same way for every model:
#   - GPU : peak torch.cuda memory allocated during the block (with CUDA
#           synchronisation so the timing is not just the kernel launch)
#   - CPU : peak process memory (RSS) during the block, sampled every 5 ms by a
#           background thread (needs psutil: pip install psutil)
# Memory is reported as peak MINUS what was already allocated when the block
# started, i.e. the extra memory the measured step needed. Data and models that
# were loaded beforehand are not counted, so create the model inside the block
# if you want its weights included (as ncf.py does for training).
# Limitation (CPU): if the process already holds free memory pages from earlier
# work, new allocations can reuse them and RSS barely moves, so small steps can
# read lower than they truly are. Evaluate splits in the same order for every
# model and run each dataset in its own process to keep runs comparable.

@contextmanager
def track_resources(device=None):
    """
    Context manager returning a dict that is filled in when the block exits:
        with track_resources(device) as res:
            ...work...
        res['time_s'], res['mem_mb']
    """
    stats = {'time_s': 0.0, 'mem_mb': 0.0}

    try:
        import torch
    except ImportError:
        torch = None

    use_cuda = (torch is not None and device is not None
                and torch.device(device).type == 'cuda' and torch.cuda.is_available())

    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        try:
            yield stats
        finally:
            torch.cuda.synchronize(device)
            stats['time_s'] = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device)
            stats['mem_mb'] = max(peak - baseline, 0) / 1e6
        return

    # CPU path: sample the process RSS in a background thread
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is None:
        # No silent fallback: mixing measurement methods would make models incomparable.
        raise ImportError("track_resources needs psutil for CPU memory: pip install psutil")

    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def _sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            stop.wait(0.005)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats['time_s'] = time.perf_counter() - start
        stop.set()
        sampler.join()
        peak[0] = max(peak[0], proc.memory_info().rss)
        stats['mem_mb'] = max(peak[0] - baseline, 0) / 1e6


# Kept so ncf.py keeps working; identical to evaluate(). device is the torch device.
def evaluate_nn(split_name, model, X, y, eval_meta, device=None):
    return evaluate(split_name, model, X, y, eval_meta, device=device)


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


def save_tuning_results(rows, params):
    os.makedirs(os.path.dirname(TUNING_RESULTS_FILE), exist_ok=True)
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
    write_header = not os.path.exists(TUNING_RESULTS_FILE)
    out.to_csv(TUNING_RESULTS_FILE, mode='a', header=write_header, index=False)