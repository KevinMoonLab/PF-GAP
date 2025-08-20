import numpy as np
from pathlib import Path
import json
import pandas as pd

def load_results(results_dir="results"):
    """
    Load results and flatten them into a pandas DataFrame where each row
    represents a test method with flat (non-nested) columns.
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found!")
        return None
    
    # Load individual seed results
    seed_results = {}
    for seed_file in results_dir.glob("*_comprehensive_results.json"):
        seed = seed_file.stem.split('_')[0]
        try:
            with open(seed_file, 'r') as f:
                seed_results[seed] = json.load(f)
            print(f"✓ Loaded results from seed {seed}")
        except Exception as e:
            print(f"⚠ Error loading {seed_file}: {e}")
    
    print(f"✓ Loaded {len(seed_results)} seed results")
    
    # Process and flatten the results into a DataFrame
    flat_data = []
    
    for seed, data in seed_results.items():
        
        # Handle KNN results (nested by k and distance metric)
        if 'knn_results' in data:
            knn_data = data['knn_results']
            for k_val, k_data in knn_data.items():
                # Extract k value as integer (remove 'k' prefix)
                k_num = int(k_val[1:]) if k_val.startswith('k') else k_val
                
                # Handle manifold results
                if 'manifold' in k_data:
                    row = {
                        'seed': seed,
                        'method': 'KNN-Manifold',
                        'k': k_num,
                    }
                    # Add all metrics as flat columns, including confusion matrix
                    manifold_metrics = k_data['manifold']
                    for metric_name, metric_value in _flatten_dict(manifold_metrics, include_complex=True).items():
                        row[metric_name] = metric_value
                    
                    # Add outlier scores if available
                    if 'outlier_scores' in data and 'knn' in data['outlier_scores']:
                        knn_outliers = data['outlier_scores']['knn']
                        if k_val in knn_outliers and 'manifold' in knn_outliers[k_val]:
                            row['outlier_scores'] = knn_outliers[k_val]['manifold']
                    
                    flat_data.append(row)
                
                # Handle euclidean results  
                if 'euclidean' in k_data:
                    row = {
                        'seed': seed,
                        'method': 'KNN-Euclidean', 
                        'k': k_num,
                    }
                    # Add all metrics as flat columns, including confusion matrix
                    euclidean_metrics = k_data['euclidean']
                    for metric_name, metric_value in _flatten_dict(euclidean_metrics, include_complex=True).items():
                        row[metric_name] = metric_value
                    
                    # Add outlier scores if available
                    if 'outlier_scores' in data and 'knn' in data['outlier_scores']:
                        knn_outliers = data['outlier_scores']['knn']
                        if k_val in knn_outliers and 'euclidean' in knn_outliers[k_val]:
                            row['outlier_scores'] = knn_outliers[k_val]['euclidean']
                    
                    flat_data.append(row)
        
        # Handle PFGAP Euclidean results (no k values)
        if 'pfgap_euclidean' in data:
            row = {
                'seed': seed,
                'method': 'PFGAP-Euclidean',
                'k': np.nan,  # No k for PFGAP methods
            }
            # Add all metrics as flat columns, including confusion matrix
            for metric_name, metric_value in _flatten_dict(data['pfgap_euclidean'], include_complex=True).items():
                row[metric_name] = metric_value
            
            # Add PFGAP outlier scores if available
            if 'outlier_scores' in data and 'pfgap' in data['outlier_scores']:
                row['outlier_scores'] = data['outlier_scores']['pfgap']
            
            flat_data.append(row)
        
        # Handle PFGAP Manifold results (future use, no k values)
        if 'pfgap_manifold' in data:
            row = {
                'seed': seed,
                'method': 'PFGAP-Manifold',
                'k': np.nan,  # No k for PFGAP methods
            }
            # Add all metrics as flat columns, including confusion matrix
            for metric_name, metric_value in _flatten_dict(data['pfgap_manifold'], include_complex=True).items():
                row[metric_name] = metric_value
            
            # Add PFGAP outlier scores if available (assuming they'd be under pfgap_manifold in outlier_scores)
            if 'outlier_scores' in data and 'pfgap_manifold' in data['outlier_scores']:
                row['outlier_scores'] = data['outlier_scores']['pfgap_manifold']
            
            flat_data.append(row)
    
    # Create DataFrame from flattened data
    results_df = pd.DataFrame(flat_data)
    
    # Convert seed to integer for better sorting/analysis
    if not results_df.empty:
        results_df['seed'] = results_df['seed'].astype(int)
    
    return results_df.drop(columns= ['classification_report_0_precision', 'classification_report_0_recall', 'classification_report_0_f1-score', 'classification_report_0_support', 'classification_report_1_precision', 'classification_report_1_recall', 'classification_report_1_f1-score', 'classification_report_1_support', 'classification_report_accuracy', 'classification_report_macro avg_precision', 'classification_report_macro avg_recall', 'classification_report_macro avg_f1-score', 'classification_report_macro avg_support', 'classification_report_weighted avg_precision', 'classification_report_weighted avg_recall', 'classification_report_weighted avg_f1-score', 'classification_report_weighted avg_support'])

def _flatten_dict(d, parent_key='', sep='_', include_complex=False):
    """Flatten nested dictionaries into a single level dictionary, 
    with option to include or skip complex nested structures like confusion matrices and classification reports"""
    items = []
    
    # Keys to skip when include_complex=False
    skip_keys = ['classification_report'] if not include_complex else []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # Skip complex nested structures unless requested
        if k in skip_keys:
            continue
            
        if isinstance(v, dict):
            # Only flatten simple dicts when include_complex=False, include all when True
            if include_complex or all(not isinstance(val, (dict, list)) for val in v.values()):
                items.extend(_flatten_dict(v, new_key, sep=sep, include_complex=include_complex).items())
            else:
                # For complex nested dicts, skip them when include_complex=False
                continue
        elif isinstance(v, list):
            # Include lists when include_complex=True (like confusion matrices, outlier scores)
            if include_complex:
                items.append((new_key, v))
            # Skip lists when include_complex=False
        else:
            # Include simple scalar values
            items.append((new_key, v))
            
    return dict(items)
