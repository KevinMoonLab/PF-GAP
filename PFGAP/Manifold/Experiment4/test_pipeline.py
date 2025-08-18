# The purpose of this file is so we can generate results reliably and easily

#& Tasks to Accomplish
"""
1. We need to first do an accuracy test (f1 score and accuracy) between pfgap and knn
2. An outlier detection test
3. Imputation test (NOTE: this is pending Ben's creating a wrapper for the PFGAP to do the imputation)

*All these tests should be using the seeds we have. 
*We want to store the tests in the results folder for easy lookup later
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Application'))
from proxUtil import getProx, getProxArrays, SymmetrizeProx, getOutlierScores
from Experiment4functions import create_3d_sphere_data, SEEDS

# Import configuration
try:
    from config import *
    print("✓ Loaded configuration from config.py")
except ImportError:
    print("⚠ Using default configuration (config.py not found)")
    # Default configuration
    K_VALUES = [1, 4, 8, 12, 16, 20]
    TRAIN_TEST_SPLIT = 0.7
    PFGAP_NUM_TREES = 11
    PFGAP_R = 5
    N_JOBS = -1
    VERBOSE = 10

# Directory setup
RESULTS_DIR = Path(__file__).parent / RESULTS_DIR
PROXIMITY_DIR = RESULTS_DIR / PROXIMITY_DIR
OUTLIER_DIR = RESULTS_DIR / OUTLIER_DIR
METRICS_DIR = RESULTS_DIR / METRICS_DIR
TEMP_DIR = RESULTS_DIR / TEMP_DIR

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, PROXIMITY_DIR, OUTLIER_DIR, METRICS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

def save_results(data, filename, directory):
    """Save results to file with error handling"""
    try:
        filepath = directory / filename
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            np.save(filepath.with_suffix('.npy'), data)
        elif isinstance(data, dict):
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(filepath.with_suffix('.txt'), 'w') as f:
                f.write(str(data))
        print(f"Saved: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def load_existing_results(seed, test_type):
    """Check if results already exist for a given seed and test type"""
    if test_type.startswith("knn_k"):
        filename = f"{seed}_knn_{test_type}.json"
        return (METRICS_DIR / filename).exists()
    elif test_type == "pfgap":
        # Check for the comprehensive PFGAP results file
        filename = f"{seed}_pfgap_all_k_results.json"
        return (METRICS_DIR / filename).exists()
    elif test_type == "proximity":
        filename = f"{seed}_proximity_matrix.npy"
        return (PROXIMITY_DIR / filename).exists()
    elif test_type == "outlier":
        # Check for the comprehensive outlier results file
        filename = f"{seed}_comprehensive_outlier_scores.json"
        return (OUTLIER_DIR / filename).exists()
    return False

def run_single_test(seed):
    """Run all tests for a single seed"""
    print(f"\n{'='*50}")
    print(f"Processing seed: {seed}")
    print(f"{'='*50}")
    
    results = {
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'knn_results': {},
        'pfgap_results': {},
        'outlier_scores': None,
        'errors': []
    }
    
    try:
        # Generate data
        print(f"Generating 3D sphere data for seed {seed}...")
        data, labels, sphere = create_3d_sphere_data(seed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=1-TRAIN_TEST_SPLIT, 
            random_state=seed, stratify=labels
        )
        
        # Save train/test data
        train_data = np.column_stack([X_train, y_train])
        test_data = np.column_stack([X_test, y_test])
        
        # Save data files for PFGAP
        train_file = RESULTS_DIR / f"{seed}_train_data.tsv"
        test_file = RESULTS_DIR / f"{seed}_test_data.tsv"
        
        np.savetxt(train_file, train_data, delimiter='\t', fmt='%.6f')
        np.savetxt(test_file, test_data, delimiter='\t', fmt='%.6f')
        
        # Test 1: KNN Classification
        print(f"Running KNN tests for seed {seed}...")
        for k in K_VALUES:
            if not load_existing_results(seed, f"knn_k{k}"):
                try:
                    knn = KNeighborsClassifier(n_neighbors=k, metric=sphere.metric.dist)
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
                    
                    knn_results = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred, average='weighted'),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True)
                    }
                    
                    results['knn_results'][f'k{k}'] = knn_results
                    
                    # Save individual KNN results
                    save_results(knn_results, f"{seed}_knn_k{k}.json", METRICS_DIR)
                    
                except Exception as e:
                    error_msg = f"KNN k={k} failed: {e}"
                    results['errors'].append(error_msg)
                    print(f"Error: {error_msg}")
        
        # Test 2: PFGAP Classification
        print(f"Running PFGAP tests for seed {seed}...")
        if not load_existing_results(seed, "pfgap"):
            try:
                # Run PFGAP
                output_dir = str(RESULTS_DIR / f"{seed}_pfgap_output")
                
                # For comparison, we'll compute distances manually and use them in our analysis
                getProx(
                    trainfile=str(train_file),
                    testfile=str(test_file),
                    num_trees=PFGAP_NUM_TREES,
                    r=PFGAP_R,
                    out=output_dir,
                    modelname=f"PF_{seed}",
                    distances=["python"]
                )
                
                # Get proximity matrices
                prox_file = Path(output_dir) / "ForestProximities.txt"
                y_file = Path(output_dir) / "ytrain.txt"
                
                if prox_file.exists() and y_file.exists():
                    prox_array, y_array = getProxArrays(str(prox_file), str(y_file))
                    prox_sym = SymmetrizeProx(prox_array)
                    
                    # Save proximity matrix
                    save_results(prox_sym, f"{seed}_proximity_matrix.npy", PROXIMITY_DIR)
                    
                    # Calculate PFGAP predictions using proximity for all k values
                    # This allows us to compare PFGAP performance across the same k values as KNN
                    pfgap_predictions = {}
                    pfgap_metrics = {}
                    test_prox = prox_sym[:len(X_train), len(X_train):]
                    
                    for k in K_VALUES:
                        y_pred_pfgap = []
                        
                        for i in range(test_prox.shape[1]):
                            # Find k nearest neighbors in training set
                            nearest_indices = np.argsort(test_prox[:, i])[-k:]
                            nearest_labels = y_train[nearest_indices]
                            # Majority vote
                            unique, counts = np.unique(nearest_labels, return_counts=True)
                            y_pred_pfgap.append(unique[np.argmax(counts)])
                        
                        y_pred_pfgap = np.array(y_pred_pfgap)
                        
                        # Calculate metrics for this k value
                        pfgap_metrics[f'k{k}'] = {
                            'accuracy': accuracy_score(y_test, y_pred_pfgap),
                            'f1_score': f1_score(y_test, y_pred_pfgap, average='weighted'),
                            'precision': precision_score(y_test, y_pred_pfgap, average='weighted'),
                            'recall': recall_score(y_test, y_pred_pfgap, average='weighted'),
                            'confusion_matrix': confusion_matrix(y_test, y_pred_pfgap).tolist(),
                            'classification_report': classification_report(y_test, y_pred_pfgap, output_dict=True)
                        }
                        
                        pfgap_predictions[f'k{k}'] = y_pred_pfgap
                    
                    # Store results for the default k value (k=5) for backward compatibility
                    results['pfgap_results'] = pfgap_metrics['k5']
                    
                    # Save PFGAP results for all k values
                    save_results(pfgap_metrics, f"{seed}_pfgap_all_k_results.json", METRICS_DIR)
                    
                else:
                    raise FileNotFoundError("PFGAP output files not found")
                    
            except Exception as e:
                error_msg = f"PFGAP failed: {e}"
                results['errors'].append(error_msg)
                print(f"Error: {error_msg}")
        
        #& Test 3: Outlier Detection
        print(f"Running outlier detection for seed {seed}...")
        if not load_existing_results(seed, "outlier"):
            try:
                outlier_results = {}
                
                # PFGAP outlier detection using proximity matrices
                if 'prox_sym' in locals():
                    print(f"  Computing PFGAP outlier scores...")
                    pfgap_outlier_scores = getOutlierScores(prox_sym, y_train)
                    outlier_results['pfgap'] = pfgap_outlier_scores.tolist()
                    
                    # Save PFGAP outlier scores
                    save_results(pfgap_outlier_scores, f"{seed}_pfgap_outlier_scores.npy", OUTLIER_DIR)
                
                # KNN outlier detection using natural KNN approach
                print(f"  Computing KNN outlier scores...")
                knn_outlier_scores = {}
                
                for k in K_VALUES:
                    try:
                        # For KNN, we'll use the distance-based outlier detection
                        # This is how KNN naturally identifies outliers
                        knn = KNeighborsClassifier(n_neighbors=k, metric=sphere.metric.dist)
                        knn.fit(X_train, y_train)
                        
                        # Get distances to k nearest neighbors for each training point
                        distances, indices = knn.kneighbors(X_train)
                        
                        # Outlier score based on average distance to k nearest neighbors
                        # Higher distances indicate more outlier-like behavior
                        avg_distances = np.mean(distances, axis=1)
                        
                        # Normalize by the median distance (similar to PFGAP approach)
                        median_dist = np.median(avg_distances)
                        mad_dist = np.median(np.abs(avg_distances - median_dist))
                        
                        if mad_dist == 0:
                            mad_dist = 1e-6
                        
                        normalized_scores = np.abs(avg_distances - median_dist) / mad_dist
                        knn_outlier_scores[f'k{k}'] = normalized_scores.tolist()
                        
                    except Exception as e:
                        print(f"    Warning: KNN outlier detection failed for k={k}: {e}")
                        knn_outlier_scores[f'k{k}'] = None
                
                outlier_results['knn'] = knn_outlier_scores
                results['outlier_scores'] = outlier_results
                
                # Save comprehensive outlier results
                save_results(outlier_results, f"{seed}_comprehensive_outlier_scores.json", OUTLIER_DIR)
                    
            except Exception as e:
                error_msg = f"Outlier detection failed: {e}"
                results['errors'].append(error_msg)
                print(f"Error: {error_msg}")
        
        # Save comprehensive results
        save_results(results, f"{seed}_comprehensive_results.json", RESULTS_DIR)
        
        # Clean up temporary files
        try:
            os.remove(train_file)
            os.remove(test_file)
        except:
            pass
            
        print(f"Completed seed {seed} successfully")
        
    except Exception as e:
        error_msg = f"Major error in seed {seed}: {e}"
        results['errors'].append(error_msg)
        print(f"Major Error: {error_msg}")
        save_results(results, f"{seed}_error_results.json", RESULTS_DIR)
    
    return results

def main():
    """Main pipeline execution"""
    print("Starting PFGAP vs KNN Testing Pipeline")
    print(f"Testing {len(SEEDS)} seeds with K values: {K_VALUES}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Check which tests have already been completed
    completed_seeds = []
    for seed in SEEDS:
        # Check if all KNN tests, PFGAP (all k values), proximity matrix, and outlier scores exist
        knn_completed = all(load_existing_results(seed, f"knn_k{k}") for k in K_VALUES)
        pfgap_completed = load_existing_results(seed, "pfgap")  # This now checks for comprehensive results
        proximity_completed = load_existing_results(seed, "proximity")
        outlier_completed = load_existing_results(seed, "outlier")  # This now checks for comprehensive results
        
        if knn_completed and pfgap_completed and proximity_completed and outlier_completed:
            completed_seeds.append(seed)
            print(f"Seed {seed} already completed, skipping...")
    
    remaining_seeds = [seed for seed in SEEDS if seed not in completed_seeds]
    
    if not remaining_seeds:
        print("All tests already completed!")
        return
    
    print(f"Running tests for {len(remaining_seeds)} remaining seeds...")
    
    # Run tests in parallel
    all_results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(run_single_test)(seed) for seed in remaining_seeds
    )
    
    # Generate summary report
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    summary = {
        'total_seeds': len(SEEDS),
        'completed_seeds': len(completed_seeds),
        'newly_completed': len(remaining_seeds),
        'timestamp': datetime.now().isoformat(),
        'k_values_tested': K_VALUES,
        'results_summary': {}
    }
    
    # Aggregate results
    for result in all_results:
        if result and 'seed' in result:
            seed = result['seed']
            
            # Get best KNN accuracy
            knn_best_acc = None
            if result.get('knn_results'):
                knn_accuracies = [v['accuracy'] for v in result['knn_results'].values()]
                knn_best_acc = max(knn_accuracies) if knn_accuracies else None
            
            # Get PFGAP accuracy (using k=5 for comparison)
            pfgap_acc = None
            if result.get('pfgap_results'):
                pfgap_acc = result['pfgap_results'].get('accuracy')
            
            # Check if outlier scores were generated
            outlier_generated = result.get('outlier_scores') is not None
            
            summary['results_summary'][seed] = {
                'knn_best_accuracy': knn_best_acc,
                'pfgap_accuracy': pfgap_acc,
                'outlier_scores_generated': outlier_generated,
                'errors': result.get('errors', [])
            }
    
    # Save summary
    save_results(summary, "pipeline_summary.json", RESULTS_DIR)
    
    print(f"\nPipeline completed! Results saved to: {RESULTS_DIR}")
    print(f"Summary: {len(completed_seeds) + len(remaining_seeds)}/{len(SEEDS)} seeds processed")

if __name__ == "__main__":
    main()

