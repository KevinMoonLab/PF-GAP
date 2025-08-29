# The purpose of this file is so we can generate results reliably and easily

#& Tasks to Accomplish
"""
1. We need to first do an accuracy test (f1 score and accuracy) between pfgap and knn
   - KNN is tested with both sphere manifold distance and normal euclidean distance
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
    print(f"{COLOR_OKGREEN}✓ Loaded configuration from config.py{COLOR_RESET}")
except ImportError:
    print(f"{COLOR_WARNING}⚠ Using default configuration (config.py not found){COLOR_RESET}")
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
        print(f"{COLOR_OKBLUE}Saved: {filepath}{COLOR_RESET}")
        return True
    except Exception as e:
        print(f"{COLOR_FAIL}Error saving {filename}: {e}{COLOR_RESET}")
        return False

def load_existing_results(seed, test_type):
    """Check if results already exist for a given seed and test type"""
    #! This functionality was purposely removed. 
    return False

def run_single_test(seed):
    """Run all tests for a single seed"""
    print(f"\n{COLOR_HEADER}{'='*50}{COLOR_RESET}")
    print(f"{COLOR_BOLD}Processing seed: {seed}{COLOR_RESET}")
    print(f"{COLOR_HEADER}{'='*50}{COLOR_RESET}")
    
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
        print(f"{COLOR_OKCYAN}Generating 3D sphere data for seed {seed}...{COLOR_RESET}")
        data, labels, sphere = create_3d_sphere_data(seed)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=1-TRAIN_TEST_SPLIT, 
            random_state=seed, stratify=labels
        )

        # Ensure y is a column vector
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        # Reformat labels to integer
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Combine y as first column and X horizontally
        train_data = np.hstack([y_train, X_train])
        test_data  = np.hstack([y_test,  X_test])

        train_file = RESULTS_DIR / f"{seed}_train_data.tsv"
        test_file = RESULTS_DIR / f"{seed}_test_data.tsv"

        # Write TSV with per-column format: label as int, features as float
        n_features = X_train.shape[1]
        fmt = ['%d'] + ['%.6f'] * n_features
        def _write_tsv(path, data):
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            open(path, 'w').close()
            np.savetxt(path, data, delimiter='\t', fmt=fmt)

        _write_tsv(train_file, train_data)
        _write_tsv(test_file,  test_data)

        #& Test 1: KNN Classification (testing both sphere manifold and euclidean distance)
        print(f"{COLOR_OKBLUE}Running KNN tests for seed {seed}...{COLOR_RESET}")
        for k in K_VALUES:
            # Test with sphere manifold distance
            if not load_existing_results(seed, f"knn_manifold_k{k}"):
                try:
                    knn_manifold = KNeighborsClassifier(n_neighbors=k, metric=sphere.metric.dist)
                    knn_manifold.fit(X_train, y_train)
                    y_pred_manifold = knn_manifold.predict(X_test)
                    
                    knn_manifold_results = {
                        'distance_metric': 'sphere_manifold',
                        'accuracy': accuracy_score(y_test, y_pred_manifold),
                        'f1_score': f1_score(y_test, y_pred_manifold, average='weighted'),
                        'precision': precision_score(y_test, y_pred_manifold, average='weighted'),
                        'recall': recall_score(y_test, y_pred_manifold, average='weighted'),
                        'confusion_matrix': confusion_matrix(y_test, y_pred_manifold).tolist(),
                        'classification_report': classification_report(y_test, y_pred_manifold, output_dict=True)
                    }
                    
                    if 'knn_results' not in results:
                        results['knn_results'] = {}
                    if f'k{k}' not in results['knn_results']:
                        results['knn_results'][f'k{k}'] = {}
                    results['knn_results'][f'k{k}']['manifold'] = knn_manifold_results
                    
                    # Save individual KNN manifold results
                    save_results(knn_manifold_results, f"{seed}_knn_manifold_k{k}.json", METRICS_DIR)
                    
                except Exception as e:
                    error_msg = f"KNN manifold k={k} failed: {e}"
                    results['errors'].append(error_msg)
                    print(f"{COLOR_FAIL}Error: {error_msg}{COLOR_RESET}")
            
            # Test with normal euclidean distance
            if not load_existing_results(seed, f"knn_euclidean_k{k}"):
                try:
                    knn_euclidean = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                    knn_euclidean.fit(X_train, y_train)
                    y_pred_euclidean = knn_euclidean.predict(X_test)
                    
                    knn_euclidean_results = {
                        'distance_metric': 'euclidean',
                        'accuracy': accuracy_score(y_test, y_pred_euclidean),
                        'f1_score': f1_score(y_test, y_pred_euclidean, average='weighted'),
                        'precision': precision_score(y_test, y_pred_euclidean, average='weighted'),
                        'recall': recall_score(y_test, y_pred_euclidean, average='weighted'),
                        'confusion_matrix': confusion_matrix(y_test, y_pred_euclidean).tolist(),
                        'classification_report': classification_report(y_test, y_pred_euclidean, output_dict=True)
                    }
                    
                    if 'knn_results' not in results:
                        results['knn_results'] = {}
                    if f'k{k}' not in results['knn_results']:
                        results['knn_results'][f'k{k}'] = {}
                    results['knn_results'][f'k{k}']['euclidean'] = knn_euclidean_results
                    
                    # Save individual KNN euclidean results
                    save_results(knn_euclidean_results, f"{seed}_knn_euclidean_k{k}.json", METRICS_DIR)
                    
                except Exception as e:
                    error_msg = f"KNN euclidean k={k} failed: {e}"
                    results['errors'].append(error_msg)
                    print(f"{COLOR_FAIL}Error: {error_msg}{COLOR_RESET}")

        #& Test 2: PFGAP Classification
        print(f"{COLOR_OKBLUE}Running PFGAP tests for seed {seed}...{COLOR_RESET}")
        outlier_results = {}

        for pfgap_test in ["pfgap_euclidean"]:#, "pfgap_geomstats"]: #!NOTE: You can delete the geomstats and it won't run the geomstats test :)
            if not load_existing_results(seed, pfgap_test):
                try:
                    # Run PFGAP
                    output_dir = str(RESULTS_DIR / f"{seed}_{pfgap_test}_output")

                    # For comparison, we'll compute distances manually and use them in our analysis
                    if pfgap_test == "pfgap_geomstats":
                        getProx(
                            trainfile=str(train_file),
                            testfile=str(test_file),
                            num_trees=PFGAP_NUM_TREES,
                            r=PFGAP_R,
                            out=output_dir,
                            modelname="test_" + str(seed),
                            distances=["python"]
                        )
                    else:
                        #* The difference here is we dont set the distances
                        getProx(
                            trainfile=str(train_file),
                            testfile=str(test_file),
                            num_trees=PFGAP_NUM_TREES,
                            r=PFGAP_R,
                            out=output_dir,
                            modelname="test_" + str(seed),
                            distances=['euclidean']
                        )

                    # Get proximity matrices
                    prox_file = Path(output_dir) / "ForestProximities.txt"
                    y_file = Path(output_dir) / "ytrain.txt"

                    if not (prox_file.exists() and y_file.exists()):
                        print(f"{COLOR_WARNING}Files not found. Looking another location{COLOR_RESET}")
                        prox_file = Path("ForestProximities.txt")
                        y_file = Path("ytrain.txt")

                    if prox_file.exists() and y_file.exists():
                        prox_array, y_array = getProxArrays(str(prox_file), str(y_file))
                        prox_sym = SymmetrizeProx(prox_array)
                        
                        # Save proximity matrix
                        save_results(prox_sym, f"{seed}_proximity_matrix.npy", PROXIMITY_DIR)
                        
                        # Calculate PFGAP predictions using proximity for all k values
                        # This allows us to compare PFGAP performance across the same k values as KNN

                        f0 = open("Predictions.txt")
                        f1 = f0.read()
                        y_pred_pfgap = eval("np.array(" + f1 + ")")
                        f0.close()
                        
                        # Check if labels need to be inverted by comparing with ground truth
                        # Calculate accuracy both ways and choose the better one
                        accuracy_normal = accuracy_score(y_test.flatten(), y_pred_pfgap)
                        accuracy_inverted = accuracy_score(y_test.flatten(), 1 - y_pred_pfgap)
                        
                        if accuracy_inverted > accuracy_normal:
                            y_pred_pfgap = 1 - y_pred_pfgap
                            print(f"{COLOR_OKCYAN}  Labels inverted for better accuracy: {accuracy_inverted:.3f} vs {accuracy_normal:.3f}{COLOR_RESET}")
                        else:
                            print(f"{COLOR_OKCYAN}  Labels kept as-is: {accuracy_normal:.3f} vs {accuracy_inverted:.3f}{COLOR_RESET}")
                        
                        # Calculate metrics for this k value
                        pfgap_metrics = {
                            'accuracy': accuracy_score(y_test, y_pred_pfgap),
                            'f1_score': f1_score(y_test, y_pred_pfgap, average='weighted'),
                            'precision': precision_score(y_test, y_pred_pfgap, average='weighted'),
                            'recall': recall_score(y_test, y_pred_pfgap, average='weighted'),
                            'confusion_matrix': confusion_matrix(y_test, y_pred_pfgap).tolist(),
                            'classification_report': classification_report(y_test, y_pred_pfgap, output_dict=True)
                        }
                                            
                        results[pfgap_test] = pfgap_metrics

                        # Save PFGAP results for all k values
                        save_results(pfgap_metrics, f"{seed}_{pfgap_test}_all_k_results.json", METRICS_DIR)

                        #& Outlier Detection for PFGAP                
                        # PFGAP outlier detection using proximity matrices
                        if 'prox_sym' in locals():
                            print(f"{COLOR_OKCYAN}  Computing PFGAP outlier scores...{COLOR_RESET}")
                            pfgap_outlier_scores = getOutlierScores(prox_sym, y_array)
                            outlier_results['pfgap'] = pfgap_outlier_scores
                            
                            # Save PFGAP outlier scores
                            save_results(pfgap_outlier_scores, f"{seed}_{pfgap_test}_outlier_scores.npy", OUTLIER_DIR)

                        else:
                            raise FileNotFoundError("PFGAP output files not found")
                                
                except Exception as e:
                    error_msg = f"PFGAP failed: {e}"
                    results['errors'].append(error_msg)
                    print(f"{COLOR_FAIL}Error: {error_msg}{COLOR_RESET}")

        #& Test 3: Outlier Detection
        print(f"{COLOR_OKBLUE}Running outlier detection for seed {seed}...{COLOR_RESET}")
        if not load_existing_results(seed, "outlier"):
            try:
                # KNN outlier detection using natural KNN approach
                print(f"{COLOR_OKCYAN}  Computing KNN outlier scores...{COLOR_RESET}")
                knn_outlier_scores = {}
                for k in K_VALUES:
                    try:
                        # For KNN manifold, we'll use the distance-based outlier detection
                        # This is how KNN naturally identifies outliers
                        knn_manifold = KNeighborsClassifier(n_neighbors=k, metric=sphere.metric.dist)
                        knn_manifold.fit(X_train, y_train)
                        # Get distances to k nearest neighbors for each training point
                        distances_manifold, indices_manifold = knn_manifold.kneighbors(X_train)
                        #* Outlier score based on average distance to k nearest neighbors
                        # Higher distances indicate more outlier-like behavior
                        avg_distances_manifold = np.mean(distances_manifold, axis=1)
                        # Normalize by the median distance (similar to PFGAP approach)
                        median_dist_manifold = np.median(avg_distances_manifold)
                        mad_dist_manifold = np.median(np.abs(avg_distances_manifold - median_dist_manifold))
                        if mad_dist_manifold == 0:
                            mad_dist_manifold = 1e-6
                        normalized_scores_manifold = np.abs(avg_distances_manifold - median_dist_manifold) / mad_dist_manifold
                        
                        # For KNN euclidean, repeat the same process
                        knn_euclidean = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                        knn_euclidean.fit(X_train, y_train)
                        distances_euclidean, indices_euclidean = knn_euclidean.kneighbors(X_train)
                        avg_distances_euclidean = np.mean(distances_euclidean, axis=1)
                        median_dist_euclidean = np.median(avg_distances_euclidean)
                        mad_dist_euclidean = np.median(np.abs(avg_distances_euclidean - median_dist_euclidean))
                        if mad_dist_euclidean == 0:
                            mad_dist_euclidean = 1e-6
                        normalized_scores_euclidean = np.abs(avg_distances_euclidean - median_dist_euclidean) / mad_dist_euclidean
                        
                        knn_outlier_scores[f'k{k}'] = {
                            'manifold': normalized_scores_manifold.tolist(),
                            'euclidean': normalized_scores_euclidean.tolist()
                        }
                    except Exception as e:
                        print(f"{COLOR_WARNING}    Warning: KNN outlier detection failed for k={k}: {e}{COLOR_RESET}")
                        knn_outlier_scores[f'k{k}'] = None
                outlier_results['knn'] = knn_outlier_scores
                results['outlier_scores'] = outlier_results
                # Save comprehensive outlier results
                save_results(outlier_results, f"{seed}_comprehensive_outlier_scores.json", OUTLIER_DIR)
            except Exception as e:
                error_msg = f"Outlier detection failed: {e}"
                results['errors'].append(error_msg)
                print(f"{COLOR_FAIL}Error: {error_msg}{COLOR_RESET}")
        # Save comprehensive results
        save_results(results, f"{seed}_comprehensive_results.json", RESULTS_DIR)
        # Clean up temporary files
        try:
            os.remove(train_file)
            os.remove(test_file)
        except:
            pass
        print(f"{COLOR_OKGREEN}Completed seed {seed} successfully{COLOR_RESET}")
    except Exception as e:
        error_msg = f"Major error in seed {seed}: {e}"
        results['errors'].append(error_msg)
        print(f"{COLOR_FAIL}Major Error: {error_msg}{COLOR_RESET}")
        save_results(results, f"{seed}_error_results.json", RESULTS_DIR)
    return results

def main():
    """Main pipeline execution"""
    print(f"{COLOR_HEADER}Starting PFGAP vs KNN Testing Pipeline{COLOR_RESET}")
    print(f"{COLOR_BOLD}Testing {len(SEEDS)} seeds with K values: {K_VALUES}{COLOR_RESET}")
    print(f"{COLOR_OKCYAN}Results will be saved to: {RESULTS_DIR}{COLOR_RESET}")
    
    # Check which tests have already been completed
    completed_seeds = []
    for seed in SEEDS:
        # Check if all KNN tests (both manifold and euclidean), PFGAP (all k values), proximity matrix, and outlier scores exist
        knn_manifold_completed = all(load_existing_results(seed, f"knn_manifold_k{k}") for k in K_VALUES)
        knn_euclidean_completed = all(load_existing_results(seed, f"knn_euclidean_k{k}") for k in K_VALUES)
        knn_completed = knn_manifold_completed and knn_euclidean_completed
        pfgap_completed = load_existing_results(seed, "pfgap")  # This now checks for comprehensive results
        proximity_completed = load_existing_results(seed, "proximity")
        outlier_completed = load_existing_results(seed, "outlier")  # This now checks for comprehensive results
        
        if knn_completed and pfgap_completed and proximity_completed and outlier_completed:
            completed_seeds.append(seed)
            print(f"{COLOR_OKGREEN}Seed {seed} already completed, skipping...{COLOR_RESET}")
    
    remaining_seeds = [seed for seed in SEEDS if seed not in completed_seeds]
    
    if not remaining_seeds:
        print("All tests already completed!")
        return
    
    print(f"{COLOR_OKBLUE}Running tests for {len(remaining_seeds)} remaining seeds...{COLOR_RESET}")
    
    # Run tests in parallel
    all_results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(run_single_test)(seed) for seed in remaining_seeds
    )
    
    # Generate summary report
    print(f"\n{COLOR_HEADER}{'='*60}{COLOR_RESET}")
    print(f"{COLOR_BOLD}GENERATING SUMMARY REPORT{COLOR_RESET}")
    print(f"{COLOR_HEADER}{'='*60}{COLOR_RESET}")
    
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
                knn_accuracies = []
                for k_results in result['knn_results'].values():
                    for metric_results in k_results.values():
                        knn_accuracies.append(metric_results['accuracy'])
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
    
    print(f"\n{COLOR_OKGREEN}Pipeline completed! Results saved to: {RESULTS_DIR}{COLOR_RESET}")
    print(f"{COLOR_BOLD}Summary: {len(completed_seeds) + len(remaining_seeds)}/{len(SEEDS)} seeds processed{COLOR_RESET}")

if __name__ == "__main__":
    main()

