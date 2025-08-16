# PFGAP vs KNN Testing Pipeline

This pipeline compares the performance of PFGAP (Proximity Forest with Geometric Analysis) against K-Nearest Neighbors (KNN) classification on 3D spherical data.

## Overview

The pipeline performs three main types of tests:
1. **Accuracy Test**: Compares classification performance between PFGAP and KNN across all k values (1, 4, 8, 12, 16, 20)
2. **Outlier Detection Test**: Generates outlier scores using both PFGAP proximity matrices and KNN distance-based methods
3. **Imputation Test**: (Pending - requires Ben's wrapper)

## Features

- **Parallel Processing**: Uses joblib for parallel execution across all CPU cores
- **Smart Resuming**: Automatically skips completed tests to avoid re-computation
- **Comprehensive Metrics**: F1 score, accuracy, precision, recall, confusion matrices
- **Multiple K Values**: Tests both KNN and PFGAP with k = 1, 4, 8, 12, 16, 20
- **Dual Outlier Detection**: Compares PFGAP proximity-based vs KNN distance-based outlier detection
- **Error Handling**: Continues processing other tests if one fails
- **Frequent Saving**: Saves results incrementally to prevent data loss

## Data

- Uses 3D spherical data generated from `Experiment4functions.py`
- Tests across 10 different random seeds: [61, 737, 821, 161, 346, 78, 2, 67, 102, 982]
- 70/30 train/test split with stratification

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure PFGAP Java JAR files are available in the Application directory

## Usage

Run the pipeline:
```bash
python test_pipeline.py
```

## Output Structure

Results are saved in the `results/` directory with the following structure:

```
results/
├── metrics/                           # Classification metrics
│   ├── {seed}_knn_k{k}.json         # KNN results for each k value
│   ├── {seed}_pfgap_all_k_results.json # PFGAP results for all k values
│   └── {seed}_pfgap_results.json    # PFGAP results (k=5, backward compatibility)
├── proximity_matrices/                # Proximity matrices
│   └── {seed}_proximity_matrix.npy
├── outlier_scores/                   # Outlier detection scores
│   ├── {seed}_pfgap_outlier_scores.npy # PFGAP outlier scores
│   └── {seed}_comprehensive_outlier_scores.json # Both PFGAP and KNN outlier scores
├── {seed}_comprehensive_results.json # Complete results for each seed
└── pipeline_summary.json             # Overall pipeline summary
```

## File Naming Convention

- `{seed}_{test_type}_{timestamp}.json` for JSON results
- `{seed}_{test_type}.npy` for numpy arrays
- `{seed}_comprehensive_results.json` for complete seed results
- `pipeline_summary.json` for overall summary

## Configuration

Key parameters can be modified in the script:
- `K_VALUES`: List of k values for KNN testing
- `TRAIN_TEST_SPLIT`: Ratio for train/test split (default: 0.7)
- `RESULTS_DIR`: Directory for saving results

## Error Handling

The pipeline is designed to be robust:
- Individual test failures don't stop the entire pipeline
- Results are saved incrementally
- Error logs are maintained for each seed
- Temporary files are cleaned up automatically

## Monitoring Progress

The pipeline provides detailed progress information:
- Shows which seeds are being processed
- Indicates when tests are skipped (already completed)
- Reports errors for individual tests
- Generates a final summary report

## Resuming Interrupted Runs

If the pipeline is interrupted, simply run it again:
- Already completed tests will be skipped
- Only remaining tests will be executed
- Results are automatically merged

## Performance

- Parallel execution across all CPU cores
- Memory-efficient processing (one seed at a time)
- Incremental saving prevents data loss
- Smart caching of completed results
