# Classification in PFGAP
========================

## Overview
--------
PFGAP builds on the original Proximity Forest model, which was designed for univariate time series classification.
The original implementation is available at: https://github.com/fpetitjean/ProximityForest/tree/master

This project extends compatibility to multivariate time series and other data types, including support for missing values.

## Label Formats
-------------
Class labels must currently be represented as integers. Support for string labels is planned for future versions.

Labels can be:
- Embedded in the training/testing file (using the `target_column` argument to specify location)
- Provided separately via `train_labels` and `test_labels` arguments

## Training Setup
--------------
Use the `train` method in PF_wrapper.py to train a classification model.

Required arguments:
- `train_file`: path to the training data file
- `train_labels`: optional if labels are embedded in `train_file`

Optional arguments:
- `test_file`: path to validation/test data
- `test_labels`: optional if labels are embedded in `test_file`

## Prediction Output
-----------------
If a test/validation set is provided, predictions will be saved to:
`[output_directory]/Validation_Predictions.txt`

## Model Saving
------------
By default, the trained model is saved as:
`[output_directory]/PF.ser`

You can customize this using:
- `save_model=True` (default)
- `model_name="PF"` (default)

## Training Options
----------------
- `num_trees` (default: 11): number of trees in the forest
- `r` (default: 5): number of candidate splits per node
- `on_tree` (default: True): whether each tree uses its own random distance
- `max_depth` (default: 0): maximum tree depth (0 means unlimited)
- `distances` (default: None): list of distance functions to use (see distances documentation)
- `parallel_trees` (default: False): parallelize tree training
- `parallel_prox` (default: False): parallelize proximity computation
- `purity` (default: "gini"): method for computing leaf node purity
  - Other options: "entropy", "variance", "mad"
- `purity_threshold` (default: 1e-6): purity threshold to mark a node as a leaf
- `memory` (default: "1g"): Java memory allocation

## Data Compatibility
------------------
Classification supports:
- Univariate and multivariate time series
- Variable-length sequences
- Missing values (with imputation)
- Numeric and non-numeric data (depending on distance functions)

See `data_format.txt` and `imputation.txt` for more details on formatting and preprocessing.

