# Regression with PFGAP

## Overview
PFGAP supports regression tasks using a modified version of the Proximity Forest algorithm. While originally designed for classification, the model can be adapted for regression by setting the `regressor=True` argument in the `train` method of `PF_wrapper.py`.

## Label Format
For regression, labels must be floating point numbers (represented as `double` in Java). These labels can be included in the training file or passed separately via the `train_labels` argument.

## Training Setup
To train a regression model, use the `train` method in `PF_wrapper.py` with the following key arguments:

- `regressor=True`: Enables regression mode.
- `regressor_aggregation`: Specifies how predictions are aggregated at leaf nodes. Options:
  - `"mean"` (default)
  - `"median"`
- `train_file`: Required training data file.
- `train_labels`: Optional if labels are embedded in `train_file`.
- `test_file` / `test_labels`: Optional validation set for prediction.
- `output_directory`: Location for output files.
- `save_model=True`: Saves the trained model as `[model_name].ser` in `output_directory`.

## Prediction Output
If a test set is provided, predictions will be saved to `Validation_Predictions.txt` in the specified `output_directory`.

## Algorithm Details
In classification mode, Proximity Forest creates a multi-way split at each node based on the number of classes. For regression, this is modified to a binary tree structure:

- Each non-leaf node has exactly two child nodes.
- Splits are determined based on proximity and purity.
- A node becomes a leaf when its purity exceeds the threshold (`purity_threshold`).

## Training Options
All standard training options from classification apply, including:

- `num_trees`: Number of trees (default: 11)
- `r`: Number of candidate splits per node (default: 5)
- `on_tree`: Whether each tree uses its own random distance (default: True)
- `max_depth`: Maximum tree depth (default: 0 for unlimited)
- `distances`: List of distance functions to use
- `parallel_trees`: Parallel training of trees
- `parallel_prox`: Parallel computation of proximities
- `purity`: Purity measure (`"variance"` or `"mad"` recommended for regression)
- `purity_threshold`: Purity threshold to mark a node as leaf (default: 1e-6)
- `memory`: Java heap allocation (default: '1g')

## Compatibility
Regression mode supports:
- Univariate and multivariate time series
- Variable-length sequences
- Missing values (with imputation)
- Numeric and non-numeric data (controlled via `numeric_data`)

Refer to `Imputation.md` for details on handling missing values.

