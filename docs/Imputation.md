#Imputation in PFGAP
====================

##Overview
--------
PFGAP supports imputation of missing values in both training and testing datasets. This is enabled via the PF_wrapper.py interface using specific keyword arguments.

##Enabling Imputation
-------------------
To inform the program to impute values, use the following arguments:

- `impute_training_data=True` and/or `impute_testing_data=True` in the `train()` method.
- `impute_testing_data=True` in the `predict()` method.

These flags tell PFGAP to log missing entries and apply imputation. This is especially important when using custom distances that treat certain strings as empty values.

##Initial Imputation
------------------
Before proximity-based updates, an initial imputation method is applied. This is controlled by the `initial_imputer` argument:

- `"mean"` (default): Mean imputation for numeric data.
- `"median"`: Median imputation for numeric data.
- `"linear"`: Time series linear interpolation.
- `"mode"`: Mode imputation for categorical data.

##Proximity-Based Updates
-----------------------
After initial imputation, PFGAP trains a model and constructs proximity matrices. These are used to update previously missing values using proximity-weighted averages or modes.

This process is repeated for a number of iterations specified by:

- `impute_iterations` (default: 5)

##Returning Imputed Data and Proximities
--------------------------------------
You can optionally return the final imputed datasets and proximity matrices:

- `return_imputed_training=True` to save imputed training data
- `return_imputed_testing=True` to save imputed testing data
- `return_proximities=True` to save the final proximity matrix

##Output Location
---------------
All output files are saved in the directory specified by:

- `output_directory` (no trailing "/")

Imputed data files are stored in the subdirectory `data/` using the same filenames as the original `train_file` and `test_file`.

##Notes
-----
- Imputation is compatible with both numeric and non-numeric data.
- Use `numeric_data=True` (default) for numeric datasets.
- For categorical, string, date, or boolean data, set `numeric_data=False`.

