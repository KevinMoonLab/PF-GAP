# Outlier Score Computation in PFGAP
===================================

## Overview
--------
PFGAP supports outlier score computation for classification tasks only. These scores are intra-class outlier scores, meaning they measure how anomalous a training instance is relative to other instances of the same class.

Outlier scores are not available for regression tasks. However, users may manually analyze training proximities using external techniques such as Local Outlier Factor (LOF) [Breunig et al., 2000].

## Enabling Outlier Score Computation
----------------------------------
To compute outlier scores during training, set the following argument in PF_wrapper.py:

    return_training_outlier_scores=True

This will trigger the computation of intra-class outlier scores for the training set.

## Output Format
-------------
The outlier scores are saved as a `.txt` file in the location specified by the `output_directory` argument. The file contains a Java-style array (including curly brackets), e.g.:

    {0.123, 0.456, 0.789, ...}

## Reading Outlier Scores in Python
--------------------------------
To read the outlier scores into a NumPy array, use the `getArray(filename)` method provided in PF_wrapper.py. This method converts the Java-style array into a Python-compatible format:

    import PF_wrapper
    scores = PF_wrapper.getArray("path/to/outlier_scores.txt")

## Special Case: Zero Denominator Handling
---------------------------------------
Outlier scores are computed using the formula:

    score_i = n / sum_j (P[i][j]^2)

where `n` is the number of training instances, and `P[i][j]` is the proximity between instance `i` and instance `j` of the same class.

In rare cases, the denominator (sum of squared proximities) may be zero. To avoid division by zero, PFGAP replaces the denominator with a small constant:

    if sum == 0.0: sum = 1e-6

This ensures numerical stability during score computation.

## Citation
--------
Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data (pp. 93–104). https://doi.org/10.1145/342009.335388
