"""
Regression-mode smoke tests on synthetic multivariate time series.

Run from the tests/ directory containing PFGAP.jar and PF_wrapper.py symlinks:
    python3 test_regression_multivariate_ts.py
"""
import numpy as np
from sklearn.model_selection import train_test_split

from regression_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    write_multivariate_dataset,
    inject_missing_values,
    assert_regression_predictions,
)


def make_multivariate_series_regression(n_samples=120, n_dims=3, length=50, seed=123):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length)
    X = np.zeros((n_samples, n_dims, length))
    y = np.zeros(n_samples)
    for i in range(n_samples):
        amplitude = rng.uniform(0.5, 2.5)
        frequency = rng.uniform(1.0, 3.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend = rng.uniform(-0.5, 0.5)
        for d in range(n_dims):
            dim_scale = 1.0 + 0.25 * d
            noise = rng.normal(0.0, 0.04, size=length)
            X[i, d] = dim_scale * amplitude * np.sin(2.0 * np.pi * frequency * t + phase + 0.3 * d) + trend * t + noise
        y[i] = amplitude + frequency + trend + 0.1 * np.mean(X[i, 0])
    return X, y


def run_multivariate_regression_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_multi_clean/data")
    out_dir = clean_dir("tests_tmp/regression_multi_clean/out")

    X, y = make_multivariate_series_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=30)
    train_file, test_file, train_labels, test_labels = write_multivariate_dataset(
        X_train, X_test, y_train, y_test, data_dir, "synthetic_multi"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        data_dimension=2,
        numeric_data=True,
        forest_mode="regression",
        regressor=True,
        purity="variance",
        distances=["dtw_i", "dtw_d"],
        array_separator=":",
        entry_separator=",",
        output_directory=str(out_dir),
        model_name="synthetic_multi_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=5,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "multivariate regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "multivariate validation")


def run_multivariate_regression_missing_dtwarow():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_multi_missing/data")
    out_dir = clean_dir("tests_tmp/regression_multi_missing/out")

    X, y = make_multivariate_series_regression(seed=456)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=31)
    X_train = inject_missing_values(X_train, missing_rate=0.12, seed=310)
    X_test = inject_missing_values(X_test, missing_rate=0.12, seed=311)
    train_file, test_file, train_labels, test_labels = write_multivariate_dataset(
        X_train, X_test, y_train, y_test, data_dir, "synthetic_multi_missing"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        data_dimension=2,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="regression",
        regressor=True,
        purity="variance",
        distances=["dtwarow_i", "dtwarow_d"],
        array_separator=":",
        entry_separator=",",
        output_directory=str(out_dir),
        model_name="synthetic_multi_missing_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=5,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "multivariate missing regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "multivariate missing validation")


def run_multivariate_regression_missing_proximity_first_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_multi_impute/data")
    out_dir = clean_dir("tests_tmp/regression_multi_impute/out")

    X, y = make_multivariate_series_regression(seed=789)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=32)
    X_train = inject_missing_values(X_train, missing_rate=0.12, seed=320)
    X_test = inject_missing_values(X_test, missing_rate=0.12, seed=321)
    train_file, test_file, train_labels, test_labels = write_multivariate_dataset(
        X_train, X_test, y_train, y_test, data_dir, "synthetic_multi_impute"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        data_dimension=2,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="regression",
        regressor=True,
        purity="variance",
        impute_training_data=True,
        impute_testing_data=True,
        imputation_initialization="proximity_first",
        missing_proximity_distances=["dtwarow_i", "dtwarow_d"],
        distances=["dtw_i", "dtw_d"],
        gap_update="dtw_alignment",
        impute_iterations=2,
        return_imputed_training=True,
        return_imputed_testing=True,
        array_separator=":",
        entry_separator=",",
        output_directory=str(out_dir),
        model_name="synthetic_multi_impute_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=5,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "multivariate imputed regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "multivariate imputed validation")


if __name__ == "__main__":
    require_jar()
    run_multivariate_regression_clean()
    run_multivariate_regression_missing_dtwarow()
    run_multivariate_regression_missing_proximity_first_imputation()
    print("PASS: regression multivariate time series tests")
