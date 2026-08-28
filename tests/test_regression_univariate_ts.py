"""
Regression-mode smoke tests on synthetic univariate time series.

Run from the tests/ directory containing PFGAP.jar and PF_wrapper.py symlinks:
    python3 test_regression_univariate_ts.py
"""
import numpy as np
from sklearn.model_selection import train_test_split

from regression_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    write_univariate_ts_csv,
    inject_missing_values,
    assert_regression_predictions,
)


def make_univariate_series_regression(n_samples=180, length=80, seed=123):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length)
    X = np.zeros((n_samples, length))
    y = np.zeros(n_samples)
    for i in range(n_samples):
        amplitude = rng.uniform(0.5, 2.5)
        frequency = rng.uniform(1.0, 4.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend = rng.uniform(-1.0, 1.0)
        noise = rng.normal(0.0, 0.05, size=length)
        X[i] = amplitude * np.sin(2.0 * np.pi * frequency * t + phase) + trend * t + noise
        y[i] = amplitude + 0.5 * frequency + 0.25 * trend
    return X, y


def run_univariate_regression_dtw_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_uni_clean/data")
    out_dir = clean_dir("tests_tmp/regression_uni_clean/out")

    X, y = make_univariate_series_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=20)
    train_file, test_file, train_labels, test_labels = write_univariate_ts_csv(
        X_train, X_test, y_train, y_test, data_dir, "synthetic_uni"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        numeric_data=True,
        forest_mode="regression",
        regressor=True,
        purity="variance",
        distances=["dtw"],
        output_directory=str(out_dir),
        model_name="synthetic_uni_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "univariate DTW regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "univariate DTW validation")


def run_univariate_regression_missing_dtwarow():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_uni_missing/data")
    out_dir = clean_dir("tests_tmp/regression_uni_missing/out")

    X, y = make_univariate_series_regression(seed=456)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=21)
    X_train = inject_missing_values(X_train, missing_rate=0.15, seed=210)
    X_test = inject_missing_values(X_test, missing_rate=0.15, seed=211)
    train_file, test_file, train_labels, test_labels = write_univariate_ts_csv(
        X_train, X_test, y_train, y_test, data_dir, "synthetic_uni_missing"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="regression",
        regressor=True,
        purity="variance",
        distances=["dtwarow"],
        output_directory=str(out_dir),
        model_name="synthetic_uni_missing_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "univariate missing DTW-AROW regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "univariate missing validation")


def run_univariate_regression_missing_impute_first():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_uni_impute/data")
    out_dir = clean_dir("tests_tmp/regression_uni_impute/out")

    X, y = make_univariate_series_regression(seed=789)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=22)
    X_train = inject_missing_values(X_train, missing_rate=0.15, seed=220)
    X_test = inject_missing_values(X_test, missing_rate=0.15, seed=221)
    train_file, test_file, train_labels, test_labels = write_univariate_ts_csv(
        X_train, X_test, y_train, y_test, data_dir, "synthetic_uni_impute"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="regression",
        regressor=True,
        purity="variance",
        impute_training_data=True,
        impute_testing_data=True,
        imputation_initialization="impute_first",
        initial_imputer="linear",
        missing_proximity_distances=["dtwarow"],
        distances=["dtw"],
        gap_update="dtw_alignment",
        impute_iterations=2,
        return_imputed_training=True,
        return_imputed_testing=True,
        output_directory=str(out_dir),
        model_name="synthetic_uni_impute_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "univariate imputed regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "univariate imputed validation")


if __name__ == "__main__":
    require_jar()
    run_univariate_regression_dtw_clean()
    run_univariate_regression_missing_dtwarow()
    run_univariate_regression_missing_impute_first()
    print("PASS: regression univariate time series tests")
