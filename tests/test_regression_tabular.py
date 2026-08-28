"""
Regression-mode smoke tests on ordinary tabular data.

Run from the tests/ directory containing PFGAP.jar and PF_wrapper.py symlinks:
    python3 test_regression_tabular.py
"""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split

from regression_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    write_features_and_labels_csv,
    inject_missing_values,
    assert_regression_predictions,
)


def run_diabetes_regression_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_tabular_clean/data")
    out_dir = clean_dir("tests_tmp/regression_tabular_clean/out")

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=12)
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "diabetes"
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
        regressor_aggregation="mean",
        distances=["euclidean"],
        output_directory=str(out_dir),
        model_name="diabetes_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "diabetes regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "diabetes validation")

    code = PF.predict(
        model_name=str(out_dir / "diabetes_regression_model"),
        testfile=str(test_file),
        test_labels=str(test_labels),
        output_directory=str(out_dir),
        return_predictions=True,
        forest_mode="regression",
    )
    assert_success(code, "diabetes regression eval")
    assert_regression_predictions(out_dir / "Predictions_saved.txt", len(y_test), "diabetes saved eval")


def run_high_dimensional_regression_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_tabular_highdim/data")
    out_dir = clean_dir("tests_tmp/regression_tabular_highdim/out")

    X, y = make_regression(
        n_samples=350,
        n_features=60,
        n_informative=15,
        noise=15.0,
        random_state=321,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=14)
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "highdim_regression"
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
        distances=["euclidean"],
        output_directory=str(out_dir),
        model_name="highdim_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "high-dimensional regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "highdim validation")


def run_diabetes_missing_nan_euclidean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_tabular_missing/data")
    out_dir = clean_dir("tests_tmp/regression_tabular_missing/out")

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=15)
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=150)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=151)
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "diabetes_missing"
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
        distances=["nan_euclidean"],
        output_directory=str(out_dir),
        model_name="diabetes_missing_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "diabetes missing regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "diabetes missing validation")


def run_diabetes_missing_proximity_first_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/regression_tabular_impute/data")
    out_dir = clean_dir("tests_tmp/regression_tabular_impute/out")

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=16)
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=160)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=161)
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "diabetes_impute"
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
        imputation_initialization="proximity_first",
        missing_proximity_distances=["nan_euclidean"],
        distances=["euclidean"],
        gap_update="standard",
        impute_iterations=2,
        return_imputed_training=True,
        return_imputed_testing=True,
        output_directory=str(out_dir),
        model_name="diabetes_impute_regression_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "diabetes imputation regression train")
    assert_regression_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "diabetes impute validation")


if __name__ == "__main__":
    require_jar()
    run_diabetes_regression_clean()
    run_high_dimensional_regression_clean()
    run_diabetes_missing_nan_euclidean()
    run_diabetes_missing_proximity_first_imputation()
    print("PASS: regression tabular tests")
