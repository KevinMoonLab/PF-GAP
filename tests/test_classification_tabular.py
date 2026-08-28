"""
Classification-mode smoke tests on ordinary tabular data.

Run from the tests/ directory containing PFGAP.jar and PF_wrapper.py symlinks:
    python3 test_classification_tabular.py
"""
import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split

from classification_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    write_features_and_labels_csv,
    inject_missing_values,
    assert_classification_predictions,
)


def run_digits_classification_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_tabular_clean/data")
    out_dir = clean_dir("tests_tmp/classification_tabular_clean/out")

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=7, stratify=y
    )
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "digits"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        numeric_data=True,
        forest_mode="classification",
        purity="gini",
        distances=["euclidean"],
        output_directory=str(out_dir),
        model_name="digits_classification_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "digits classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "digits validation")

    code = PF.predict(
        model_name=str(out_dir / "digits_classification_model"),
        testfile=str(test_file),
        test_labels=str(test_labels),
        output_directory=str(out_dir),
        return_predictions=True,
        forest_mode="classification",
    )
    assert_success(code, "digits classification eval")
    assert_classification_predictions(out_dir / "Predictions_saved.txt", len(y_test), "digits saved eval")


def run_breast_cancer_classification_clean_entropy():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_tabular_entropy/data")
    out_dir = clean_dir("tests_tmp/classification_tabular_entropy/out")

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=8, stratify=y
    )
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "breast_cancer"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        numeric_data=True,
        forest_mode="classification",
        purity="entropy",
        distances=["euclidean"],
        output_directory=str(out_dir),
        model_name="breast_cancer_entropy_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "breast cancer entropy classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "breast cancer validation")


def run_digits_missing_nan_euclidean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_tabular_missing/data")
    out_dir = clean_dir("tests_tmp/classification_tabular_missing/out")

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=9, stratify=y
    )
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=90)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=91)
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "digits_missing"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="classification",
        purity="gini",
        distances=["nan_euclidean"],
        output_directory=str(out_dir),
        model_name="digits_missing_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "digits missing nan_euclidean classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "digits missing validation")


def run_digits_missing_proximity_first_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_tabular_impute/data")
    out_dir = clean_dir("tests_tmp/classification_tabular_impute/out")

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=10, stratify=y
    )
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=100)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=101)
    train_file, test_file, train_labels, test_labels = write_features_and_labels_csv(
        X_train, X_test, y_train, y_test, data_dir, "digits_impute"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="classification",
        purity="gini",
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
        model_name="digits_impute_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "digits proximity-first imputation classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "digits impute validation")


if __name__ == "__main__":
    require_jar()
    run_digits_classification_clean()
    run_breast_cancer_classification_clean_entropy()
    run_digits_missing_nan_euclidean()
    run_digits_missing_proximity_first_imputation()
    print("PASS: classification tabular tests")
