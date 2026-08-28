"""
Classification-mode smoke tests on aeon Japanese Vowels multivariate series.

Run from the tests/ directory containing PFGAP.jar and PF_wrapper.py symlinks:
    python3 test_classification_multivariate_japanese_vowels.py
"""
import numpy as np

from classification_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    write_multivariate_dataset,
    inject_missing_values,
    assert_classification_predictions,
)


def load_japanese_vowels_data():
    try:
        from aeon.datasets import load_japanese_vowels
    except ImportError as exc:
        raise ImportError("aeon is required for this test: pip install aeon") from exc
    X_train, y_train = load_japanese_vowels(split="TRAIN")
    X_test, y_test = load_japanese_vowels(split="TEST")
    return X_train, X_test, y_train, y_test


def run_japanese_vowels_clean_dtwarow():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_jv_clean/data")
    out_dir = clean_dir("tests_tmp/classification_jv_clean/out")

    X_train, X_test, y_train, y_test = load_japanese_vowels_data()
    train_file, test_file, train_labels, test_labels = write_multivariate_dataset(
        X_train, X_test, y_train, y_test, data_dir, "japanese_vowels"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        data_dimension=2,
        numeric_data=True,
        forest_mode="classification",
        purity="gini",
        distances=["dtw_i", "dtw_d"],
        array_separator=":",
        entry_separator=",",
        output_directory=str(out_dir),
        model_name="japanese_vowels_model",
        save_model=True,
        return_predictions=True,
        num_trees=5,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "Japanese Vowels clean classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "Japanese Vowels clean validation")


def run_japanese_vowels_missing_dtwarow():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_jv_missing/data")
    out_dir = clean_dir("tests_tmp/classification_jv_missing/out")

    X_train, X_test, y_train, y_test = load_japanese_vowels_data()
    X_train = inject_missing_values(X_train, missing_rate=0.15, seed=100)
    X_test = inject_missing_values(X_test, missing_rate=0.15, seed=101)
    train_file, test_file, train_labels, test_labels = write_multivariate_dataset(
        X_train, X_test, y_train, y_test, data_dir, "japanese_vowels_missing"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        data_dimension=2,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="classification",
        purity="gini",
        distances=["dtwarow_i", "dtwarow_d"],
        array_separator=":",
        entry_separator=",",
        output_directory=str(out_dir),
        model_name="japanese_vowels_missing_model",
        save_model=True,
        return_predictions=True,
        num_trees=5,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "Japanese Vowels missing classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "Japanese Vowels missing validation")


def run_japanese_vowels_missing_proximity_first_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_jv_impute/data")
    out_dir = clean_dir("tests_tmp/classification_jv_impute/out")

    X_train, X_test, y_train, y_test = load_japanese_vowels_data()
    X_train = inject_missing_values(X_train, missing_rate=0.15, seed=110)
    X_test = inject_missing_values(X_test, missing_rate=0.15, seed=111)
    train_file, test_file, train_labels, test_labels = write_multivariate_dataset(
        X_train, X_test, y_train, y_test, data_dir, "japanese_vowels_impute"
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        train_labels=str(train_labels),
        test_labels=str(test_labels),
        data_dimension=2,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="classification",
        purity="gini",
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
        model_name="japanese_vowels_impute_model",
        save_model=True,
        return_predictions=True,
        num_trees=5,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "Japanese Vowels proximity-first imputation classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(y_test), "Japanese Vowels impute validation")


if __name__ == "__main__":
    require_jar()
    run_japanese_vowels_clean_dtwarow()
    run_japanese_vowels_missing_dtwarow()
    run_japanese_vowels_missing_proximity_first_imputation()
    print("PASS: classification Japanese Vowels tests")
