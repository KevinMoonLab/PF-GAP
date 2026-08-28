"""
Label-free isolation mode smoke tests on univariate time series using local GunPoint.

This removes the first column labels and verifies that isolation mode treats rows
as feature-only data.

Run from project root:
    python tests/isolation/test_isolation_gunpoint.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from isolation_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    read_pf_array,
    write_features_only_csv,
    inject_missing_values,
)


def load_local_gunpoint():
    candidates = [
        (Path("Data/GunPoint_TRAIN.csv"), Path("Data/GunPoint_TEST.csv"), "\t"),
        (Path("Data/GunPoint_TRAIN.tsv"), Path("Data/GunPoint_TEST.tsv"), "\t"),
        (Path("Data/GunPoint_TRAIN.txt"), Path("Data/GunPoint_TEST.txt"), "\t"),
    ]
    for train_path, test_path, sep in candidates:
        if train_path.exists() and test_path.exists():
            train = pd.read_csv(train_path, sep=sep, header=None)
            test = pd.read_csv(test_path, sep=sep, header=None)
            return train, test
    raise FileNotFoundError("Could not find local GunPoint train/test files under Data/.")


def run_gunpoint_isolation_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_gunpoint_clean/data")
    out_dir = clean_dir("tests_tmp/isolation_gunpoint_clean/out")

    gun_train, gun_test = load_local_gunpoint()
    X_train = gun_train.iloc[:, 1:]
    X_test = gun_test.iloc[:, 1:]
    train_file, test_file = write_features_only_csv(X_train, X_test, data_dir, "gunpoint_features_only")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        numeric_data=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        distances=["dtw"],
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "gunpoint_isolation_model"),
        save_model=True,
        num_trees=7,
        r=3,
        seed=123,
    )
    assert_success(code, "gunpoint clean isolation train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)
    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(test_scores))


def run_gunpoint_isolation_missing_dtwarow():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_gunpoint_missing/data")
    out_dir = clean_dir("tests_tmp/isolation_gunpoint_missing/out")

    gun_train, gun_test = load_local_gunpoint()
    X_train = inject_missing_values(gun_train.iloc[:, 1:].to_numpy(), missing_rate=0.20, seed=200)
    X_test = inject_missing_values(gun_test.iloc[:, 1:].to_numpy(), missing_rate=0.20, seed=201)
    train_file, test_file = write_features_only_csv(X_train, X_test, data_dir, "gunpoint_missing_features_only")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        distances=["dtwarow"],
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "gunpoint_isolation_missing_model"),
        save_model=True,
        num_trees=7,
        r=3,
        seed=123,
    )
    assert_success(code, "gunpoint missing isolation train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)


def run_gunpoint_isolation_missing_with_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_gunpoint_impute/data")
    out_dir = clean_dir("tests_tmp/isolation_gunpoint_impute/out")

    gun_train, gun_test = load_local_gunpoint()
    X_train = inject_missing_values(gun_train.iloc[:, 1:].to_numpy(), missing_rate=0.20, seed=210)
    X_test = inject_missing_values(gun_test.iloc[:, 1:].to_numpy(), missing_rate=0.20, seed=211)
    train_file, test_file = write_features_only_csv(X_train, X_test, data_dir, "gunpoint_impute_features_only")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        impute_training_data=True,
        impute_testing_data=True,
        imputation_initialization="impute_first",
        initial_imputer="linear",
        missing_proximity_distances=["dtwarow"],
        distances=["dtw"],
        gap_update="dtw_alignment",
        impute_iterations=2,
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        return_imputed_training=True,
        return_imputed_testing=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "gunpoint_isolation_impute_model"),
        save_model=True,
        num_trees=7,
        r=3,
        seed=123,
    )
    assert_success(code, "gunpoint missing isolation impute train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)


if __name__ == "__main__":
    require_jar()
    run_gunpoint_isolation_clean()
    run_gunpoint_isolation_missing_dtwarow()
    run_gunpoint_isolation_missing_with_imputation()
    print("PASS: isolation GunPoint tests")
