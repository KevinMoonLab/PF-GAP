"""
Label-free isolation mode smoke tests on ordinary tabular data.

Run from project root:
    python tests/isolation/test_isolation_tabular.py
"""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from isolation_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    read_pf_array,
    write_features_only_csv,
    inject_missing_values,
)


def run_tabular_isolation_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_tabular_clean/data")
    out_dir = clean_dir("tests_tmp/isolation_tabular_clean/out")

    X, _ = load_breast_cancer(return_X_y=True)
    X_train, X_test = train_test_split(X, test_size=0.35, random_state=13)
    train_file, test_file = write_features_only_csv(X_train, X_test, data_dir, "breast_cancer_features_only")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        numeric_data=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        distances=["euclidean"],
        bootstrap_trees=False,
        isolation_num_branches=2,
        isolation_min_leaf_size=1,
        return_training_outlier_scores=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "isolation_tabular_model"),
        save_model=True,
        num_trees=11,
        r=5,
        seed=123,
    )
    assert_success(code, "tabular clean isolation train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)
    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(test_scores))

    code = PF.predict(
        model_name=str(out_dir / "isolation_tabular_model"),
        testfile=str(test_file),
        output_directory=str(out_dir),
        forest_mode="isolation",
    )
    assert_success(code, "tabular clean isolation eval")

    saved_scores = read_pf_array(out_dir / "IsolationScores_saved.txt")
    assert len(saved_scores) == len(X_test)
    assert np.all(np.isfinite(saved_scores))


def run_tabular_isolation_missing_no_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_tabular_missing/data")
    out_dir = clean_dir("tests_tmp/isolation_tabular_missing/out")

    X, _ = load_breast_cancer(return_X_y=True)
    X_train, X_test = train_test_split(X, test_size=0.35, random_state=14)
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=140)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=141)
    train_file, test_file = write_features_only_csv(X_train, X_test, data_dir, "breast_cancer_missing_features_only")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        distances=["nan_euclidean"],
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "isolation_tabular_missing_model"),
        save_model=True,
        num_trees=7,
        r=3,
        seed=123,
    )
    assert_success(code, "tabular missing isolation train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)
    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(test_scores))


def run_tabular_isolation_missing_with_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_tabular_impute/data")
    out_dir = clean_dir("tests_tmp/isolation_tabular_impute/out")

    X, _ = load_breast_cancer(return_X_y=True)
    X_train, X_test = train_test_split(X, test_size=0.35, random_state=15)
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=150)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=151)
    train_file, test_file = write_features_only_csv(X_train, X_test, data_dir, "breast_cancer_impute_features_only")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        numeric_data=True,
        has_missing_values=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        impute_training_data=True,
        impute_testing_data=True,
        imputation_initialization="proximity_first",
        missing_proximity_distances=["nan_euclidean"],
        distances=["euclidean"],
        gap_update="standard",
        impute_iterations=2,
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        return_imputed_training=True,
        return_imputed_testing=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "isolation_tabular_impute_model"),
        save_model=True,
        num_trees=7,
        r=3,
        seed=123,
    )
    assert_success(code, "tabular missing isolation impute train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)
    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(test_scores))


if __name__ == "__main__":
    require_jar()
    run_tabular_isolation_clean()
    run_tabular_isolation_missing_no_imputation()
    run_tabular_isolation_missing_with_imputation()
    print("PASS: isolation tabular tests")
