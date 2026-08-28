"""
Label-free isolation mode smoke tests on multivariate aeon Daphnet data.

Requires aeon. The Daphnet loader returns one multivariate sequence with shape
(time, channels), so this script windows it into PFGAP multivariate instances.

Run from project root:
    python tests/isolation/test_isolation_daphnet.py
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from isolation_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    read_pf_array,
    inject_missing_values,
    write_multivariate_ts_file,
    make_sliding_windows_multivariate,
)


def load_daphnet_windows():
    try:
        from aeon.datasets import load_daphnet_s06r02e0
    except ImportError as exc:
        raise ImportError("aeon is required for this test: pip install aeon") from exc

    X, y = load_daphnet_s06r02e0()
    # X has shape (time, 9). Convert it to instances of shape (9, window_length).
    windows = make_sliding_windows_multivariate(
        X,
        window_length=256,
        step=512,
        max_windows=60,
    )
    train_windows, test_windows = train_test_split(
        windows,
        test_size=0.35,
        random_state=17,
        shuffle=True,
    )
    return train_windows, test_windows


def write_daphnet_files(X_train, X_test, data_dir, prefix):
    data_dir = Path(data_dir)
    train_file = write_multivariate_ts_file(
        X_train,
        data_dir / f"{prefix}_train.txt",
        entry_separator=",",
        array_separator=":",
    )
    test_file = write_multivariate_ts_file(
        X_test,
        data_dir / f"{prefix}_test.txt",
        entry_separator=",",
        array_separator=":",
    )
    return train_file, test_file


def run_daphnet_isolation_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_daphnet_clean/data")
    out_dir = clean_dir("tests_tmp/isolation_daphnet_clean/out")

    X_train, X_test = load_daphnet_windows()
    train_file, test_file = write_daphnet_files(X_train, X_test, data_dir, "daphnet_clean")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        data_dimension=2,
        numeric_data=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        distances=["dtw_i", "dtw_d"],
        array_separator=":",
        entry_separator=",",
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "daphnet_isolation_model"),
        save_model=True,
        num_trees=5,
        r=3,
        seed=123,
    )
    assert_success(code, "daphnet clean isolation train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)
    assert np.all(np.isfinite(train_scores))
    assert np.all(np.isfinite(test_scores))


def run_daphnet_isolation_missing_dtwarow():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_daphnet_missing/data")
    out_dir = clean_dir("tests_tmp/isolation_daphnet_missing/out")

    X_train, X_test = load_daphnet_windows()
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=300)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=301)
    train_file, test_file = write_daphnet_files(X_train, X_test, data_dir, "daphnet_missing")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        data_dimension=2,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        distances=["dtwarow_i", "dtwarow_d"],
        array_separator=":",
        entry_separator=",",
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "daphnet_isolation_missing_model"),
        save_model=True,
        num_trees=5,
        r=3,
        seed=123,
    )
    assert_success(code, "daphnet missing isolation train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)


def run_daphnet_isolation_missing_with_imputation():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/isolation_daphnet_impute/data")
    out_dir = clean_dir("tests_tmp/isolation_daphnet_impute/out")

    X_train, X_test = load_daphnet_windows()
    X_train = inject_missing_values(X_train, missing_rate=0.10, seed=310)
    X_test = inject_missing_values(X_test, missing_rate=0.10, seed=311)
    train_file, test_file = write_daphnet_files(X_train, X_test, data_dir, "daphnet_impute")

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        data_dimension=2,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="isolation",
        purity="isolation_path_length",
        impute_training_data=True,
        impute_testing_data=True,
        imputation_initialization="proximity_first",
        missing_proximity_distances=["dtwarow_i", "dtwarow_d"],
        distances=["dtw_i", "dtw_d"],
        gap_update="dtw_alignment",
        impute_iterations=2,
        array_separator=":",
        entry_separator=",",
        bootstrap_trees=False,
        return_training_outlier_scores=True,
        return_imputed_training=True,
        return_imputed_testing=True,
        output_directory=str(out_dir),
        model_name=str(out_dir / "daphnet_isolation_impute_model"),
        save_model=True,
        num_trees=5,
        r=3,
        seed=123,
    )
    assert_success(code, "daphnet missing isolation impute train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")
    assert len(train_scores) == len(X_train)
    assert len(test_scores) == len(X_test)


if __name__ == "__main__":
    require_jar()
    run_daphnet_isolation_clean()
    run_daphnet_isolation_missing_dtwarow()
    run_daphnet_isolation_missing_with_imputation()
    print("PASS: isolation Daphnet tests")
