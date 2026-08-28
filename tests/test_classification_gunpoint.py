"""
Classification-mode smoke tests on local GunPoint with embedded labels.

Run from the tests/ directory containing PFGAP.jar, PF_wrapper.py, and Data/ symlink:
    python3 test_classification_gunpoint.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from classification_common import (
    clean_dir,
    require_jar,
    require_pf_wrapper,
    assert_success,
    inject_missing_values,
    assert_classification_predictions,
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


def run_gunpoint_embedded_labels_clean():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_gunpoint_clean/data")
    out_dir = clean_dir("tests_tmp/classification_gunpoint_clean/out")

    train, test = load_local_gunpoint()
    train_file = data_dir / "gunpoint_train.csv"
    test_file = data_dir / "gunpoint_test.csv"
    train.to_csv(train_file, index=False, header=False)
    test.to_csv(test_file, index=False, header=False)

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        exists_testlabels=True,
        numeric_data=True,
        forest_mode="classification",
        purity="gini",
        distances=["dtw"],
        output_directory=str(out_dir),
        model_name="gunpoint_embedded_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "GunPoint embedded-label clean classification train")
    preds = assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(test), "GunPoint clean validation")


def run_gunpoint_embedded_labels_missing_dtwarow():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_gunpoint_missing/data")
    out_dir = clean_dir("tests_tmp/classification_gunpoint_missing/out")

    train, test = load_local_gunpoint()
    train_missing = inject_missing_values(train, missing_rate=0.25, seed=42, label_position="first")
    test_missing = inject_missing_values(test, missing_rate=0.25, seed=43, label_position="first")
    train_file = data_dir / "gunpoint_missing_train.csv"
    test_file = data_dir / "gunpoint_missing_test.csv"
    pd.DataFrame(train_missing).to_csv(train_file, index=False, header=False)
    pd.DataFrame(test_missing).to_csv(test_file, index=False, header=False)

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        exists_testlabels=True,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="classification",
        purity="gini",
        distances=["dtwarow"],
        output_directory=str(out_dir),
        model_name="gunpoint_missing_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "GunPoint embedded-label missing classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(test), "GunPoint missing validation")


def run_gunpoint_embedded_labels_missing_impute_first():
    PF = require_pf_wrapper()
    data_dir = clean_dir("tests_tmp/classification_gunpoint_impute/data")
    out_dir = clean_dir("tests_tmp/classification_gunpoint_impute/out")

    train, test = load_local_gunpoint()
    train_missing = inject_missing_values(train, missing_rate=0.25, seed=52, label_position="first")
    test_missing = inject_missing_values(test, missing_rate=0.25, seed=53, label_position="first")
    train_file = data_dir / "gunpoint_impute_train.csv"
    test_file = data_dir / "gunpoint_impute_test.csv"
    pd.DataFrame(train_missing).to_csv(train_file, index=False, header=False)
    pd.DataFrame(test_missing).to_csv(test_file, index=False, header=False)

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        exists_testlabels=True,
        numeric_data=True,
        has_missing_values=True,
        forest_mode="classification",
        purity="gini",
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
        model_name="gunpoint_impute_model",
        save_model=True,
        return_predictions=True,
        num_trees=7,
        r=3,
        seed=123,
        verbosity=1,
    )
    assert_success(code, "GunPoint missing impute_first classification train")
    assert_classification_predictions(out_dir / "Validation_Predictions.txt", len(test), "GunPoint impute validation")


if __name__ == "__main__":
    require_jar()
    run_gunpoint_embedded_labels_clean()
    run_gunpoint_embedded_labels_missing_dtwarow()
    run_gunpoint_embedded_labels_missing_impute_first()
    print("PASS: classification GunPoint tests")
