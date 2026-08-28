"""
Obvious synthetic outlier test for distance-based isolation mode.

Run from the directory containing:
    PFGAP.jar
    PF_wrapper.py
    isolation_common.py

Example:
    python3 test_isolation_obvious_outlier.py
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
)


def make_obvious_outlier_data(
    n_normal=200,
    n_features=20,
    seed=123,
):
    """
    Create one compact normal cloud and a few very obvious outliers.

    Normal data:
        X_normal ~ N(0, 1)

    Outliers:
        fixed points far from the origin in several directions.

    Returns
    -------
    X_train : np.ndarray
        Feature-only training data.
    outlier_indices : list[int]
        Row indices of the injected outliers.
    """
    rng = np.random.default_rng(seed)

    X_normal = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_normal, n_features),
    )

    outliers = np.zeros((3, n_features))

    # Outlier 1: all coordinates very large and positive.
    outliers[0, :] = 12.0

    # Outlier 2: all coordinates very large and negative.
    outliers[1, :] = -12.0

    # Outlier 3: sparse but extreme coordinate pattern.
    outliers[2, 0] = 25.0
    outliers[2, 1] = -25.0
    outliers[2, 2] = 20.0
    outliers[2, 3] = -20.0

    X_train = np.vstack([X_normal, outliers])

    outlier_indices = list(range(n_normal, n_normal + len(outliers)))

    return X_train, outlier_indices


def write_features_only(X, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(path, index=False, header=False)
    return path


def run_obvious_outlier_test():
    require_jar()
    PF = require_pf_wrapper()

    data_dir = clean_dir("tests_tmp/isolation_obvious_outlier/data")
    out_dir = clean_dir("tests_tmp/isolation_obvious_outlier/out")

    X_train, true_outlier_indices = make_obvious_outlier_data(
        n_normal=200,
        n_features=20,
        seed=123,
    )

    # Use the same data as train and test so we can inspect both
    # outlier_scores.txt and IsolationScores.txt.
    train_file = write_features_only(
        X_train,
        data_dir / "obvious_outlier_train.csv",
    )

    test_file = write_features_only(
        X_train,
        data_dir / "obvious_outlier_test.csv",
    )

    code = PF.train(
        train_file=str(train_file),
        test_file=str(test_file),
        numeric_data=True,
        forest_mode="isolation",
        purity="isolation_min_child",
        distances=["euclidean"],
        bootstrap_trees=False,
        isolation_num_branches=2,
        isolation_min_leaf_size=1,
        return_training_outlier_scores=True,
        output_directory=str(out_dir),
        model_name="obvious_outlier_isolation_model",
        save_model=True,
        num_trees=101,
        r=50,
        seed=123,
        verbosity=1,
    )

    assert_success(code, "obvious outlier isolation train")

    train_scores = read_pf_array(out_dir / "outlier_scores.txt")
    test_scores = read_pf_array(out_dir / "IsolationScores.txt")

    print("\nTrue outlier row indices:")
    print(true_outlier_indices)

    print("\nTraining score summary:")
    print("min:", np.min(train_scores))
    print("mean:", np.mean(train_scores))
    print("max:", np.max(train_scores))

    print("\nTest score summary:")
    print("min:", np.min(test_scores))
    print("mean:", np.mean(test_scores))
    print("max:", np.max(test_scores))

    # Highest scores should be most anomalous.
    top_k = 10
    top_train = np.argsort(train_scores)[-top_k:][::-1]
    top_test = np.argsort(test_scores)[-top_k:][::-1]

    print(f"\nTop {top_k} training indices by outlier score:")
    print(top_train)
    print(train_scores[top_train])

    print(f"\nTop {top_k} test indices by isolation score:")
    print(top_test)
    print(test_scores[top_test])

    true_outlier_set = set(true_outlier_indices)
    top_train_set = set(top_train.tolist())
    top_test_set = set(top_test.tolist())

    missing_from_train_top = true_outlier_set - top_train_set
    missing_from_test_top = true_outlier_set - top_test_set

    if missing_from_train_top:
        raise AssertionError(
            "Some injected outliers were not in the top training scores: "
            + str(sorted(missing_from_train_top))
        )

    if missing_from_test_top:
        raise AssertionError(
            "Some injected outliers were not in the top test scores: "
            + str(sorted(missing_from_test_top))
        )

    print("\nPASS: all injected outliers appeared in the top scores.")

    # Saved-model eval check.
    code = PF.predict(
        model_name=str(out_dir / "obvious_outlier_isolation_model"),
        testfile=str(test_file),
        output_directory=str(out_dir),
        forest_mode="isolation",
        numeric_data=True,
    )

    assert_success(code, "obvious outlier isolation eval")

    saved_scores = read_pf_array(out_dir / "IsolationScores_saved.txt")

    top_saved = np.argsort(saved_scores)[-top_k:][::-1]
    top_saved_set = set(top_saved.tolist())

    print(f"\nTop {top_k} saved-model test indices by isolation score:")
    print(top_saved)
    print(saved_scores[top_saved])

    missing_from_saved_top = true_outlier_set - top_saved_set

    if missing_from_saved_top:
        raise AssertionError(
            "Some injected outliers were not in the top saved-model scores: "
            + str(sorted(missing_from_saved_top))
        )

    print("\nPASS: saved-model isolation scores also identify injected outliers.")


if __name__ == "__main__":
    run_obvious_outlier_test()
