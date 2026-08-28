from pathlib import Path
import shutil
import ast
import numpy as np
import pandas as pd


def clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_jar():
    if not Path("PFGAP.jar").exists():
        raise FileNotFoundError("PFGAP.jar not found. Run this script from the tests/ directory containing the PFGAP.jar symlink.")


def require_pf_wrapper():
    try:
        import PF_wrapper as PF
        return PF
    except ImportError as exc:
        raise ImportError("Could not import PF_wrapper. Run this script from the tests/ directory containing PF_wrapper.py or a symlink to it.") from exc


def assert_success(code, context="PF call"):
    if code != 0:
        raise RuntimeError(f"{context} failed with exit code {code}")


def assert_nonempty_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    if path.stat().st_size == 0:
        raise AssertionError(f"Expected output file is empty: {path}")


def read_pf_array(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise AssertionError(f"Expected output file is empty: {path}")
    text = text.replace("{", "[").replace("}", "]")
    return np.array(ast.literal_eval(text), dtype=float)


def inject_missing_values(data, missing_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(data, copy=True)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float)
    flat = arr.reshape(-1)
    n_missing = int(flat.size * missing_rate)
    if n_missing > 0:
        idx = rng.choice(flat.size, size=n_missing, replace=False)
        flat[idx] = np.nan
    return arr


def write_features_and_labels_csv(X_train, X_test, y_train, y_test, data_dir, prefix):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / f"{prefix}_train.csv"
    test_file = data_dir / f"{prefix}_test.csv"
    train_labels = data_dir / f"{prefix}_labels_train.csv"
    test_labels = data_dir / f"{prefix}_labels_test.csv"
    pd.DataFrame(X_train).to_csv(train_file, index=False, header=False)
    pd.DataFrame(X_test).to_csv(test_file, index=False, header=False)
    pd.DataFrame(y_train).to_csv(train_labels, index=False, header=False)
    pd.DataFrame(y_test).to_csv(test_labels, index=False, header=False)
    return train_file, test_file, train_labels, test_labels


def write_univariate_ts_csv(X_train, X_test, y_train, y_test, data_dir, prefix):
    return write_features_and_labels_csv(X_train, X_test, y_train, y_test, data_dir, prefix)


def write_multivariate_ts_file(X, filename, entry_separator=",", array_separator=":"):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        for instance in X:
            dims = []
            for dim_values in instance:
                dims.append(entry_separator.join(str(v) for v in dim_values))
            f.write(array_separator.join(dims) + "\n")
    return filename


def write_multivariate_dataset(X_train, X_test, y_train, y_test, data_dir, prefix):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = write_multivariate_ts_file(X_train, data_dir / f"{prefix}_train.txt")
    test_file = write_multivariate_ts_file(X_test, data_dir / f"{prefix}_test.txt")
    train_labels = data_dir / f"{prefix}_labels_train.csv"
    test_labels = data_dir / f"{prefix}_labels_test.csv"
    pd.DataFrame(y_train).to_csv(train_labels, index=False, header=False)
    pd.DataFrame(y_test).to_csv(test_labels, index=False, header=False)
    return train_file, test_file, train_labels, test_labels


def assert_regression_predictions(path, expected_length, name):
    preds = read_pf_array(path)
    if len(preds) != expected_length:
        raise AssertionError(f"{name}: expected {expected_length} predictions, got {len(preds)}")
    if not np.all(np.isfinite(preds)):
        raise AssertionError(f"{name}: predictions contain non-finite values")
    print(f"{name}: len={len(preds)}, min={np.min(preds):.6g}, mean={np.mean(preds):.6g}, max={np.max(preds):.6g}")
    return preds
