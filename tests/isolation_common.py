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
        raise FileNotFoundError("PFGAP.jar not found. Run this script from the project root or copy PFGAP.jar here.")


def require_pf_wrapper():
    try:
        import PF_wrapper as PF
        return PF
    except ImportError as exc:
        raise ImportError("Could not import PF_wrapper. Run this script from the folder containing PF_wrapper.py or add it to PYTHONPATH.") from exc


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
    assert_nonempty_file(path)
    text = path.read_text(encoding="utf-8").strip()
    text = text.replace("{", "[").replace("}", "]")
    return np.array(ast.literal_eval(text))


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


def write_features_only_csv(X_train, X_test, data_dir, prefix):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / f"{prefix}_train.csv"
    test_file = data_dir / f"{prefix}_test.csv"
    pd.DataFrame(X_train).to_csv(train_file, index=False, header=False)
    pd.DataFrame(X_test).to_csv(test_file, index=False, header=False)
    return train_file, test_file


def write_multivariate_ts_file(X, filename, entry_separator=",", array_separator=":"):
    """
    Write a collection of multivariate time series as PFGAP 2D rows.
    X is expected to have shape (n_instances, n_dims, series_length), or to be
    iterable where each instance has shape (n_dims, series_length).
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        for instance in X:
            dims = []
            for dim_values in instance:
                dims.append(entry_separator.join(str(v) for v in dim_values))
            f.write(array_separator.join(dims) + "\n")
    return filename


def make_sliding_windows_multivariate(X, window_length=256, step=512, max_windows=None):
    """
    Convert a single multivariate sequence with shape (time, dims) into
    PFGAP-style instances with shape (n_windows, dims, window_length).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (time, dims), got {X.shape}")

    windows = []
    for start in range(0, X.shape[0] - window_length + 1, step):
        stop = start + window_length
        # PFGAP convention: instance[dim][time]
        windows.append(X[start:stop, :].T)
        if max_windows is not None and len(windows) >= max_windows:
            break

    if not windows:
        raise ValueError("No windows were generated. Reduce window_length or step.")
    return np.asarray(windows)
