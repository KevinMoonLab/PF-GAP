import subprocess
import numpy as np
import os
import ast


def _bool(value):
    return "true" if value else "false"


def _list_arg(values):
    if values is None:
        return "[]"
    if isinstance(values, str):
        return values
    return "[" + ",".join(str(v) for v in values) + "]"


def _separator_arg(value):
    if value == "\t":
        return "\\t"
    return value


def _ensure_output_directory(output_directory):
    if output_directory == "":
        return os.getcwd() + "/"

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    return output_directory.rstrip("/") + "/"


def _append_common_distance_arg(msg_list, arg_name, distances):
    msg_list.append(f"-{arg_name}={_list_arg(distances)}")


def _append_if_not_none(msg_list, arg_name, value):
    if value is not None:
        msg_list.append(f"-{arg_name}={value}")
        
def _proximity_type_arg(value):
    if value is None:
        return "PFGAP"

    value = str(value).strip().upper()

    valid = {
        "PFGAP",
        "BREIMAN",
        "DEPTH_WEIGHTED",
    }

    if value not in valid:
        raise ValueError(
            "proximity_type must be one of: "
            + ", ".join(sorted(valid))
            )
            
    return value
    
def _normalized_reader_type(value):
    if value is None:
        return None

    return str(value).strip().upper()


def _validate_lazy_reader_options(
    reader_type,
    file_pattern,
    impute_training_data=False,
    impute_testing_data=False,
    return_imputed_training=False,
    return_imputed_testing=False,
):
    normalized = _normalized_reader_type(reader_type)

    if normalized != "PER_FILE_PARQUET":
        return

    if file_pattern is None or not str(file_pattern).strip():
        raise ValueError(
            "reader_type='PER_FILE_PARQUET' requires file_pattern. "
            "Example: 'series_{num:04d}.parquet'."
        )

    if (
        impute_training_data
        or impute_testing_data
        or return_imputed_training
        or return_imputed_testing
    ):
        raise ValueError(
            "Imputation is not currently supported for lazy "
            "PER_FILE_PARQUET datasets. Disable imputation and "
            "imputed-data output options."
        )
        

def train(
    train_file,
    test_file=None,
    train_labels=None,
    test_labels=None,
    exists_testlabels=False,
    return_predictions=False,
    return_proximities=False,
    proximity_type="PFGAP",
    save_model=True,
    model_name="PF",
    output_directory="",
    repeats=1,
    num_trees=11,
    r=5,
    forest_mode=None, # defaults to being a classifier
    isolation_num_branches=2,
    isolation_min_leaf_size=1,
    regression_num_branches=2,
    bootstrap_trees=True,
    seed=None,
    on_tree=True,
    max_depth=0,
    shuffle=False,
    export=1,
    verbosity=1,
    file_has_header=False,
    target_column="first",
    distances=None,
    memory="1g",
    parallel_trees=False,
    parallel_predict=False,
    parallel_prox=False,
    parallel_split_assignments=False,
    parallel_split_assignment_threshold=128,

    # Missing/imputation controls
    has_missing_values=None,
    impute_training_data=False,
    impute_testing_data=False,
    impute_iterations=5,
    return_imputed_training=False,
    return_imputed_testing=False,
    initial_imputer="mean",
    knn_distances=None,
    DTWImpute=False,
    imputation_initialization="impute_first",
    gap_update=None,
    missing_proximity_distances=None,
    missing_indicators=("", "NA", "NaN", "null", "nan", "NAN"),

    # Data controls
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    array_separator=":",
    reader_type=None,
    id_column=None,
    time_column=None,
    feature_columns=None,
    label_columns=None,
    hdf5_dataset_path="/X",
    hdf5_label_dataset_path="/y",
    file_pattern=None,
    
    # Standardization controls
    standardization="none",
    standardization_scope="per_dimension",
    standardization_variance="population",
    standardization_stats=None,
    save_standardization_stats=False,
    standardization_stats_output=None,

    # Other outputs/model controls
    return_training_outlier_scores=False,
    regressor=False,
    purity="gini",
    purity_threshold=1e-6,
    regressor_aggregation="mean"
):
    if data_dimension not in [1, 2]:
        raise ValueError("Keyword argument 'data_dimension' must be 1 or 2.")

    is2D = data_dimension == 2
    
    if forest_mode is None:
        forest_mode = "regression" if regressor else "classification"
        
    forest_model = forest_mode.lower()
    
    if forest_mode not in {"classification", "regression", "isolation"}:
        raise ValueError("forest_mode must be one of: 'classification', 'regression', or 'isolation'.")
        
    if forest_mode == "regression":
        regressor = True
    elif forest_mode in {"classification", "isolation"}:
        regressor = False
        
    # a user should select a compatible purity, if they forgot.
    if forest_mode == "isolation" and purity == "gini":
        purity = "isolation_path_length"
        
    if forest_mode == "regression" and purity == "gini":
        purity = "variance"

    if has_missing_values is None:
        has_missing_values = (
            impute_training_data
            or impute_testing_data
            or imputation_initialization == "proximity_first"
            or missing_proximity_distances is not None
        )

    if gap_update is None:
        gap_update = "dtw_alignment" if DTWImpute else "standard"

    entry_separator = _separator_arg(entry_separator)
    array_separator = _separator_arg(array_separator)
    output_directory = _ensure_output_directory(output_directory)
    
    if return_imputed_training and not impute_training_data:
        impute_training_data = True

    if return_imputed_testing and not impute_testing_data:
        impute_testing_data = True
        
    _validate_lazy_reader_options(
        reader_type=reader_type,
        file_pattern=file_pattern,
        impute_training_data=impute_training_data,
        impute_testing_data=impute_testing_data,
        return_imputed_training=return_imputed_training,
        return_imputed_testing=return_imputed_testing,
    )
        
    model_name = os.path.basename(os.path.normpath(str(model_name)))

    msgList = ["java", "-Xmx" + memory, "-jar", "PFGAP.jar", "-eval=false"]

    msgList.extend([
        "-train=" + str(train_file),
        "-train_labels=" + str(train_labels),
        "-test=" + str(test_file),
        "-test_labels=" + str(test_labels),
        "-exists_testlabels=" + _bool(exists_testlabels),

        "-repeats=" + str(repeats),
        "-trees=" + str(num_trees),
        "-r=" + str(r),
        "-on_tree=" + _bool(on_tree),
        "-max_depth=" + str(max_depth),
        "-shuffle=" + _bool(shuffle),

        "-export=" + str(export),
        "-verbosity=" + str(verbosity),
        "-csv_has_header=" + _bool(file_has_header),
        "-target_column=" + target_column,

        "-getprox=" + _bool(return_proximities),
        "-proximity_type=" + _proximity_type_arg(proximity_type),
        "-get_predictions=" + _bool(return_predictions),
        "-savemodel=" + _bool(save_model),
        "-modelname=" + model_name,

        "-parallelTrees=" + _bool(parallel_trees),
        "-parallelProx=" + _bool(parallel_prox),
        "-parallelPredict=" + _bool(parallel_predict),
        "-parallelSplit=" + _bool(parallel_split_assignments),
        "-parallelSplitThreshold=" + str(parallel_split_assignment_threshold),

        "-hasMissingValues=" + _bool(has_missing_values),
        "-perform_train_imputation=" + _bool(impute_training_data),
        "-perform_test_imputation=" + _bool(impute_testing_data),
        "-numImputes=" + str(impute_iterations),
        "-impute_train=" + _bool(return_imputed_training),
        "-impute_test=" + _bool(return_imputed_testing),

        "-is2D=" + _bool(is2D),
        "-isNumeric=" + _bool(numeric_data),

        "-get_training_outlier_scores=" + _bool(return_training_outlier_scores),
        "-initial_imputer=" + initial_imputer,

        "-isRegression=" + _bool(regressor),
        "-purity_measure=" + purity,
        "-purity_threshold=" + str(purity_threshold),
        "-voting=" + regressor_aggregation,
        
        "-forest_mode=" + forest_mode,
        "-isolation_num_branches=" + str(isolation_num_branches),
        "-isolation_min_leaf_size=" + str(isolation_min_leaf_size),
        "-regression_num_branches=" + str(regression_num_branches),
        "-bootstrap_trees=" + _bool(bootstrap_trees),

        "-DTWImpute=" + _bool(DTWImpute),
        "-imputation_initialization=" + imputation_initialization,
        "-gap_update=" + gap_update,
        
        "-standardization=" + str(standardization),
        "-standardization_scope=" + str(standardization_scope),
        "-standardization_variance=" + str(standardization_variance),
        "-save_standardization_stats=" + _bool(save_standardization_stats),

        "-MissingStrings=" + _list_arg(missing_indicators),
        "-entry_separator=" + entry_separator,
        "-array_separator=" + array_separator,
        "-out=" + output_directory,
    ])

    _append_common_distance_arg(msgList, "distances", distances)
    _append_common_distance_arg(msgList, "missing_proximity_distances", missing_proximity_distances)
    
    _append_if_not_none(msgList, "reader_type", reader_type)
    _append_if_not_none(msgList, "file_pattern", file_pattern)
    _append_if_not_none(msgList, "id_column", id_column)
    _append_if_not_none(msgList, "time_column", time_column)

    if feature_columns is not None:
        msgList.append(f"-feature_columns={_list_arg(feature_columns)}")

    if label_columns is not None:
        msgList.append(f"-label_columns={_list_arg(label_columns)}")
        
    _append_if_not_none(msgList, "hdf5_dataset_path", hdf5_dataset_path)
    _append_if_not_none(msgList, "hdf5_label_dataset_path", hdf5_label_dataset_path)
    
    _append_if_not_none(msgList, "standardization_stats", standardization_stats)
    _append_if_not_none(msgList, "standardization_stats_output", standardization_stats_output)

    if knn_distances is not None:
        _append_common_distance_arg(msgList, "knn_distances", knn_distances)
        
    if seed is not None:
        msgList.append("-seed=" + str(seed))

    return subprocess.call(msgList)


def predict(
    model_name,
    testfile,
    test_labels=None,
    exists_testlabels=False,
    return_predictions=False,
    return_proximities=False,
    proximity_type="PFGAP",
    output_directory="",
    shuffle=False,
    export=1,
    verbosity=1,
    file_has_header=False,
    forest_mode=None,
    target_column="first",
    parallel_trees=False,
    parallel_prox=False,
    parallel_predict=False,
    memory="1g",

    # Data controls
    data_dimension=1,
    numeric_data=True,
    entry_separator=",",
    array_separator=":",
    reader_type=None,
    file_pattern=None,
    id_column=None,
    time_column=None,
    feature_columns=None,
    label_columns=None,
    hdf5_dataset_path="/X",
    hdf5_label_dataset_path="/y",
    
    # Standardization controls - probably set during training
    standardization="none",
    standardization_scope="per_dimension",
    standardization_variance="population",
    standardization_stats=None,
    save_standardization_stats=False,
    standardization_stats_output=None,

    # Missing/imputation controls
    has_missing_values=None,
    impute_testing_data=False,
    impute_iterations=5,
    return_imputed_testing=False,
    initial_imputer="mean",
    DTWImpute=False,
    gap_update=None,
    imputation_initialization="impute_first",
    missing_proximity_distances=None,
    knn_distances=None,
    missing_indicators=("", "NA", "NaN", "null", "nan", "NAN"),

    # Optional runtime distances
    distances=None
):
    if data_dimension not in [1, 2]:
        raise ValueError("Keyword argument 'data_dimension' must be 1 or 2.")

    is2D = data_dimension == 2

    if has_missing_values is None:
        has_missing_values = (
            impute_testing_data
            or imputation_initialization == "proximity_first"
            or missing_proximity_distances is not None
        )
        
    # you must impute the data if you want imputed data returned.
    if return_imputed_testing and not impute_testing_data:
        impute_testing_data = True
        
    _validate_lazy_reader_options(
        reader_type=reader_type,
        file_pattern=file_pattern,
        impute_testing_data=impute_testing_data,
        return_imputed_testing=return_imputed_testing,
    )

    if gap_update is None:
        gap_update = "dtw_alignment" if DTWImpute else "standard"

    entry_separator = _separator_arg(entry_separator)
    array_separator = _separator_arg(array_separator)
    output_directory = _ensure_output_directory(output_directory)

    msgList = ["java", "-Xmx" + memory, "-jar", "PFGAP.jar", "-eval=true"]

    msgList.extend([
        "-train=" + str(testfile),
        "-test=" + str(testfile),
        "-test_labels=" + str(test_labels),
        "-exists_testlabels=" + _bool(exists_testlabels),

        "-shuffle=" + _bool(shuffle),
        "-export=" + str(export),
        "-verbosity=" + str(verbosity),
        "-csv_has_header=" + _bool(file_has_header),
        "-target_column=" + target_column,

        "-getprox=" + _bool(return_proximities),
        "-proximity_type=" + _proximity_type_arg(proximity_type),
        "-get_predictions=" + _bool(return_predictions),
        "-modelname=" + model_name,

        "-parallelTrees=" + _bool(parallel_trees),
        "-parallelProx=" + _bool(parallel_prox),
        "-parallelPredict=" + _bool(parallel_predict),

        "-is2D=" + _bool(is2D),
        "-isNumeric=" + _bool(numeric_data),

        "-entry_separator=" + entry_separator,
        "-array_separator=" + array_separator,

        "-initial_imputer=" + initial_imputer,
        "-hasMissingValues=" + _bool(has_missing_values),
        "-perform_test_imputation=" + _bool(impute_testing_data),
        "-numImputes=" + str(impute_iterations),
        "-impute_test=" + _bool(return_imputed_testing),

        "-DTWImpute=" + _bool(DTWImpute),
        "-imputation_initialization=" + imputation_initialization,
        "-gap_update=" + gap_update,

        "-MissingStrings=" + _list_arg(missing_indicators),
        "-out=" + output_directory,
    ])

    _append_common_distance_arg(msgList, "distances", distances)
    _append_common_distance_arg(msgList, "missing_proximity_distances", missing_proximity_distances)
    
    _append_if_not_none(msgList, "reader_type", reader_type)
    _append_if_not_none(msgList, "file_pattern", file_pattern)
    _append_if_not_none(msgList, "id_column", id_column)
    _append_if_not_none(msgList, "time_column", time_column)

    if feature_columns is not None:
        msgList.append(f"-feature_columns={_list_arg(feature_columns)}")

    if label_columns is not None:
        msgList.append(f"-label_columns={_list_arg(label_columns)}")
        
    _append_if_not_none(msgList, "hdf5_dataset_path", hdf5_dataset_path)
    _append_if_not_none(msgList, "hdf5_label_dataset_path", hdf5_label_dataset_path)

    if knn_distances is not None:
        _append_common_distance_arg(msgList, "knn_distances", knn_distances)

    return subprocess.call(msgList)



def getArray(filename):
    with open(filename) as f:
        contents = f.read()

    contents = contents.replace("{", "[")
    contents = contents.replace("}", "]")

    return np.array(ast.literal_eval(contents))

