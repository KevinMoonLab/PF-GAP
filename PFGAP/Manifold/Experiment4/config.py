"""
Configuration file for the PFGAP vs KNN testing pipeline
"""

# Test Configuration
K_VALUES = [1, 4, 8, 12, 16, 20]
TRAIN_TEST_SPLIT = 0.7
RANDOM_STATE = 42

# PFGAP Configuration
PFGAP_NUM_TREES = 11
PFGAP_R = 5
PFGAP_REPEATS = 1
PFGAP_ON_TREE = "true"
PFGAP_SHUFFLE = "false"
PFGAP_EXPORT = 1
PFGAP_VERBOSITY = 1
PFGAP_CSV_HAS_HEADER = "false"
PFGAP_TARGET_COLUMN = "last"  # Since we append labels to the end

# Data Configuration
DATA_DIMENSIONS = 3  # 3D spherical data
CLUSTER_SAMPLES = 150  # Samples per cluster
VON_MISES_KAPPA = 20  # Concentration parameter for von Mises-Fisher

# Parallel Processing
N_JOBS = -1  # Use all available CPU cores
VERBOSE = 10  # Joblib verbosity level

# File Paths
RESULTS_DIR = "results"
PROXIMITY_DIR = "proximity_matrices"
OUTLIER_DIR = "outlier_scores"
METRICS_DIR = "metrics"
TEMP_DIR = "temp"

# File Extensions
PROXIMITY_EXT = ".npy"
OUTLIER_EXT = ".npy"
METRICS_EXT = ".json"
COMPREHENSIVE_EXT = ".json"
SUMMARY_EXT = ".json"

# Error Handling
CONTINUE_ON_ERROR = True
SAVE_ERROR_RESULTS = True
CLEANUP_TEMP_FILES = True

# Logging
LOG_LEVEL = "INFO"
SAVE_LOG_TO_FILE = True
LOG_FILE = "pipeline.log"
