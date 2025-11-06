package core;

import distance.MEASURE;
import imputation.Imputer;
import java.io.Serializable;
import java.util.*;

public class AppContextSnapshot implements Serializable {
    private static final long serialVersionUID = 1L;

    public boolean config_majority_vote_tie_break_randomly;
    public boolean config_skip_distance_when_exemplar_matches_query;
    public boolean config_use_random_choice_when_min_distance_is_equal;

    public long rand_seed;
    public int verbosity;
    public int export_level;

    public String training_file;
    //public String testing_file;
    public String training_labels;
    //public String testing_labels;
    public boolean is2D;
    public boolean isNumeric;
    //public boolean hasMissingValues;
    //public int numImputes;
    //public String entry_separator;
    //public String array_separator;
    //public String output_dir;
    //public boolean csv_has_header;
    //public boolean target_column_is_first;
    //public boolean eval;
    //public int length;
    public String purity_measure;
    public boolean isRegression;
    public String voting;
    public double purity_threshold;

    //public int num_repeats;
    public int num_trees;
    public int num_candidates_per_split;
    public boolean random_dm_per_node;
    //public boolean shuffle_dataset;
    //public boolean warmup_java;
    public boolean garbage_collect_after_each_repetition;
    public int print_test_progress_for_each_instances;

    public MEASURE[] enabled_distance_measures;
    //public boolean savemodel;
    //public boolean getprox;
    //public boolean get_training_outlier_scores;
    //public boolean get_predictions;
    //public String modelname;
    public MEASURE[] userdistances;
    public List<String[]> Descriptors;
    public HashSet<String> MissingStrings;
    public Map<Integer, Object> meta_predictions;
    //public boolean parallelTrees;
    //public boolean parallelProx;
    //public boolean parallelPredict;
    public int max_depth;
    public boolean impute_train;
    //public boolean impute_test;
    //public boolean exists_testlabels;
    public boolean useSparseProximities;
    public String datasetName;

    // Optional: if needed for label restoration
    public Map<Object, Integer> initialClassLabels;
}