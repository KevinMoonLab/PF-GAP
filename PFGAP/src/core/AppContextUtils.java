package core;

import core.contracts.ObjectDataset;
import datasets.ListObjectDataset;

public class AppContextUtils {

    public static AppContextSnapshot captureSnapshot() {
        AppContextSnapshot snapshot = new AppContextSnapshot();

        snapshot.config_majority_vote_tie_break_randomly = AppContext.config_majority_vote_tie_break_randomly;
        snapshot.config_skip_distance_when_exemplar_matches_query = AppContext.config_skip_distance_when_exemplar_matches_query;
        snapshot.config_use_random_choice_when_min_distance_is_equal = AppContext.config_use_random_choice_when_min_distance_is_equal;

        snapshot.rand_seed = AppContext.rand_seed;
        snapshot.verbosity = AppContext.verbosity;
        snapshot.export_level = AppContext.export_level;

        snapshot.training_file = AppContext.training_file;
        //snapshot.testing_file = AppContext.testing_file;
        snapshot.training_labels = AppContext.training_labels;
        //snapshot.testing_labels = AppContext.testing_labels;
        snapshot.is2D = AppContext.is2D;
        snapshot.isNumeric = AppContext.isNumeric;
        //snapshot.hasMissingValues = AppContext.hasMissingValues;
        //snapshot.numImputes = AppContext.numImputes;
        //snapshot.entry_separator = AppContext.entry_separator;
        //snapshot.array_separator = AppContext.array_separator;
        //snapshot.output_dir = AppContext.output_dir;
        //snapshot.csv_has_header = AppContext.csv_has_header;
        //snapshot.target_column_is_first = AppContext.target_column_is_first;
        //snapshot.eval = AppContext.eval;
        //snapshot.length = AppContext.length;
        snapshot.purity_measure = AppContext.purity_measure;
        snapshot.isRegression = AppContext.isRegression;
        snapshot.voting = AppContext.voting;
        snapshot.purity_threshold = AppContext.purity_threshold;

        //snapshot.num_repeats = AppContext.num_repeats;
        snapshot.num_trees = AppContext.num_trees;
        snapshot.num_candidates_per_split = AppContext.num_candidates_per_split;
        snapshot.random_dm_per_node = AppContext.random_dm_per_node;
        //snapshot.shuffle_dataset = AppContext.shuffle_dataset;
        //snapshot.warmup_java = AppContext.warmup_java;
        snapshot.garbage_collect_after_each_repetition = AppContext.garbage_collect_after_each_repetition;
        snapshot.print_test_progress_for_each_instances = AppContext.print_test_progress_for_each_instances;

        snapshot.enabled_distance_measures = AppContext.enabled_distance_measures;
        //snapshot.savemodel = AppContext.savemodel;
        //snapshot.getprox = AppContext.getprox;
        //snapshot.get_training_outlier_scores = AppContext.get_training_outlier_scores;
        //snapshot.get_predictions = AppContext.get_predictions;
        //snapshot.modelname = AppContext.modelname;
        snapshot.userdistances = AppContext.userdistances;
        snapshot.Descriptors = AppContext.Descriptors;
        //snapshot.parallelTrees = AppContext.parallelTrees;
        //snapshot.parallelProx = AppContext.parallelProx;
        //snapshot.parallelPredict = AppContext.parallelPredict;
        snapshot.max_depth = AppContext.max_depth;
        snapshot.impute_train = AppContext.impute_train;
        //snapshot.impute_test = AppContext.impute_test;
        //snapshot.exists_testlabels = AppContext.exists_testlabels;
        snapshot.useSparseProximities = AppContext.useSparseProximities;
        snapshot.datasetName = AppContext.getDatasetName();

        // Optional: include label mapping if available
        ObjectDataset trainData = AppContext.getTraining_data();
        if (trainData instanceof ListObjectDataset) {
            snapshot.initialClassLabels = ((ListObjectDataset) trainData)._get_initial_class_labels();
        }

        return snapshot;
    }
}