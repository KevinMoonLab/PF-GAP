package core;

import datasets.ListObjectDataset;
import trees.ProximityForest;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.HashSet;
import java.util.Set;

public class ModelIO {

    public static void saveModel(String path, ProximityForest forest, ListObjectDataset trainData, AppContextSnapshot snapshot) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(forest);
            out.writeObject(trainData);
            out.writeObject(snapshot);
        }
    }

    /*public static LoadedModel loadModel(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            ProximityForest forest = (ProximityForest) in.readObject();
            ListObjectDataset trainData = (ListObjectDataset) in.readObject();
            AppContextSnapshot snapshot = (AppContextSnapshot) in.readObject();
            return new LoadedModel(forest, trainData, snapshot);
        }
    }*/

    public static LoadedModel loadModel(String path) throws IOException, ClassNotFoundException {
        // Collect JARs from descriptors
        Set<File> interopJars = new HashSet<>();
        if (AppContext.Descriptors != null) {
            for (String[] desc : AppContext.Descriptors) {
                if (desc.length > 0 && desc[0] != null && !desc[0].isEmpty()) {
                    String[] parts = desc[0].split(":");
                    if (parts.length >= 2 && parts[0].equals("javadistance")) {
                        File jarFile = new File(parts[1]);
                        if (jarFile.exists()) {
                            interopJars.add(jarFile);
                        } else {
                            System.err.println("Warning: JAR file not found: " + jarFile.getAbsolutePath());
                        }
                    }
                }
            }
        }

        // Create a custom class loader
        URL[] urls = interopJars.stream().map(f -> {
            try {
                return f.toURI().toURL();
            } catch (MalformedURLException e) {
                throw new RuntimeException("Invalid JAR path: " + f.getAbsolutePath(), e);
            }
        }).toArray(URL[]::new);

        ClassLoader interopLoader = new URLClassLoader(urls, ClassLoader.getSystemClassLoader());

        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path)) {
            @Override
            protected Class<?> resolveClass(ObjectStreamClass desc) throws IOException, ClassNotFoundException {
                try {
                    return Class.forName(desc.getName(), false, interopLoader);
                } catch (ClassNotFoundException e) {
                    return super.resolveClass(desc);
                }
            }
        }) {
            ProximityForest forest = (ProximityForest) in.readObject();
            ListObjectDataset trainData = (ListObjectDataset) in.readObject();
            AppContextSnapshot snapshot = (AppContextSnapshot) in.readObject();
            return new LoadedModel(forest, trainData, snapshot);
        }
    }


    public static void applySnapshot(AppContextSnapshot snapshot) {
        AppContext.config_majority_vote_tie_break_randomly = snapshot.config_majority_vote_tie_break_randomly;
        AppContext.config_skip_distance_when_exemplar_matches_query = snapshot.config_skip_distance_when_exemplar_matches_query;
        AppContext.config_use_random_choice_when_min_distance_is_equal = snapshot.config_use_random_choice_when_min_distance_is_equal;

        AppContext.rand_seed = snapshot.rand_seed;
        AppContext.verbosity = snapshot.verbosity;
        AppContext.export_level = snapshot.export_level;

        AppContext.training_file = snapshot.training_file;
        //AppContext.testing_file = snapshot.testing_file;
        AppContext.training_labels = snapshot.training_labels;
        //AppContext.testing_labels = snapshot.testing_labels;
        AppContext.is2D = snapshot.is2D;
        AppContext.isNumeric = snapshot.isNumeric;
        //AppContext.hasMissingValues = snapshot.hasMissingValues;
        //AppContext.numImputes = snapshot.numImputes;
        //AppContext.entry_separator = snapshot.entry_separator;
        //AppContext.array_separator = snapshot.array_separator;
        //AppContext.output_dir = snapshot.output_dir;
        //AppContext.csv_has_header = snapshot.csv_has_header;
        //AppContext.target_column_is_first = snapshot.target_column_is_first;
        //AppContext.eval = snapshot.eval;
        //AppContext.length = snapshot.length;
        AppContext.purity_measure = snapshot.purity_measure;
        AppContext.isRegression = snapshot.isRegression;
        AppContext.voting = snapshot.voting;
        AppContext.purity_threshold = snapshot.purity_threshold;

        //AppContext.num_repeats = snapshot.num_repeats;
        AppContext.num_trees = snapshot.num_trees;
        AppContext.num_candidates_per_split = snapshot.num_candidates_per_split;
        AppContext.random_dm_per_node = snapshot.random_dm_per_node;
        //AppContext.shuffle_dataset = snapshot.shuffle_dataset;
        //AppContext.warmup_java = snapshot.warmup_java;
        AppContext.garbage_collect_after_each_repetition = snapshot.garbage_collect_after_each_repetition;
        AppContext.print_test_progress_for_each_instances = snapshot.print_test_progress_for_each_instances;

        AppContext.enabled_distance_measures = snapshot.enabled_distance_measures;
        //AppContext.savemodel = snapshot.savemodel;
        //AppContext.getprox = snapshot.getprox;
        //AppContext.get_training_outlier_scores = snapshot.get_training_outlier_scores;
        //AppContext.get_predictions = snapshot.get_predictions;
        //AppContext.modelname = snapshot.modelname;
        AppContext.userdistances = snapshot.userdistances;
        AppContext.Descriptors = snapshot.Descriptors;
        //AppContext.parallelTrees = snapshot.parallelTrees;
        //AppContext.parallelProx = snapshot.parallelProx;
        //AppContext.parallelPredict = snapshot.parallelPredict;
        AppContext.max_depth = snapshot.max_depth;
        AppContext.impute_train = snapshot.impute_train;
        //AppContext.impute_test = snapshot.impute_test;
        //AppContext.exists_testlabels = snapshot.exists_testlabels;
        AppContext.useSparseProximities = snapshot.useSparseProximities;
        AppContext.setDatasetName(snapshot.datasetName);
    }

    public static class LoadedModel {
        public final ProximityForest forest;
        public final ListObjectDataset trainData;
        public final AppContextSnapshot snapshot;

        public LoadedModel(ProximityForest forest, ListObjectDataset trainData, AppContextSnapshot snapshot) {
            this.forest = forest;
            this.trainData = trainData;
            this.snapshot = snapshot;
        }
    }
}
