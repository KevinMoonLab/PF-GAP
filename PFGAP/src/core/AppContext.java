package core;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

//import core.contracts.Dataset;
//import distance.elastic.MEASURE;
import core.contracts.ObjectDataset;
import distance.MEASURE;
import imputation.Imputer;
import imputation.MeanImpute;
import imputation.MissingIndices;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class AppContext {
	
	private static final long serialVersionUID = -502980220452234173L;
	public static final String version = "1.0.0";
	
	public static final int ONE_MB = 1048576;	
	public static final String TIMESTAMP_FORMAT_LONG = "yyyy-MM-dd HH:mm:ss.SSS";	
	public static final String TIMESTAMP_FORMAT_SHORT = "HH:mm:ss.SSS";	
	
	
	//********************************************************************
	//DEVELOPMENT and TESTING AREA -- 
	public static boolean config_majority_vote_tie_break_randomly = true;
	public static boolean config_skip_distance_when_exemplar_matches_query = true;
	public static boolean config_use_random_choice_when_min_distance_is_equal = true;	
	//********************************************************************
	
	//DEFAULT SETTINGS, these are overridden by command line arguments
	public static long rand_seed;	//TODO set seed to reproduce results
	public static Random rand;
	
	public static int verbosity = 0; //0, 1, 2 
	public static int export_level = 1; //0, 1, 2 

	public static String training_file = System.getProperty("user.dir") + "/Data/" + "GunPoint" + "_TRAIN.tsv"; //"E:/data/ucr/cleaned/ItalyPowerDemand/ItalyPowerDemand_TRAIN.csv";
	public static String testing_file = System.getProperty("user.dir") + "/Data/" + "GunPoint" + "_TEST.tsv"; //"E:/data/ucr/cleaned/ItalyPowerDemand/ItalyPowerDemand_TEST.csv";
	public static String training_labels = null; // sometimes this is inferred from training_file.
	public static String testing_labels = null; // sometimes this is inferred from testing_file.
	public static boolean is2D = false; // this becomes true for multiTS and (probably) graph data.
	public static boolean isNumeric = true; // TODO: write distances for string, boolean, date types.
	public static boolean hasMissingValues = false; //this COULD be figured out... but on the other hand one should probably know their data before ramming it into a classifier.
	public static Imputer initial_imputer = new MeanImpute();
	public static int numImputes = 0; //when this is greater than 0, hasMissingValues becomes true.
	public static String entry_separator = "\t"; // the default for univariate time series (tsv).
	public static String array_separator = ":"; // is there a convention for this??
	// in the matrix case, "rows" are separated by firstSeparator and "columns" by secondSeparator.
	// in the list case (univariate time series, tabular data), only the firstSeparator is used.
	public static String output_dir = "output/";
	public static boolean csv_has_header = false;
	public static boolean target_column_is_first = true;
	public static boolean eval;
	public static int length; //firstSeparator
	public static String purity_measure = "gini";
	public static boolean isRegression = false;
	public static String voting = "mean";
	public static double purity_threshold = 1e-6;


	public static int num_repeats = 1;
	public static int num_trees = 11;
	public static int num_candidates_per_split = 1;
	public static boolean random_dm_per_node = true;
	public static boolean shuffle_dataset = false;
		
	public static boolean warmup_java = false;
	public static boolean garbage_collect_after_each_repetition = true;	
	
	public static int print_test_progress_for_each_instances = 100;
	
	// These distances are the default when none are specified.
	public static MEASURE[] enabled_distance_measures = new MEASURE[] {
			MEASURE.euclidean,
			MEASURE.dtw,
			MEASURE.dtwcv,
			MEASURE.ddtw,
			MEASURE.ddtwcv,
			MEASURE.wdtw,
			MEASURE.wddtw,
			MEASURE.lcss,
			MEASURE.erp,
			MEASURE.twe,
			MEASURE.msm
	};	

	public static Runtime runtime = Runtime.getRuntime();
    public static boolean savemodel;
	public static boolean getprox;
	public static boolean get_training_outlier_scores;
	public static boolean get_predictions = false;
	public static String modelname = "Thor";
	public static MEASURE[] userdistances; //= {MEASURE.dtw};
	public static MEASURE[] KNNdistances; //only used in KNN initial imputation.
	public static List<String[]> Descriptors = new ArrayList<>(); //this is specifically to store file names for custom java distances.
	public static boolean parallelTrees = false; //false;
	public static boolean parallelProx = false; //false;
	public static boolean parallelPredict = false; // if parallelTrees=true, predictions will be made in parallel across trees.
	// parallelPredict refers to parallelization across data instances (will not happen if parallelTrees=true).
	public static int max_depth; //initializes to 0.
	public static boolean impute_train = false;
	public static boolean impute_test = false;
	public static boolean DTWImpute = false;
	public static HashSet<String> MissingStrings;
	public static Map<Integer, Object> meta_predictions;
	// the missing indices are now part of the ListObjectDataset.
	/*//public static ArrayList<Integer> missing_train_indices;
	//public static List<Object> missing_train_indices = Collections.synchronizedList(new ArrayList<>());
	public static List<MissingIndices> missing_train_indices = new CopyOnWriteArrayList<>();
	//public static ArrayList<Integer> missing_test_indices;
	public static List<MissingIndices> missing_test_indices = new CopyOnWriteArrayList<>();*/

	//private static transient Dataset train_data;
	private static transient ObjectDataset train_data;
	//private static transient Dataset test_data;
	private static transient  ObjectDataset test_data;
	private static String datasetName;
	public static boolean exists_testlabels = false;
	public static transient double[][] training_proximities;
	public static transient double[][] testing_training_proximities;
	public static boolean useSparseProximities = true; //should be dense if returned??
	public static Map<Integer, Map<Integer, Double>> training_proximities_sparse;
	public static Map<Integer, Map<Integer, Double>> testing_training_proximities_sparse;

	static {
		rand = new Random();
	}

	public static Random getRand() {
		return rand;
	}

	//public static Dataset getTraining_data() {
	public static ObjectDataset getTraining_data() {
		return train_data;
	}

	//public static void setTraining_data(Dataset train_data) {
	public static void setTraining_data(ObjectDataset train_data) {
		AppContext.train_data = train_data;
	}

	//public static Dataset getTesting_data() {
	public static ObjectDataset getTesting_data() {
		return test_data;
	}

	//public static void setTesting_data(Dataset test_data) {
	public static void setTesting_data(ObjectDataset test_data) {
		AppContext.test_data = test_data;
	}

	public static String getDatasetName() {
		return datasetName;
	}

	public static void setDatasetName(String datasetName) {
		AppContext.datasetName = datasetName;
	}
}
