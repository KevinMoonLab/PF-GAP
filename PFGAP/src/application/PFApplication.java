package application;

import core.AppContext;
import core.ExperimentRunner;
import datasets.readers.ReaderType;
import distance.DistanceRegistry;
import distance.MEASURE;
import imputation.initial.*;
import preprocessing.standardization.StandardizationConfig;
import preprocessing.standardization.StandardizationMethod;
import preprocessing.standardization.StandardizationScope;
import preprocessing.standardization.VarianceConvention;
import proximities.ProximityType;
import util.GeneralUtilities;
import util.PrintUtilities;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Main entry point for the Proximity Forest application
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class PFApplication {
	
	public static final String UCR_dataset = "GunPoint"; //"ItalyPowerDemand";
	//TODO test support file paths with a space?
	public static final String[] test_args = new String[]{
			"-train=" + System.getProperty("user.dir") + "/Data/" + UCR_dataset + "_TRAIN.tsv", //"-train=E:/data/ucr/" + UCR_dataset + "/" + UCR_dataset + "_TRAIN.txt",
			"-test=" + System.getProperty("user.dir") + "/Data/" + UCR_dataset + "_TEST.tsv",
//			"-train=E:/data/satellite/sample100000_TRAIN.txt", 
//			"-test=E:/data/satellite/sample100000_TEST.txt",
			"-out=output",
			"-repeats=1",
			"-trees=10",
			"-r=5",
			"-on_tree=true",
			"-shuffle=true",
//			"-jvmwarmup=true",	//disabled
			"-export=1",
			"-verbosity=1",
			"-csv_has_header=false", 
			"-target_column=first"	//first or last
            };

	private static MEASURE[] parseMeasureList(
			String raw,
			String argumentName) {

		if (raw == null) {
			return new MEASURE[]{};
		}

		String trimmed = raw.trim();

		if (trimmed.equals("[]")) {
			return new MEASURE[]{};
		}

		if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
			throw new IllegalArgumentException(
					"Invalid -" + argumentName
							+ " format. Expected [distance1,distance2]."
			);
		}

		String contents = trimmed.substring(1, trimmed.length() - 1).trim();

		if (contents.isEmpty()) {
			return new MEASURE[]{};
		}

		String[] names = contents.split(",");
		MEASURE[] measures = new MEASURE[names.length];

		for (int i = 0; i < names.length; i++) {
			String name = names[i].trim();

			if (!DistanceRegistry.contains(name)) {
				throw new IllegalArgumentException(
						"Unknown distance in -" + argumentName + ": "
								+ name
								+ ". Available distances: "
								+ DistanceRegistry.getAll().keySet()
				);
			}

			measures[i] = DistanceRegistry.get(name);

			if (argumentName.equals("missing_proximity_distances")
					&& !isMissingCompatibleDistance(measures[i])) {
				throw new IllegalArgumentException(
						"Distance "
								+ name
								+ " is not missing-compatible. Use one of: "
								+ "nan_euclidean, nan_euclidean_i, "
								+ "dtwarow, dtwarow_i, dtwarow_d."
				);
			}
		}

		return measures;
	}

	private static boolean isMissingCompatibleDistance(MEASURE measure) {

		return measure == MEASURE.nan_euclidean
				|| measure == MEASURE.nan_euclidean_i
				|| measure == MEASURE.dtwarow
				|| measure == MEASURE.dtwarow_i
				|| measure == MEASURE.dtwarow_d;
	}

	private static ReaderType parseReaderType(String raw) {

		if (raw == null || raw.trim().isEmpty()) {
			return null;
		}

		try {
			return ReaderType.valueOf(raw.trim().toUpperCase(Locale.ROOT));
		} catch (IllegalArgumentException e) {
			throw new IllegalArgumentException(
					"Invalid reader_type: "
							+ raw
							+ ". Valid options are: "
							+ Arrays.toString(ReaderType.values())
			);
		}
	}

	private static List<String> parseStringList(String raw) {

		List<String> values = new ArrayList<>();

		if (raw == null) {
			return values;
		}

		String trimmed = raw.trim();

		if (trimmed.isEmpty()
				|| trimmed.equalsIgnoreCase("None")
				|| trimmed.equals("[]")) {
			return values;
		}

		/*
		 * Accept both:
		 *
		 *    temp,pressure,humidity
		 *
		 * and:
		 *
		 *    [temp,pressure,humidity]
		 */
		if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
			trimmed = trimmed.substring(1, trimmed.length() - 1).trim();
		}

		if (trimmed.isEmpty()) {
			return values;
		}

		String[] parts = trimmed.split(",");

		for (String part : parts) {
			String value = part.trim();

			if (!value.isEmpty()) {
				values.add(value);
			}
		}

		return values;
	}

	private static String parseNullableString(String raw) {

		if (raw == null) {
			return null;
		}

		String trimmed = raw.trim();

		if (trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")) {
			return null;
		}

		return trimmed;
	}

	public static void main(String[] args) throws IOException {
		//Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir testdir0"});
		try {
			
			//args = test_args;
			//Integer testint = Integer.parseInt("2 3 3.444"[0]);
			//some default settings are specified in the AppContext class but here we
			//override the default settings using the provided command line arguments

			// set some before the switch case:
			String imputerType = null;
			StandardizationMethod standardizationMethod =
					StandardizationMethod.NONE;

			StandardizationScope standardizationScope =
					StandardizationScope.PER_DIMENSION;

			VarianceConvention standardizationVariance =
					VarianceConvention.POPULATION;

			String standardizationStatsPath =
					null;

			boolean saveStandardizationStats =
					false;

			String standardizationStatsOutput =
					null;

			for (int i = 0; i < args.length; i++) {
				//String[] options = args[i].trim().split("=");
				String[] options = args[i].trim().split("=", 2);

				if (options.length != 2) {
					throw new IllegalArgumentException(
							"Invalid command-line argument: "
									+ args[i]
									+ ". Expected -name=value."
					);
				}
				
				switch(options[0]) {
				case "-seed":
					AppContext.setRandomSeed(Long.parseLong(options[1]));
					break;
				case "-bootstrap_trees":
					AppContext.bootstrap_trees = Boolean.parseBoolean(options[1]);
					break;
				case "-eval":
					AppContext.eval = Boolean.parseBoolean(options[1]);
					break;
				case "-train":
					AppContext.training_file = options[1];
					break;
				case "-test":
					if (Objects.equals(options[1], "None")) {
						AppContext.testing_file = null;
					} else {
						AppContext.testing_file = options[1];
					}
					break;
				case "-train_labels":
					if (Objects.equals(options[1], "None")) {
						AppContext.training_labels = null;
					} else {
						AppContext.training_labels = options[1];
					}
					break;
				case "-test_labels":
					if (Objects.equals(options[1], "None")) {
						AppContext.testing_labels = null;
					} else {
						AppContext.testing_labels = options[1];
						AppContext.exists_testlabels = true;
					}
					break;
				case "-exists_testlabels":
					if (AppContext.exists_testlabels) {
						break;
					} else {
						AppContext.exists_testlabels = Boolean.parseBoolean(options[1]);
						break;
					}
				case "-reader_type":
					AppContext.readerType = parseReaderType(options[1]);
					break;
				case "-file_pattern":
					AppContext.file_pattern = parseNullableString(options[1]);
					break;
				case "-train_reader_type":
					AppContext.trainingReaderType =
							parseReaderType(options[1]);
					break;

				case "-test_reader_type":
					AppContext.testingReaderType =
							parseReaderType(options[1]);
					break;

				case "-train_file_pattern":
					AppContext.trainingFilePattern =
							parseNullableString(options[1]);
					break;

				case "-test_file_pattern":
					AppContext.testingFilePattern =
							parseNullableString(options[1]);
					break;
				case "-id_column":
					AppContext.id_column = parseNullableString(options[1]);
					break;

				case "-time_column":
					AppContext.time_column = parseNullableString(options[1]);
					break;

				case "-feature_columns":
					AppContext.feature_columns = parseStringList(options[1]);
					break;

				case "-label_columns":
					AppContext.label_columns = parseStringList(options[1]);
					break;
				case "-hdf5_dataset_path":
					AppContext.hdf5_dataset_path = parseNullableString(options[1]);
					break;
				case "-hdf5_label_dataset_path":
					AppContext.hdf5_label_dataset_path = parseNullableString(options[1]);
					break;
				case "-standardization":
					standardizationMethod =
							StandardizationMethod.fromString(
									options[1]
							);
					break;

				case "-standardization_scope":
					standardizationScope =
							StandardizationScope.fromString(
									options[1]
							);
					break;

				case "-standardization_variance":
					standardizationVariance =
							VarianceConvention.fromString(
									options[1]
							);
					break;

				case "-standardization_stats":
					standardizationStatsPath =
							parseNullableString(
									options[1]
							);
					break;

				case "-save_standardization_stats":
					saveStandardizationStats =
							Boolean.parseBoolean(
									options[1]
							);
					break;

					case "-standardization_stats_output":
						standardizationStatsOutput =
								parseNullableString(
										options[1]
								);
						break;
				case "-isRegression":
					AppContext.isRegression = Boolean.parseBoolean(options[1]);
					break;
				case "-forest_mode":
					AppContext.forest_mode = options[1].trim().toLowerCase();

					if (AppContext.forest_mode.equals("regression")) {
						AppContext.isRegression = true;
					} else if (AppContext.forest_mode.equals("classification")
							|| AppContext.forest_mode.equals("isolation")) {
						AppContext.isRegression = false;
					} else {
						throw new IllegalArgumentException(
								"Invalid forest_mode: " + options[1]
						);
					}

					break;
				case "-isolation_num_branches":
					AppContext.isolation_num_branches = Integer.parseInt(options[1]);
					break;

				case "-regression_num_branches":
					AppContext.regression_num_branches = Integer.parseInt(options[1]);
					break;

				case "-isolation_min_leaf_size":
					AppContext.isolation_min_leaf_size = Integer.parseInt(options[1]);
					break;
				case "-purity_measure":
					AppContext.purity_measure = options[1];
					break;
				case "-voting":
					AppContext.voting = options[1];
					break;
				case "-purity_threshold":
					AppContext.purity_threshold = Double.parseDouble(options[1]);
					break;
				case "-impute_train": // should we *return* imputed training set?
					AppContext.impute_train = Boolean.parseBoolean(options[1]);
					break;
				case "-impute_test": // should we *return* imputed test set?
					AppContext.impute_test = Boolean.parseBoolean(options[1]);
					break;
				case "-perform_train_imputation": // should we impute train data?
					AppContext.perform_train_imputation =
							Boolean.parseBoolean(options[1]);
					break;
				case "-perform_test_imputation": // should we impute test data?
					AppContext.perform_test_imputation =
							Boolean.parseBoolean(options[1]);
					break;
				case "-is2D":
					AppContext.is2D = Boolean.parseBoolean(options[1]);
					break;
				case "-isNumeric":
					AppContext.isNumeric = Boolean.parseBoolean(options[1]);
					break;
				case "-hasMissingValues":
					AppContext.hasMissingValues = Boolean.parseBoolean(options[1]);
					break;
				case "-numImputes":
					AppContext.numImputes = Integer.parseInt(options[1]);
					break;
				case "-entry_separator":
					AppContext.entry_separator = options[1];
					break;
				case "-array_separator":
					AppContext.array_separator = options[1];
					break;
				case "-out":
					AppContext.output_dir = options[1];
					break;
				case "-repeats":
					AppContext.num_repeats = Integer.parseInt(options[1]);
					break;
				case "-trees":
					AppContext.num_trees = Integer.parseInt(options[1]);
					break;
				case "-r":
					AppContext.num_candidates_per_split = Integer.parseInt(options[1]);
					break;
				case "-on_tree":
					AppContext.random_dm_per_node = Boolean.parseBoolean(options[1]);
					break;
				case "-max_depth":
					AppContext.max_depth = Integer.parseInt(options[1]);
					break;
				case "-shuffle":
					AppContext.shuffle_dataset = Boolean.parseBoolean(options[1]);
					break;
//				case "-jvmwarmup":	//TODO 
//					AppContext.warmup_java = Boolean.parseBoolean(options[1]);
//					break;
				case "-csv_has_header":
					AppContext.csv_has_header = Boolean.parseBoolean(options[1]);
					break;
				case "-target_column":
					if (options[1].trim().equals("first")) {
						AppContext.target_column_is_first = true;
					}else if (options[1].trim().equals("last")) {
						AppContext.target_column_is_first = false;
					}else {
						throw new Exception("Invalid Commandline Arguments");
					}
					break;
				case "-export":
					AppContext.export_level =  Integer.parseInt(options[1]);
					break;
				case "-verbosity":
					AppContext.verbosity =  Integer.parseInt(options[1]);
					break;
				case "-get_training_outlier_scores":
					AppContext.get_training_outlier_scores = Boolean.parseBoolean(options[1]);
					break;
				case "-getprox":
					AppContext.getprox = Boolean.parseBoolean(options[1]);
					break;
				case "-get_predictions":
					AppContext.get_predictions = Boolean.parseBoolean(options[1]);
					break;
				case "-modelname":
					AppContext.modelname = options[1];
					break;
				case "-savemodel":
					AppContext.savemodel = Boolean.parseBoolean(options[1]);
					break;
				case "-parallelTrees":
					AppContext.parallelTrees = Boolean.parseBoolean(options[1]);
					break;
				case "-parallelProx":
					AppContext.parallelProx = Boolean.parseBoolean(options[1]);
					break;
				case "-parallelPredict":
					AppContext.parallelPredict = Boolean.parseBoolean(options[1]);
					break;
				case "-parallelSplit":
					AppContext.parallel_split_assignments = Boolean.parseBoolean(options[1]);
					break;
				case "-parallelSplitThreshold":
					AppContext.parallel_split_assignment_threshold = Integer.parseInt(options[1]);
					break;
				case "-knn_distances":
					//String[] distanceNames = options[1].split(",");
					/*MEASURE[] measures = Arrays.stream(distanceNames)
							.map(String::trim)
							.map(name -> {
								if (!DistanceRegistry.contains(name)) {
									throw new IllegalArgumentException("Unknown distance: " + name);
								}
								return DistanceRegistry.get(name);
							})
							.toArray(MEASURE[]::new);
					AppContext.KNNdistances = measures;*/
					String ktemp = options[1];
					String ktemp_rm = ktemp.substring(1, ktemp.length() - 1); // Removes '[' and ']'
					String[] kcontents = ktemp_rm.split(","); // Splits by ","
					List<String> kcontentsList = Arrays.asList(kcontents);
					int knumberofdists = kcontentsList.size();
					MEASURE[] ktoadd = new MEASURE[knumberofdists];

					//Map<String, MEASURE> measuresByName = new HashMap<>();
					Map<String, MEASURE> kmeasuresByName = DistanceRegistry.getAll();

					for (int j=0; j < knumberofdists; j++){
						MEASURE convertedEntry;
						convertedEntry = kmeasuresByName.get(kcontentsList.get(j));
						//MEASURE convertedEntry = measuresByName.get(contentsList.get(j));
						ktoadd[j] = convertedEntry;
					}

					if (Objects.equals(kcontentsList.get(0), "")){
						AppContext.KNNdistances = new MEASURE[]{}; //new MEASURE[numberofdists];
					} else {
						AppContext.KNNdistances = ktoadd;
					}
					break;
				case "-initial_imputer":
					//String inputString = options[1];
					imputerType = options[1];

					/*switch (inputString.toLowerCase()) {
						case "mean":
							AppContext.initial_imputer = new MeanImpute();
							break;
						case "global_mean":
							AppContext.initial_imputer = new GlobalMeanImpute();
							break;
						case "linear":
							AppContext.initial_imputer = new LinearImpute();
							break;
						case "median":
							AppContext.initial_imputer = new MedianImpute();
							break;
						case "global_median":
							AppContext.initial_imputer = new GlobalMedianImpute();
							break;
						case "mode":
							AppContext.initial_imputer = new ModeImpute();
							break;
						case "global_mode":
							AppContext.initial_imputer = new GlobalModeImpute();
							break;
						case "knn":
							if (AppContext.KNNdistances == null || AppContext.KNNdistances.length == 0) {
								throw new IllegalArgumentException("KNN distances must be specified using -knn_distances");
							}
							AppContext.initial_imputer = new KNNImputer(AppContext.KNNdistances, 5);
							break;
						default:
							throw new IllegalArgumentException("Unknown imputer: " + options[1]);
					}*/
					break;

				case "-DTWImpute":
					AppContext.DTWImpute = Boolean.parseBoolean(options[1]);

					if (AppContext.DTWImpute) {
						AppContext.gap_update_strategy = "dtw_alignment";
					} else if (AppContext.gap_update_strategy == null) {
						AppContext.gap_update_strategy = "standard";
					}

					break;
				case "-imputation_initialization":
					String initStrategy = options[1].trim().toLowerCase();

					if (!initStrategy.equals("impute_first")
							&& !initStrategy.equals("proximity_first")) {
						throw new IllegalArgumentException(
								"Invalid -imputation_initialization value: "
										+ options[1]
										+ ". Use impute_first or proximity_first."
						);
					}

					AppContext.imputation_initialization_strategy = initStrategy;
					break;
				case "-proximity_type":
					try {
						AppContext.proximityType =
								ProximityType.valueOf(
										options[1].trim().toUpperCase(Locale.ROOT)
								);
					} catch (IllegalArgumentException e) {
						throw new Exception(
								"Invalid proximity_type: "
										+ options[1]
										+ ". Valid options are: "
										+ Arrays.toString(ProximityType.values())
						);
					}
					break;
				case "-gap_update":
					String gapUpdate = options[1].trim().toLowerCase();

					if (!gapUpdate.equals("standard")
							&& !gapUpdate.equals("dtw_alignment")) {
						throw new IllegalArgumentException(
								"Invalid -gap_update value: "
										+ options[1]
										+ ". Use standard or dtw_alignment."
						);
					}

					AppContext.gap_update_strategy = gapUpdate;

					/*
					 * Backward compatibility with older boolean flag.
					 */
					AppContext.DTWImpute = gapUpdate.equals("dtw_alignment");
					break;

				case "-missing_proximity_distances":
					AppContext.missing_proximity_distances =
							parseMeasureList(options[1], "missing_proximity_distances");
					break;
				case "-MissingStrings":
					String temp_strings = options[1];
					String temp_strings_rm = temp_strings.substring(1, temp_strings.length() - 1); // Removes '[' and ']'
					String[] contents_strings = temp_strings_rm.split(","); // Splits by ","
					AppContext.MissingStrings = new HashSet<>(Arrays.asList(contents_strings));
					break;
				case "-distances":
					String temp = options[1];
					String temp_rm = temp.substring(1, temp.length() - 1); // Removes '[' and ']'
					String[] contents = temp_rm.split(","); // Splits by ","
					List<String> contentsList = Arrays.asList(contents);
					int numberofdists = contentsList.size();
					MEASURE[] toadd = new MEASURE[numberofdists];

					//Map<String, MEASURE> measuresByName = new HashMap<>();
					Map<String, MEASURE> measuresByName = DistanceRegistry.getAll();

					for (int j=0; j < numberofdists; j++){
						MEASURE convertedEntry;
						String distanceString = contentsList.get(j);
						if (distanceString.startsWith("javadistance:")) {
							// check the format
							String[] parts = distanceString.split(":");
							if (parts.length < 2) {
								throw new IllegalArgumentException("Invalid descriptor format. Use javadistance:path/to/file[:ClassName]");
							}
							// check that it's a real file.
							String path = parts[1];
							File file = new File(path);
							if (!file.exists()) {
								throw new IllegalArgumentException("File not found: " + path);
							}

							// Save to AppContext so that it can be invoked when initialized.
							String[] descriptor = new String[]{distanceString};
							AppContext.Descriptors.add(descriptor);
							convertedEntry = measuresByName.get("javadistance");
						} else if (distanceString.startsWith("python:")) {
							// check the format
							String[] parts = distanceString.split(":");
							if (parts.length < 2) {
								throw new IllegalArgumentException("Invalid descriptor format. Use python:path/to/file[:FunctionName]");
							}
							// check that it's a real file.
							String path = parts[1];
							File file = new File(path);
							if (!file.exists()) {
								throw new IllegalArgumentException("File not found: " + path);
							}

							// Save to AppContext so that it can be invoked when initialized.
							String[] descriptor = new String[]{distanceString};
							AppContext.Descriptors.add(descriptor);
							convertedEntry = measuresByName.get("python");
						} else if (distanceString.startsWith("maple:")) {
							// check the format
							String[] parts = distanceString.split(":");
							if (parts.length < 2) {
								throw new IllegalArgumentException("Invalid descriptor format. Use maple:path/to/file[:FunctionName]");
							}
							// check that it's a real file.
							String path = parts[1];
							File file = new File(path);
							if (!file.exists()) {
								throw new IllegalArgumentException("File not found: " + path);
							}

							// Save to AppContext so that it can be invoked when initialized.
							String[] descriptor = new String[]{distanceString};
							AppContext.Descriptors.add(descriptor);
							convertedEntry = measuresByName.get("maple");
						} else if (distanceString.startsWith("meta_")) {
							// check the format
							String[] parts = distanceString.split(":");
							if (parts.length < 2) {
								throw new IllegalArgumentException("Invalid descriptor format. Use meta_type:path/to/file[:method]");
							}
							// check that it's a real file.
							String path = parts[1];
							File file = new File(path);
							if (!file.exists()) {
								throw new IllegalArgumentException("File not found: " + path);
							}

							// Save to AppContext so that it can be invoked when initialized.
							String[] descriptor = new String[]{distanceString};
							AppContext.Descriptors.add(descriptor);

							// Use the prefix (e.g., "meta_file_classmatch") to get the correct MEASURE
							String key = distanceString.split(":")[0];
							convertedEntry = measuresByName.get(key);
						} else {
							// we'll just add an empty string list (to keep track of indices).
							String[] descriptor = new String[]{""};
							AppContext.Descriptors.add(descriptor);
							convertedEntry = measuresByName.get(contentsList.get(j));
						}
						//MEASURE convertedEntry = measuresByName.get(contentsList.get(j));
						toadd[j] = convertedEntry;
					}

					if (Objects.equals(contentsList.get(0), "")){
						AppContext.userdistances = new MEASURE[]{}; //new MEASURE[numberofdists];
					} else {
						AppContext.userdistances = toadd;
					}

					//AppContext.userdistances = toadd;
					break;
				default:
					throw new Exception("Invalid Commandline Arguments");
				}
			}

			if (imputerType !=null) {
				switch (imputerType) {
					case "knn":
						if (AppContext.KNNdistances == null || AppContext.KNNdistances.length == 0)
							throw new IllegalArgumentException("Missing -knn_distances for KNN imputer.");
						AppContext.initial_imputer = new KNNImputer(AppContext.KNNdistances, 5);
						break;
					case "mean":
						AppContext.initial_imputer = new MeanImpute();
						break;
					case "global_mean":
						AppContext.initial_imputer = new GlobalMeanImpute();
						break;
					case "linear":
						AppContext.initial_imputer = new LinearImpute();
						break;
					case "median":
						AppContext.initial_imputer = new MedianImpute();
						break;
					case "global_median":
						AppContext.initial_imputer = new GlobalMedianImpute();
						break;
					case "mode":
						AppContext.initial_imputer = new ModeImpute();
						break;
					case "global_mode":
						AppContext.initial_imputer = new GlobalModeImpute();
						break;
				}
			}

			AppContext.standardizationConfig =
					StandardizationConfig.builder()
							.setMethod(
									standardizationMethod
							)
							.setScope(
									standardizationScope
							)
							.setVarianceConvention(
									standardizationVariance
							)
							.setStatisticsPath(
									standardizationStatsPath
							)
							.setSaveFittedStatistics(
									saveStandardizationStats
							)
							.setStatisticsOutputPath(
									standardizationStatsOutput
							)
							.build();

			AppContext.standardizationConfig.requireImplemented();

			if (AppContext.warmup_java) {
				GeneralUtilities.warmUpJavaRuntime();
			}
						
			ExperimentRunner experiment = new ExperimentRunner();
			//experiment.run(false);
			experiment.run(AppContext.eval);
			
		}catch(Exception e) {			
            PrintUtilities.abort(e);
		}
		
	}


}

