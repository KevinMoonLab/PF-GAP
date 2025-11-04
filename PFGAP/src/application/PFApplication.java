package application;

import core.AppContext;
import core.ExperimentRunner;
import distance.DistanceRegistry;
import distance.MEASURE;
import imputation.*;
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

	public static void main(String[] args) throws IOException {
		//Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir testdir0"});
		try {
			
			//args = test_args;
			//Integer testint = Integer.parseInt("2 3 3.444"[0]);
			//some default settings are specified in the AppContext class but here we
			//override the default settings using the provided command line arguments
			String imputerType = null;
			for (int i = 0; i < args.length; i++) {
				String[] options = args[i].trim().split("=");
				
				switch(options[0]) {
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
				case "-isRegression":
					AppContext.isRegression = Boolean.parseBoolean(options[1]);
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
				case "-impute_train":
					AppContext.impute_train = Boolean.parseBoolean(options[1]);
					break;
				case "-impute_test":
					AppContext.impute_test = Boolean.parseBoolean(options[1]);
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

