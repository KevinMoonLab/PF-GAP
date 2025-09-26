package application;

import core.AppContext;
import core.ExperimentRunner;
//import distance.elastic.MEASURE;
import distance.DistanceRegistry;
import distance.MEASURE;
import imputation.LinearImpute;
import imputation.MeanImpute;
import imputation.MedianImpute;
import imputation.ModeImpute;
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
					}
					break;
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
				case "-initial_imputer":
					String inputString = options[1];

					switch (inputString.toLowerCase()) {
						case "mean":
							AppContext.initial_imputer = new MeanImpute();
							break;
						case "linear":
							AppContext.initial_imputer = new LinearImpute();
							break;
						case "median":
							AppContext.initial_imputer = new MedianImpute();
							break;
						case "mode":
							AppContext.initial_imputer = new ModeImpute();
							break;
						default:
							throw new IllegalArgumentException("Unknown imputer: " + options[1]);
					}
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

					/*// Associate string keys with MEASURE values
					measuresByName.put("basicDTW", MEASURE.basicDTW);
					measuresByName.put("dtwDistance", MEASURE.dtwDistance);
					measuresByName.put("dtwDistanceEfficient", MEASURE.dtwDistanceEfficient);
					measuresByName.put("erp", MEASURE.erp);
					measuresByName.put("lcss", MEASURE.lcss);
					measuresByName.put("msm", MEASURE.msm);
					measuresByName.put("pdtw", MEASURE.pdtw);
					measuresByName.put("scdtw", MEASURE.scdtw);
					measuresByName.put("twe", MEASURE.twe);
					measuresByName.put("wdtw", MEASURE.wdtw);
					measuresByName.put("francoisDTW", MEASURE.francoisDTW);
					measuresByName.put("smoothDTW", MEASURE.smoothDTW);
					measuresByName.put("dtw", MEASURE.dtw);
					measuresByName.put("dca_dtw", MEASURE.dca_dtw);
					measuresByName.put("euclidean", MEASURE.euclidean);
					measuresByName.put("equality", MEASURE.equality);
					measuresByName.put("dtwcv", MEASURE.dtwcv);
					measuresByName.put("ddtwcv", MEASURE.ddtwcv);
					measuresByName.put("wddtw", MEASURE.wddtw);
					measuresByName.put("ddtw", MEASURE.ddtw);
					measuresByName.put("shifazDTW", MEASURE.shifazDTW);
					measuresByName.put("shifazDTWCV", MEASURE.shifazDTWCV);
					measuresByName.put("shifazDDTW", MEASURE.shifazDDTW);
					measuresByName.put("shifazDDTWCV", MEASURE.shifazDDTWCV);
					measuresByName.put("shifazWDTW", MEASURE.shifazWDTW);
					measuresByName.put("shifazWDDTW", MEASURE.shifazWDDTW);
					measuresByName.put("shifazEUCLIDEAN", MEASURE.shifazEUCLIDEAN);
					measuresByName.put("shifazERP", MEASURE.shifazERP);
					measuresByName.put("shifazMSM", MEASURE.shifazMSM);
					measuresByName.put("shifazLCSS", MEASURE.shifazLCSS);
					measuresByName.put("shifazTWE", MEASURE.shifazTWE);
					measuresByName.put("maple", MEASURE.maple);
					measuresByName.put("python", MEASURE.python);
					measuresByName.put("manhattan", MEASURE.manhattan);
					measuresByName.put("shapeHoG1dDTW", MEASURE.shapeHoG1dDTW);
					measuresByName.put("dtw_i", MEASURE.dtw_i);
					measuresByName.put("dtw_d", MEASURE.dtw_d);

					// Multivariate Independent (_I)
					measuresByName.put("ddtw_i", MEASURE.ddtw_i);
					measuresByName.put("shifazDDTW_I", MEASURE.shifazDDTW_I);
					measuresByName.put("wdtw_i", MEASURE.wdtw_i);
					measuresByName.put("shifazWDTW_I", MEASURE.shifazWDTW_I);
					measuresByName.put("wddtw_i", MEASURE.wddtw_i);
					measuresByName.put("shifazWDDTW_I", MEASURE.shifazWDDTW_I);
					measuresByName.put("twe_i", MEASURE.twe_i);
					measuresByName.put("shifazTWE_I", MEASURE.shifazTWE_I);
					measuresByName.put("erp_i", MEASURE.erp_i);
					measuresByName.put("shifazERP_I", MEASURE.shifazERP_I);
					measuresByName.put("euclidean_i", MEASURE.euclidean_i);
					measuresByName.put("shifazEUCLIDEAN_I", MEASURE.shifazEUCLIDEAN_I);
					measuresByName.put("lcss_i", MEASURE.lcss_i);
					measuresByName.put("shifazLCSS_I", MEASURE.shifazLCSS_I);
					measuresByName.put("msm_i", MEASURE.msm_i);
					measuresByName.put("shifazMSM_I", MEASURE.shifazMSM_I);
					measuresByName.put("manhattan_i", MEASURE.manhattan_i);
					measuresByName.put("shifazMANHATTAN_I", MEASURE.shifazMANHATTAN_I);
					measuresByName.put("cid_i", MEASURE.cid_i);
					measuresByName.put("shifazCID_I", MEASURE.shifazCID_I);
					measuresByName.put("sbd_i", MEASURE.sbd_i);
					measuresByName.put("shifazSBD_I", MEASURE.shifazSBD_I);

// Multivariate Dependent (_D)
					measuresByName.put("ddtw_d", MEASURE.ddtw_d);
					measuresByName.put("wdtw_d", MEASURE.wdtw_d);
					measuresByName.put("wddtw_d", MEASURE.wddtw_d);
					measuresByName.put("shapeHoGdtw_d", MEASURE.shapeHoGdtw_d);
					//measuresByName.put("euclidean_d", MEASURE.euclidean_d);
					//measuresByName.put("manhattan_d", MEASURE.manhattan_d);

// Shape-based multivariate DTW
					measuresByName.put("shapeHoGdtw", MEASURE.shapeHoGdtw);
					measuresByName.put("shifazShapeHoGDTW", MEASURE.shifazShapeHoGDTW);*/

					for (int j=0; j < numberofdists; j++){
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
						} else {
							// we'll just add an empty string list (to keep track of indices).
							String[] descriptor = new String[]{""};
							AppContext.Descriptors.add(descriptor);
						}
						MEASURE convertedEntry = measuresByName.get(contentsList.get(j));
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

