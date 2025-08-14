package application;

import core.AppContext;
import core.ExperimentRunner;
import core.ProximityForestResult;
import distance.elastic.MEASURE;
import trees.ProximityForest;
import util.GeneralUtilities;
import util.PrintUtilities;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Main entry point for the Proximity Forest application
 *
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class PFApplicationEval {

    public static final String UCR_dataset = "GunPoint"; //"ItalyPowerDemand";
    //TODO test support file paths with a space?
    public static final String[] test_args = new String[]{
            "-train=" + System.getProperty("user.dir") + "/Data/" + UCR_dataset + "_TRAIN.tsv", //"-train=E:/data/ucr/" + UCR_dataset + "/" + UCR_dataset + "_TRAIN.txt",
            "-test=" + System.getProperty("user.dir") + "/Data/" + UCR_dataset + "_TEST.tsv",
//			"-train=E:/data/satellite/sample100000_TRAIN.txt",
//			"-test=E:/data/satellite/sample100000_TEST.txt",
            "-out=output",
            //"-repeats=1",
            //"-trees=10",
            //"-r=5",
            //"-on_tree=true",
            //"-shuffle=true",
//			"-jvmwarmup=true",	//disabled
            "-export=1",
            "-verbosity=1",
            "-csv_has_header=false",
            "-target_column=first"	//first or last
    };

    public static void main(String[] args) {
        try {

            //args = test_args;
            //Integer testint = Integer.parseInt("2 3 3.444"[0]);
            //some default settings are specified in the AppContext class but here we
            //override the default settings using the provided command line arguments
            //System.out.println(args);
            for (int i = 0; i < args.length; i++) {
                String[] options = args[i].trim().split("=");

                switch(options[0]) {
                    case "-train":
                        AppContext.training_file = options[1];
                        //AppContext.testing_file = options[1]; //not used.
                        break;
                    case "-test":
                        AppContext.testing_file = options[1];
                        break;
                    case "-out":
                        AppContext.output_dir = options[1];
                        break;
                    /*case "-repeats":
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
                        break;*/
                    case "-shuffle":
                        //Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir ok"});
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
                    case "-modelname":
                        AppContext.modelname = options[1];
                        break;
                    case "-distances":
                        //Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir ok"});
                        /*String temp = options[1];
                        //Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir testdir"});
                        String temp_rm = temp.substring(1, temp.length() - 1); // Removes '[' and ']'
                        String[] contents = temp_rm.split(","); // Splits by ","
                        List<String> contentsList = Arrays.asList(contents);
                        int numberofdists = contentsList.size();
                        MEASURE[] toadd = new MEASURE[numberofdists];

                        Map<String, MEASURE> measuresByName = new HashMap<>();

                        // Associate string keys with MEASURE values
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

                        for (int j=0; j < numberofdists; j++){
                            MEASURE convertedEntry = measuresByName.get(contentsList.get(j));
                            toadd[j] = convertedEntry;
                        }

                        AppContext.userdistances = toadd;
                        */
                        break;
                    default:
                        //Runtime.getRuntime().exec(new String[]{"/bin/bash", "-c", "mkdir ohno"});
                        throw new Exception("Invalid Commandline Arguments");
                }
            }

            if (AppContext.warmup_java) {
                GeneralUtilities.warmUpJavaRuntime();
            }

            ExperimentRunner experiment = new ExperimentRunner();
            experiment.run(true);


        }catch(Exception e) {
            PrintUtilities.abort(e);
        }

    }


}

