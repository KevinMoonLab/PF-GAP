package core;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

//import datasets.ListDataset;
import datasets.ListObjectDataset;
import imputation.MeanImpute;
import imputation.MissingIndicesBuilder;
import org.apache.commons.lang3.ArrayUtils;
import proximities.PFImpute;
import trees.ProximityForest;
import util.GeneralUtilities;
import util.PrintUtilities;

import static application.PFApplication.UCR_dataset;
import static proximities.PFGAP.computeTrainProximities;


/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ExperimentRunner {

	//ListDataset train_data;
	ListObjectDataset train_data;
	//ListDataset test_data;
	ListObjectDataset test_data;
	private static String csvSeparatpr = "\t"; //for tsv files.

	public ExperimentRunner(){

	}

	public void run(boolean eval) throws Exception {
		//read data files
		//we assume no header in the csv files, and that class label is in the first column, modify if necessary
		//ListDataset train_data_original;
		ListObjectDataset train_data_original;
		//ListDataset test_data_original = null; //we will overwrite this.
		ListObjectDataset test_data_original = null; //we will overwrite this.
		if(AppContext.testing_file != null){
			/*test_data_original =
					CSVReader.readCSVToListDataset(AppContext.testing_file, AppContext.csv_has_header,
							AppContext.target_column_is_first, csvSeparatpr);*/
			test_data_original =
					DelimitedFileReader.readToListObjectDataset(
							AppContext.testing_file,
							AppContext.testing_labels,
							AppContext.firstSeparator,
							AppContext.secondSeparator,
							AppContext.csv_has_header,
							AppContext.is2D,
							AppContext.isNumeric,
							AppContext.hasMissingValues,
							AppContext.target_column_is_first);
		}
		//ListDataset test_data_original =
		//		CSVReader.readCSVToListDataset(AppContext.testing_file, AppContext.csv_has_header,
		//				AppContext.target_column_is_first, csvSeparatpr);
		if(!eval) {
			/*train_data_original =
					CSVReader.readCSVToListDataset(AppContext.training_file, AppContext.csv_has_header,
							AppContext.target_column_is_first, csvSeparatpr);*/
			train_data_original =
					DelimitedFileReader.readToListObjectDataset(
							AppContext.training_file,
							AppContext.training_labels,
							AppContext.firstSeparator,
							AppContext.secondSeparator,
							AppContext.csv_has_header,
							AppContext.is2D,
							AppContext.isNumeric,
							AppContext.hasMissingValues,
							AppContext.target_column_is_first);
		}
		else{
			train_data_original = test_data_original;
		}



		/**
		 * We do some reordering of class labels in this implementation,
		 * this is not necessary if HashMaps are used in some places in the algorithm,
		 * but since we used an array in cases where we need HashMaps to store class distributions maps,
		 * I had to to keep class labels contiguous.
		 *
		 * I intend to change this later, and use a library like Trove, Colt or FastUtil which implements primitive HashMaps
		 * After thats done, we will not be reordering class here.
		 *
		 */
		train_data = train_data_original.reorder_class_labels(null);
		//train_data.setLength(train_data_original.length());
		if (AppContext.hasMissingValues){
			train_data.setMissingIndices(MissingIndicesBuilder.buildFromDataset(train_data.getData()));
		}

		if(AppContext.testing_file != null) {
			test_data = test_data_original.reorder_class_labels(train_data._get_initial_class_labels());
			//test_data.setLength(test_data_original.length());
			if (AppContext.hasMissingValues){
				test_data.setMissingIndices(MissingIndicesBuilder.buildFromDataset(test_data.getData()));
			}
		}


		AppContext.setTraining_data(train_data);
		AppContext.setTesting_data(test_data);

		//allow garbage collector to reclaim this memory, since we have made copies with reordered class labels
		train_data_original = null;
		test_data_original = null;
		System.gc();

		//setup environment
		File training_file = new File(AppContext.training_file);
		String datasetName = training_file.getName().replaceAll("_TRAIN.txt", "");	//this is just some quick fix for UCR datasets
		AppContext.setDatasetName(datasetName);

		if(!eval) {
			PrintUtilities.printConfiguration();
		}

		System.out.println();

		//if we need to shuffle
		if (AppContext.shuffle_dataset) {
			System.out.println("Shuffling the training set...");
			train_data.shuffle();
		}


		for (int i = 0; i < AppContext.num_repeats; i++) {

			// This is training the model.
			if(!eval) {
				if (AppContext.verbosity > 0) {
					System.out.println("-----------------Repetition No: " + (i + 1) + " (" + datasetName + ") " + "  -----------------");
					PrintUtilities.printMemoryUsage();
				}else if (AppContext.verbosity == 0 && i == 0) {
					System.out.println("Repetition, Dataset, Accuracy, TrainingTime(ms), TestingTime(ms), MeanDepthPerTree");
				}

			//if(!eval) {
				//create model
				//ProximityForest forest = new ProximityForest(i,AppContext.userdistances);

				if (AppContext.hasMissingValues && AppContext.isNumeric) {
					//first, do the mean impute. Later, we'll let users select which imputer to use.
					System.out.println("Performing initial imputation...");
					MeanImpute.Impute(train_data);
					for (int j = 0; j < AppContext.numImputes; j++){
						//do the PF update (PFImpute is NOT an actual imputer, but an updater.)
						ProximityForest forest = new ProximityForest(i,AppContext.userdistances);
						forest.train(train_data);
						System.out.println("Updating missing values...");
						computeTrainProximities(forest, train_data);
						PFImpute.trainNumericImpute(train_data);
						//forest = null;
						//System.gc();
					}
				}

				//train model
				ProximityForest forest = new ProximityForest(i,AppContext.userdistances);
				forest.train(train_data);

				if(AppContext.savemodel) {
					// save the trained model
					try {
						FileOutputStream fileOutputStream = new FileOutputStream(AppContext.output_dir + AppContext.modelname + ".ser");
						ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
						objectOutputStream.writeObject(forest);
						objectOutputStream.close();
					} catch (IOException e) {
						//	e.printStackTrace
					}
				}

				if (AppContext.impute_train) {
					GeneralUtilities.writeDelimitedData(
							train_data.getData(),
							AppContext.output_dir + AppContext.training_file,
							AppContext.firstSeparator,
							AppContext.secondSeparator
					);
				}

				//test model if needed
				if(AppContext.testing_file != null) {
					ProximityForestResult result = forest.test(test_data);
					test_data = null; // erase the test data information.

					//Now we print the Predictions array to a text file.
					PrintWriter writer0 = new PrintWriter(AppContext.output_dir + "Predictions.txt", "UTF-8");
					//TODO: output the predictions in terms of the original classes.
					writer0.print(ArrayUtils.toString(result.Predictions));
					writer0.close();

					//print and export resultS
					result.printResults(datasetName, i, "");
					//AppContext.output_dir = null;
				}

				if(AppContext.getprox) {
					//Calculate array of forest proximities.
					AppContext.useSparseProximities = false;
					System.out.println("Computing Forest Proximities...");
					double t5 = System.currentTimeMillis();
					computeTrainProximities(forest, train_data);
					double t6 = System.currentTimeMillis();
					System.out.print("Done Computing Forest Proximities. ");
					System.out.print("Computation time: ");
					System.out.println(t6 - t5 + "ms");


					//Now we print the PFGAP array to a text file.
					PrintWriter writer = new PrintWriter(AppContext.output_dir + "ForestProximities.txt", "UTF-8");
					//writer.print(ArrayUtils.toString(PFGAP));
					writer.print(ArrayUtils.toString(AppContext.training_proximities));
					writer.close();
					Integer[] ytrain = new Integer[train_data.size()];
					for (Integer k = 0; k < train_data.size(); k++) {
						ytrain[k] = train_data.get_class(k);
					}
					PrintWriter writer2 = new PrintWriter(AppContext.output_dir + "ytrain.txt", "UTF-8");
					writer2.print(ArrayUtils.toString(ytrain));
					writer2.close();
				}
				////print and export resultS
				//result.printResults(datasetName, i, "");
				//AppContext.output_dir = null;

				//export level is integer because I intend to add few levels in future, each level with a higher verbosity
				/*if (AppContext.export_level > 0) {
					result.exportJSON(datasetName, i);
				}*/
			}
			else{
				//evaluate saved model??
				ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(AppContext.modelname+ ".ser"));
				ProximityForest forest1 = (ProximityForest) objectInputStream.readObject(); //AppContext.userdistances
				//forest1.predict(test_data);
			/*ArrayList<Integer> Predictions_saved = new ArrayList<>();
			for (int k=0; k < test_data.size(); k++){
				Predictions_saved.add(forest1.predict(test_data.get_series(k)));
			}*/
				ProximityForestResult result1 = forest1.test(test_data);

				//TODO: calculate proximities for the test points.
				test_data = null; // erase test data.

				//Now we print the Predictions array of the saved model to a text file.
				PrintWriter writer0a = new PrintWriter(AppContext.output_dir + "Predictions_saved.txt", "UTF-8");
				//writer0a.print(ArrayUtils.toString(Predictions_saved));
				//TODO: output the predictions in terms of the original classes.
				writer0a.print(ArrayUtils.toString(result1.Predictions));
				writer0a.close();

			}

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}

		}

	}

}
