package core;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import datasets.ListDataset;
import org.apache.commons.lang3.ArrayUtils;
import trees.ProximityForest;
import util.PrintUtilities;

import static application.PFApplication.UCR_dataset;
import static core.PFGAP.ForestProximity;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ExperimentRunner {

	ListDataset train_data;
	ListDataset test_data;
	private static String csvSeparatpr = "\t"; //for tsv files.

	public ExperimentRunner(){

	}

	public void run(boolean eval) throws Exception {
		//read data files
		//we assume no header in the csv files, and that class label is in the first column, modify if necessary
		ListDataset train_data_original;
		ListDataset test_data_original =
				CSVReader.readCSVToListDataset(AppContext.testing_file, AppContext.csv_has_header,
						AppContext.target_column_is_first, csvSeparatpr);
		if(!eval) {
			train_data_original =
					CSVReader.readCSVToListDataset(AppContext.training_file, AppContext.csv_has_header,
							AppContext.target_column_is_first, csvSeparatpr);
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
		test_data = test_data_original.reorder_class_labels(train_data._get_initial_class_labels());


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

			if(!eval) {
				if (AppContext.verbosity > 0) {
					System.out.println("-----------------Repetition No: " + (i + 1) + " (" + datasetName + ") " + "  -----------------");
					PrintUtilities.printMemoryUsage();
				}else if (AppContext.verbosity == 0 && i == 0) {
					System.out.println("Repetition, Dataset, Accuracy, TrainingTime(ms), TestingTime(ms), MeanDepthPerTree");
				}

			//if(!eval) {
				//create model
				ProximityForest forest = new ProximityForest(i,AppContext.userdistances);

				//train model
				forest.train(train_data);

				if(AppContext.savemodel) {
					// save the trained model
					try {
						FileOutputStream fileOutputStream = new FileOutputStream(AppContext.modelname + ".ser");
						ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
						objectOutputStream.writeObject(forest);
						objectOutputStream.close();
					} catch (IOException e) {
						//	e.printStackTrace
					}
				}

				//test model
				ProximityForestResult result = forest.test(test_data);

				//Now we print the Predictions array to a text file.
				PrintWriter writer0 = new PrintWriter("Predictions.txt", "UTF-8");
				writer0.print(ArrayUtils.toString(result.Predictions));
				writer0.close();

				if(AppContext.getprox) {
					//Calculate array of forest proximities.
					System.out.println("Computing Forest Proximities...");
					double t5 = System.currentTimeMillis();
					Double[][] PFGAP = new Double[train_data.size()][train_data.size()];
					if(AppContext.parallelProx){
						ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

						List<Future<?>> futures = new ArrayList<>();

						for (int k = 0; k < train_data.size(); k++) {
							final int finalK = k;
							futures.add(executor.submit(() -> {
								for (int j = 0; j < train_data.size(); j++) {
									double prox = ForestProximity(finalK, j, forest);
									PFGAP[finalK][j] = prox;
								}
							}));
						}

						// Wait for all tasks to complete
						for (Future<?> future : futures) {
							future.get(); // Handle exceptions as needed
						}

						executor.shutdown();

					} else{
						for (Integer k = 0; k < train_data.size(); k++) {
							for (Integer j = 0; j < train_data.size(); j++) {
								Double prox = ForestProximity(k, j, forest);
								PFGAP[k][j] = prox;
							}
						}
					}

					double t6 = System.currentTimeMillis();
					System.out.print("Done Computing Forest Proximities. ");
					System.out.print("Computation time: ");
					System.out.println(t6 - t5 + "ms");


					//Now we print the PFGAP array to a text file.
					PrintWriter writer = new PrintWriter("ForestProximities.txt", "UTF-8");
					writer.print(ArrayUtils.toString(PFGAP));
					writer.close();
					Integer[] ytrain = new Integer[train_data.size()];
					for (Integer k = 0; k < train_data.size(); k++) {
						ytrain[k] = train_data.get_class(k);
					}
					PrintWriter writer2 = new PrintWriter("ytrain.txt", "UTF-8");
					writer2.print(ArrayUtils.toString(ytrain));
					writer2.close();
				}
				//print and export resultS
				result.printResults(datasetName, i, "");

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
				//Now we print the Predictions array of the saved model to a text file.
				PrintWriter writer0a = new PrintWriter("Predictions_saved.txt", "UTF-8");
				//writer0a.print(ArrayUtils.toString(Predictions_saved));
				writer0a.print(ArrayUtils.toString(result1.Predictions));
				writer0a.close();

			}

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}

		}

	}

}
