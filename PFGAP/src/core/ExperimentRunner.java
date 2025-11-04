package core;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

//import datasets.ListDataset;
import datasets.ListObjectDataset;
import imputation.MissingIndicesBuilder;
import org.apache.commons.lang3.ArrayUtils;
import proximities.DTWPFImpute;
import proximities.OutlierScorer;
import proximities.PFImpute;
import trees.ProximityForest;
import util.GeneralUtilities;
import util.PrintUtilities;

import static application.PFApplication.UCR_dataset;
import static proximities.PFGAP.computeTestTrainProximities;
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

		if(!eval) {

			ListObjectDataset train_data_original;
			//ListDataset test_data_original = null; //we will overwrite this.
			ListObjectDataset test_data_original = null; //we will overwrite this.
			if (AppContext.testing_file != null) {
			/*test_data_original =
					CSVReader.readCSVToListDataset(AppContext.testing_file, AppContext.csv_has_header,
							AppContext.target_column_is_first, csvSeparatpr);*/
				//TODO: make testing_labels optional.
				test_data_original =
						DelimitedFileReader.readToListObjectDataset(
								AppContext.testing_file,
								AppContext.testing_labels,
								AppContext.entry_separator,
								AppContext.array_separator,
								AppContext.csv_has_header,
								AppContext.is2D,
								AppContext.isNumeric,
								AppContext.hasMissingValues,
								AppContext.target_column_is_first,
								true,
								AppContext.isRegression);
			}
			//ListDataset test_data_original =
			//		CSVReader.readCSVToListDataset(AppContext.testing_file, AppContext.csv_has_header,
			//AppContext.target_column_is_first, csvSeparatpr);


			/*train_data_original =
					CSVReader.readCSVToListDataset(AppContext.training_file, AppContext.csv_has_header,
							AppContext.target_column_is_first, csvSeparatpr);*/
			train_data_original =
					DelimitedFileReader.readToListObjectDataset(
							AppContext.training_file,
							AppContext.training_labels,
							AppContext.entry_separator,
							AppContext.array_separator,
							AppContext.csv_has_header,
							AppContext.is2D,
							AppContext.isNumeric,
							AppContext.hasMissingValues,
							AppContext.target_column_is_first,
							false,
							AppContext.isRegression);
			//}
			//else{
			//	train_data_original = test_data_original;
			//}


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
			//if (!eval) {
			if (!AppContext.isRegression) {
				train_data = train_data_original.reorder_class_labels(null);
			} else {
				train_data = train_data_original; //do we need to make a deep copy?
			}
			//train_data.setLength(train_data_original.length());
			if (AppContext.hasMissingValues) {
				train_data.setMissingIndices(MissingIndicesBuilder.buildFromDataset(train_data.getData()));
			}

			if (AppContext.testing_file != null) {
				if (!AppContext.isRegression) {
					test_data = test_data_original.reorder_class_labels(train_data._get_initial_class_labels());
				} else {
					test_data = test_data_original; // again do we need to make a copy?
				}
				//test_data.setLength(test_data_original.length());
				if (AppContext.hasMissingValues) {
					test_data.setMissingIndices(MissingIndicesBuilder.buildFromDataset(test_data.getData()));
				}
			}
		/*} else {
			ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(AppContext.modelname+ ".ser"));
			ProximityForest forest1 = (ProximityForest) objectInputStream.readObject(); //AppContext.userdistances
			Map<Object, Integer> labelMap = (Map<Object, Integer>) objectInputStream.readObject();
			if (!AppContext.isRegression) {
				test_data = test_data_original.reorder_class_labels(labelMap);
			} else {
				test_data = test_data_original; // again do we need to make a copy?
			}
			//test_data.setLength(test_data_original.length());
			if (AppContext.hasMissingValues) {
				test_data.setMissingIndices(MissingIndicesBuilder.buildFromDataset(test_data.getData()));
			}
		}*/


			AppContext.setTraining_data(train_data);
			AppContext.setTesting_data(test_data);

			//allow garbage collector to reclaim this memory, since we have made copies with reordered class labels
			train_data_original = null;
			test_data_original = null;
			System.gc();

			//setup environment
			File training_file = new File(AppContext.training_file);
			String datasetName = training_file.getName().replaceAll("_TRAIN.txt", "");    //this is just some quick fix for UCR datasets
			AppContext.setDatasetName(datasetName);

			//Is this really important?
			//if(!eval) {
			//	PrintUtilities.printConfiguration();
			//}

			System.out.println();

			//if we need to shuffle
			if (AppContext.shuffle_dataset) {
				System.out.println("Shuffling the training set...");
				train_data.shuffle();
			}


			for (int i = 0; i < AppContext.num_repeats; i++) {

				// This is training the model.
				//if(!eval) {
				if (AppContext.verbosity > 0) {
					System.out.println("-----------------Repetition No: " + (i + 1) + " (" + datasetName + ") " + "  -----------------");
					PrintUtilities.printMemoryUsage();
				} else if (AppContext.verbosity == 0 && i == 0) {
					System.out.println("Repetition, Dataset, Score, TrainingTime(ms), TestingTime(ms), MeanDepthPerTree");
				}

				//if(!eval) {
				//create model
				//ProximityForest forest = new ProximityForest(i,AppContext.userdistances);

				if (AppContext.hasMissingValues && AppContext.isNumeric && !AppContext.DTWImpute) {
					System.out.println("Imputing the training set...");
					//first, do the mean impute. Later, we'll let users select which imputer to use.
					AppContext.initial_imputer.Impute(train_data);
					for (int j = 0; j < AppContext.numImputes; j++) {
						//do the PF update (PFImpute is NOT an actual imputer, but an updater.)
						ProximityForest forest = new ProximityForest(i, AppContext.userdistances);
						forest.train(train_data);
						computeTrainProximities(forest, train_data);
						PFImpute.trainNumericImpute(train_data);
						//forest = null;
						//System.gc();
					}
					System.out.println("Done imputing the training set.");
				}

				if (AppContext.hasMissingValues && AppContext.isNumeric && AppContext.DTWImpute) {
					System.out.println("Imputing the training set...");
					//first, do the mean impute. Later, we'll let users select which imputer to use.
					AppContext.initial_imputer.Impute(train_data);
					for (int j = 0; j < AppContext.numImputes; j++) {
						//do the PF update (PFImpute is NOT an actual imputer, but an updater.)
						ProximityForest forest = new ProximityForest(i, AppContext.userdistances);
						forest.train(train_data);
						computeTrainProximities(forest, train_data);
						DTWPFImpute.buildAlignmentPathCache(train_data, train_data, AppContext.training_proximities_sparse, AppContext.is2D, -1);
						DTWPFImpute.trainNumericImpute(train_data);
						//forest = null;
						//System.gc();
					}
					System.out.println("Done imputing the training set.");
				}

				//train model
				ProximityForest forest = new ProximityForest(i, AppContext.userdistances);
				forest.train(train_data);

				if (AppContext.savemodel) {
					// save the trained model
					try {
						/*FileOutputStream fileOutputStream = new FileOutputStream(AppContext.output_dir + AppContext.modelname + ".ser");
						ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
						objectOutputStream.writeObject(forest);
						objectOutputStream.writeObject(train_data);
						//objectOutputStream.writeObject(train_data._get_initial_class_labels());
						objectOutputStream.close();*/
						AppContextSnapshot snapshot = AppContextUtils.captureSnapshot();
						ModelIO.saveModel(AppContext.output_dir + AppContext.modelname + ".ser", forest, train_data, snapshot);
					} catch (IOException e) {
						//	e.printStackTrace
					}
				}

				if (AppContext.impute_train) {
					GeneralUtilities.writeDelimitedData(
							train_data.getData(),
							AppContext.output_dir + AppContext.training_file,
							AppContext.array_separator,
							AppContext.entry_separator
					);
				}


				//test model if needed
				if (AppContext.testing_file != null) {


					// impute the test set, if needed.
					if (AppContext.hasMissingValues && AppContext.isNumeric && !AppContext.DTWImpute) {
						//first, do the mean impute. Later, we'll let users select which imputer to use.
						System.out.println("Performing initial imputation...");
						AppContext.initial_imputer.Impute(test_data);
						for (int j = 0; j < AppContext.numImputes; j++){
							//do the PF update (PFImpute is NOT an actual imputer, but an updater.)
							System.out.println("Updating missing values...");
							computeTestTrainProximities(forest, test_data, train_data); //what is train_data??
							PFImpute.testNumericImpute(test_data, train_data); //again, is train_data defined??
						}
					}

					// new DTW-based imputer
					if (AppContext.hasMissingValues && AppContext.isNumeric && AppContext.DTWImpute) {
						//first, do the mean impute. Later, we'll let users select which imputer to use.
						System.out.println("Imputing the test set...");
						AppContext.initial_imputer.Impute(test_data);
						for (int j = 0; j < AppContext.numImputes; j++){
							//do the PF update (PFImpute is NOT an actual imputer, but an updater.)
							computeTestTrainProximities(forest, test_data, train_data); //what is train_data??
							DTWPFImpute.buildAlignmentPathCache(test_data, train_data, AppContext.testing_training_proximities_sparse, AppContext.is2D, -1);
							DTWPFImpute.testNumericImpute(test_data, train_data);

						}
					}

					// Might this be too soon?
					if (AppContext.impute_test) {
						GeneralUtilities.writeDelimitedData(
								test_data.getData(),
								AppContext.output_dir + AppContext.testing_file,
								AppContext.array_separator,
								AppContext.entry_separator
						);
					}


					ProximityForestResult result = forest.test(test_data);

					//Now we print the Predictions array to a text file.
					if (AppContext.get_predictions) {
						if (!AppContext.isRegression) {
							//List<Object> predictedLabels = test_data._internal_class_list(); // reordered labels for classification
							List<Object> predictedLabels = result.Predictions;
							Map<Integer, Object> newToOriginal = test_data.invertLabelMap(test_data._get_initial_class_labels());
							List<Object> originalPredictions = predictedLabels.stream()
									.map(newToOriginal::get)
									.collect(Collectors.toList());
							PrintWriter writer0 = new PrintWriter(AppContext.output_dir + "Validation_Predictions.txt", StandardCharsets.UTF_8);
							//writer0.print(ArrayUtils.toString(result.Predictions));
							writer0.print(ArrayUtils.toString(originalPredictions));
							writer0.close();
						} else {
							//int t = forest.getResultSet();
							PrintWriter writer0 = new PrintWriter(AppContext.output_dir + "Validation_Predictions.txt", StandardCharsets.UTF_8);
							writer0.print(ArrayUtils.toString(result.Predictions));
							//writer0.print(ArrayUtils.toString(predictedLabels));
							writer0.close();
						}
					}

					//print and export resultS
					//TODO: fix the buggy results for imputation.
					if (!AppContext.hasMissingValues) {
						result.printResults(datasetName, i, "");
					}
					//AppContext.output_dir = null;

					//if (!AppContext.getprox) {
					//	test_data = null; // erase the test data information (or after test/train prox)
					//}

				}

				// what if they want outlier scores? must be a classification problem.
				if (AppContext.get_training_outlier_scores && AppContext.getprox && !AppContext.isRegression) {
					// in this case, we use a dense representation of the proximities.
					AppContext.useSparseProximities = false;
					System.out.println("Computing Training Proximities...");
					computeTrainProximities(forest, train_data);
					System.out.println("Computing Training Outlier Scores...");
					double[] scores = OutlierScorer.getOutlierScores(
							AppContext.useSparseProximities,
							false, // symmetrize??
							true, // parallelize
							train_data._internal_class_array(),
							AppContext.training_proximities,
							AppContext.training_proximities_sparse
					);

					PrintWriter writer2 = new PrintWriter(AppContext.output_dir + "outlier_scores.txt", StandardCharsets.UTF_8);
					writer2.print(ArrayUtils.toString(scores));
					writer2.close();

				}

				if (AppContext.get_training_outlier_scores && !AppContext.getprox && !AppContext.isRegression) {
					// In this case we can use a sparse representation, since we won't need to return the proximities.
					AppContext.useSparseProximities = false;
					System.out.println("Computing Training Proximities...");
					computeTrainProximities(forest, train_data);
					// and now the outlier scores
					System.out.println("Computing Training Outlier Scores...");
					double[] scores = OutlierScorer.getOutlierScores(
							AppContext.useSparseProximities,
							false, // symmetrize??
							true, // parallelize
							train_data._internal_class_array(),
							AppContext.training_proximities,
							AppContext.training_proximities_sparse
					);

					PrintWriter writer2 = new PrintWriter(AppContext.output_dir + "outlier_scores.txt", StandardCharsets.UTF_8);
					writer2.print(ArrayUtils.toString(scores));
					writer2.close();

				}

				if (AppContext.getprox) {
					//Calculate array of forest proximities.
					if (!AppContext.get_training_outlier_scores) {
						// if a user already wants the scores, the proximities have already been computed.
						AppContext.useSparseProximities = false;
						System.out.println("Computing Training Proximities...");
						double t5 = System.currentTimeMillis();
						computeTrainProximities(forest, train_data);
						double t6 = System.currentTimeMillis();
						System.out.print("Done Computing Training Proximities. ");
						System.out.print("Computation time: ");
						System.out.println(t6 - t5 + "ms");
					}


					//Now we print the PFGAP array to a text file.
					PrintWriter writer = new PrintWriter(AppContext.output_dir + "TrainingProximities.txt", StandardCharsets.UTF_8);
					//writer.print(ArrayUtils.toString(PFGAP));
					writer.print(ArrayUtils.toString(AppContext.training_proximities));
					writer.close();
					//Integer[] ytrain = new Integer[train_data.size()];
					//for (Integer k = 0; k < train_data.size(); k++) {
					//	ytrain[k] = train_data.get_class(k);
					//}
					//PrintWriter writer2 = new PrintWriter(AppContext.output_dir + "ytrain.txt", "UTF-8");
					//writer2.print(ArrayUtils.toString(ytrain));
					//writer2.close();

					if (test_data != null) {
						System.out.println("Computing Test/Train Proximities...");
						double t7 = System.currentTimeMillis();
						computeTestTrainProximities(forest, test_data, train_data);
						double t8 = System.currentTimeMillis();
						System.out.print("Done Computing Test/Train Proximities. ");
						System.out.print("Computation time: ");
						System.out.println(t8 - t7 + "ms");


						//Now we print the PFGAP array to a text file.
						PrintWriter writer3 = new PrintWriter(AppContext.output_dir + "TestTrainProximities.txt", StandardCharsets.UTF_8);
						//writer.print(ArrayUtils.toString(PFGAP));
						writer3.print(ArrayUtils.toString(AppContext.testing_training_proximities));
						writer3.close();
						//Integer[] ytest = new Integer[test_data.size()];
						//for (int k = 0; k < test_data.size(); k++) {
						//	ytest[k] = test_data.get_class(k);
						//}
						//PrintWriter writer4 = new PrintWriter(AppContext.output_dir + "ytest.txt", StandardCharsets.UTF_8);
						//writer4.print(ArrayUtils.toString(ytest));
						//writer4.close();
						//test_data = null;
					}
				}
				test_data = null;
				////print and export resultS
				//result.printResults(datasetName, i, "");
				//AppContext.output_dir = null;

				//export level is integer because I intend to add few levels in future, each level with a higher verbosity
				/*if (AppContext.export_level > 0) {
					result.exportJSON(datasetName, i);
				}*/
			}
			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}
		} else{
				//evaluate saved model??
				//ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(AppContext.modelname+ ".ser"));
				//ProximityForest forest1 = (ProximityForest) objectInputStream.readObject(); //AppContext.userdistances
				//Map<Object, Integer> labelMap = (Map<Object, Integer>) objectInputStream.readObject();
				//forest1.predict(test_data);
			/*ArrayList<Integer> Predictions_saved = new ArrayList<>();
			for (int k=0; k < test_data.size(); k++){
				Predictions_saved.add(forest1.predict(test_data.get_series(k)));
			}*/

			/*ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(AppContext.modelname+ ".ser"));
			ProximityForest forest1 = (ProximityForest) objectInputStream.readObject(); //AppContext.userdistances
			ListObjectDataset train_data1 = (ListObjectDataset) objectInputStream.readObject();*/
			//Map<Object, Integer> labelMap = (Map<Object, Integer>) objectInputStream.readObject();
			ModelIO.LoadedModel loaded = ModelIO.loadModel(AppContext.modelname + ".ser");
			ModelIO.applySnapshot(loaded.snapshot);
			ProximityForest forest1 = loaded.forest;
			ListObjectDataset train_data = loaded.trainData;

			ListObjectDataset test_data_original;
			test_data_original =
				DelimitedFileReader.readToListObjectDataset(
						AppContext.testing_file,
						AppContext.testing_labels,
						AppContext.entry_separator,
						AppContext.array_separator,
						AppContext.csv_has_header,
						AppContext.is2D,
						AppContext.isNumeric,
						AppContext.hasMissingValues,
						AppContext.target_column_is_first,
						true,
						AppContext.isRegression);

			if (!AppContext.isRegression) {
				//test_data = test_data_original.reorder_class_labels(labelMap);
				test_data = test_data_original.reorder_class_labels(train_data._get_initial_class_labels());
			} else {
				test_data = test_data_original; // again do we need to make a copy?
			}
			//test_data.setLength(test_data_original.length());
			if (AppContext.hasMissingValues) {
				test_data.setMissingIndices(MissingIndicesBuilder.buildFromDataset(test_data.getData()));
			}

			AppContext.setTesting_data(test_data);

			//allow garbage collector to reclaim this memory, since we have made copies with reordered class labels
			test_data_original = null;
			System.gc();

			//setup environment
			File training_file = new File(AppContext.training_file);
			String datasetName = training_file.getName().replaceAll("_TRAIN.txt", "");    //this is just some quick fix for UCR datasets
			AppContext.setDatasetName(datasetName);

			for (int i = 0; i < AppContext.num_repeats; i++) {

				//Perform imputation, if needed.
				if (AppContext.hasMissingValues && AppContext.isNumeric && !AppContext.DTWImpute) {
					//first, do the mean impute. Later, we'll let users select which imputer to use.
					System.out.println("Performing initial imputation...");
					AppContext.initial_imputer.Impute(test_data);
					for (int j = 0; j < AppContext.numImputes; j++){
						//do the PF update (PFImpute is NOT an actual imputer, but an updater.)
						System.out.println("Updating missing values...");
						computeTestTrainProximities(forest1, test_data, train_data); //what is train_data??
						PFImpute.testNumericImpute(test_data, train_data); //again, is train_data defined??
					}
				}

				// new DTW-based imputer
				if (AppContext.hasMissingValues && AppContext.isNumeric && AppContext.DTWImpute) {
					//first, do the mean impute. Later, we'll let users select which imputer to use.
					System.out.println("Performing initial imputation...");
					AppContext.initial_imputer.Impute(test_data);
					for (int j = 0; j < AppContext.numImputes; j++){
						//do the PF update (PFImpute is NOT an actual imputer, but an updater.)
						System.out.println("Updating missing values...");
						computeTestTrainProximities(forest1, test_data, train_data); //what is train_data??
						DTWPFImpute.buildAlignmentPathCache(test_data, train_data, AppContext.testing_training_proximities_sparse, AppContext.is2D, -1);
						DTWPFImpute.testNumericImpute(test_data, train_data);

					}
				}

				ProximityForestResult result1 = forest1.test(test_data);
				//TODO: fix the buggy scores for imputation.
				if (!AppContext.hasMissingValues) {
					result1.printResults(datasetName, i, "");
				}

				if (AppContext.impute_test) {
					GeneralUtilities.writeDelimitedData(
							test_data.getData(),
							AppContext.output_dir + AppContext.testing_file,
							AppContext.array_separator,
							AppContext.entry_separator
					);
				}

				if (AppContext.getprox) {
					AppContext.useSparseProximities = false;
					System.out.println("Computing Test/Train Proximities...");
					double t7 = System.currentTimeMillis();
					computeTestTrainProximities(forest1, test_data, train_data);
					double t8 = System.currentTimeMillis();
					System.out.print("Done Computing Test/Train Proximities. ");
					System.out.print("Computation time: ");
					System.out.println(t8 - t7 + "ms");


					//Now we print the PFGAP array to a text file.
					PrintWriter writer3 = new PrintWriter(AppContext.output_dir + "TestTrainProximities.txt", StandardCharsets.UTF_8);
					//writer.print(ArrayUtils.toString(PFGAP));
					writer3.print(ArrayUtils.toString(AppContext.testing_training_proximities));
					writer3.close();
					//Integer[] ytest = new Integer[test_data.size()];
					//for (int k = 0; k < test_data.size(); k++) {
					//	ytest[k] = test_data.get_class(k);
					//}
					//PrintWriter writer4 = new PrintWriter(AppContext.output_dir + "ytest.txt", StandardCharsets.UTF_8);
					//writer4.print(ArrayUtils.toString(ytest));
					//writer4.close();
				}

				//Now we print the Predictions array of the saved model to a text file.
				if (AppContext.get_predictions) {
					if (!AppContext.isRegression) {
						//List<Object> predictedLabels = test_data._internal_class_list(); // reordered labels
						List<Object> predictedLabels = result1.Predictions;
						Map<Integer, Object> newToOriginal = test_data.invertLabelMap(test_data._get_initial_class_labels());
						List<Object> originalPredictions = predictedLabels.stream()
								.map(newToOriginal::get)
								.collect(Collectors.toList());
						PrintWriter writer0a = new PrintWriter(AppContext.output_dir + "Predictions_saved.txt", StandardCharsets.UTF_8);
						//writer0a.print(ArrayUtils.toString(Predictions_saved));
						//TODO: output the predictions in terms of the original classes.
						//writer0a.print(ArrayUtils.toString(result1.Predictions));
						writer0a.print(ArrayUtils.toString(originalPredictions));
						writer0a.close();
					} else {
						//System.out.println("Printing results...");
						//System.out.println(result1.Predictions);
						PrintWriter writer0a = new PrintWriter(AppContext.output_dir + "Predictions_saved.txt", StandardCharsets.UTF_8);
						writer0a.print(ArrayUtils.toString(result1.Predictions));
						writer0a.close();
					}
				}

				test_data = null; // erase test data.

			}

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}

		}

	}

}
