package core;

import datasets.readers.lazy.LazySeriesRef;
import datasets.ListObjectDataset;
import datasets.readers.*;
import imputation.util.MissingIndicesBuilder;
import imputation.ProximityImputation;
// import org.apache.commons.lang3.ArrayUtils;
import outlier.IsolationDepthScorer;
import outlier.OutlierScorer;
import output.*;
import preprocessing.standardization.*;
import trees.ProximityForest;
import util.GeneralUtilities;
import util.PrintUtilities;

import java.io.File;
import java.io.IOException;
// import java.io.PrintWriter;
// import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
// import java.util.stream.Collectors;

import static proximities.PFGAP.computeTestTrainProximities;
import static proximities.PFGAP.computeTrainProximities;

public class ExperimentRunner {

	private static final String TRAINING_OUTLIER_SCORES =
			"training_outlier_scores.csv";

	private static final String VALIDATION_OUTLIER_SCORES =
			"validation_outlier_scores.csv";

	private static final String TEST_OUTLIER_SCORES =
			"test_outlier_scores.csv";

	private static final String VALIDATION_PREDICTIONS =
			"validation_predictions.csv";

	private static final String TEST_PREDICTIONS =
			"test_predictions.csv";

	private static final String TRAINING_PROXIMITIES_SPARSE =
			"training_proximities.mtx";

	private static final String TRAINING_PROXIMITIES_DENSE =
			"training_proximities.csv";

	private static final String TEST_TRAIN_PROXIMITIES_SPARSE =
			"test_train_proximities.mtx";

	private static final String TEST_TRAIN_PROXIMITIES_DENSE =
			"test_train_proximities.csv";

	ListObjectDataset train_data;
	ListObjectDataset test_data;

	public ExperimentRunner() {
	}

	public void run(boolean eval) throws Exception {

		if (!eval) {
			runTrainingMode();
		} else {
			runEvaluationMode();
		}
	}

	private void runTrainingMode() throws Exception {

		prepareSuppliedStandardization();

		ListObjectDataset trainDataOriginal = readTrainingData();

		prepareTrainingStatisticsBeforeTestRead(trainDataOriginal);

		ListObjectDataset testDataOriginal = AppContext.testing_file != null
				? readTestData()
				: null;

		// currently we don't support imputation on lazy datasets.
		validateDatasetCapabilities(trainDataOriginal, testDataOriginal);

		/*
		* Fit statistics only from training data, then apply the same fitted
		* transformation to both training and validation/test data.
		*/
		/*applyTrainingStandardization(
				trainDataOriginal,
				testDataOriginal
		);*/

		applyPreparedStandardization(trainDataOriginal, testDataOriginal);

		train_data = prepareTrainingData(trainDataOriginal);

		if (testDataOriginal != null) {
			test_data = prepareTestingData(
					testDataOriginal,
					train_data._get_initial_class_labels()
			);
		} else {
			test_data = null;
		}

		AppContext.setTraining_data(train_data);
		AppContext.setTesting_data(test_data);

		trainDataOriginal = null;
		testDataOriginal = null;
		System.gc();

		String datasetName = inferDatasetName(AppContext.training_file);
		AppContext.setDatasetName(datasetName);

		System.out.println();

		if (AppContext.shuffle_dataset) {
			System.out.println("Shuffling the training set...");
			train_data.shuffle();
		}

		ExperimentResultWriter experimentResults = new ExperimentResultWriter();

		for (int repetition = 0; repetition < AppContext.num_repeats; repetition++) {

			printRepetitionHeader(repetition, datasetName);

			clearProximityResults();

			Map<String, String> artifacts = new LinkedHashMap<>();

			Map<String, Double> additionalMetrics =
					new LinkedHashMap<>();

			Map<String, Long> additionalCounts =
					new LinkedHashMap<>();

			Map<String, Double> additionalTimings =
					new LinkedHashMap<>();

			if (AppContext.perform_train_imputation) {
				ProximityImputation.imputeTraining(
						train_data,
						repetition,
						(forest, data) -> computeTrainProximities(forest, data)
				);
			}

			clearProximityResults(); // not the final proximities.

			ProximityForest forest = new ProximityForest(
					repetition,
					AppContext.userdistances
			);

			forest.train(train_data);

			if (AppContext.savemodel) {
				//saveModel(forest);
				Path modelPath = saveModel(forest, repetition);
				artifacts.put("model", relativeArtifactPath(modelPath));
			}

			if (AppContext.impute_train) {
				writeTrainingData();
			}

			ProximityForestResult result = null;

			if (test_data != null) {
				//runValidationTesting(forest, repetition, datasetName);
				result = runValidationTesting(
						forest,
						repetition,
						datasetName,
						artifacts,
						additionalMetrics,
						additionalCounts,
						additionalTimings
				);
			}

			ScoreArtifact trainingOutlierArtifact =
					handleTrainingOutlierScores(
							forest,
							repetition
					);

			if (trainingOutlierArtifact != null) {
				artifacts.put(
						"trainingOutlierScores",
						relativeArtifactPath(
								trainingOutlierArtifact.path()
						)
				);

				addScoreArtifactResults(
						trainingOutlierArtifact,
						AppContext.isIsolationMode()
								? "trainingIsolationScore"
								: "trainingProximityOutlierScore",
						AppContext.isIsolationMode()
								? "trainingIsolationScoringMilliseconds"
								: "trainingProximityOutlierScoringMilliseconds",
						additionalMetrics,
						additionalCounts,
						additionalTimings
				);
			}

			//handleRequestedTrainingProximities(forest);

			Map<String, Path> proximityArtifacts = handleRequestedTrainingProximities(forest, repetition);

			addArtifacts(artifacts, proximityArtifacts);

			if (result == null) {
				result = forest.getResultSet();
			}

			//result.collateResults();

			experimentResults.add(
					buildExperimentResultRecord(
							result,
							datasetName,
							repetition,
							artifacts,
							additionalMetrics,
							additionalCounts,
							additionalTimings
					)
			);

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}
		}

		writeExperimentResults(experimentResults);
	}

	private void runEvaluationMode() throws Exception {

		ModelIO.LoadedModel loaded =
				ModelIO.loadModel(
						AppContext.modelname + ".ser"
				);

		ModelIO.applySnapshot(
				loaded.snapshot
		);

		ProximityForest forest =
				loaded.forest;

		ListObjectDataset trainData =
				loaded.trainData;

		this.train_data =
				trainData;

		AppContext.setTraining_data(
				trainData
		);

		ListObjectDataset testDataOriginal =
				readTestData();

		validateDatasetCapabilities(
				trainData,
				testDataOriginal
		);

		applyEvaluationStandardization(
				testDataOriginal
		);

		test_data =
				prepareTestingData(
						testDataOriginal,
						trainData._get_initial_class_labels()
				);

		AppContext.setTesting_data(
				test_data
		);

		testDataOriginal =
				null;

		System.gc();

		String datasetName =
				inferDatasetName(
						AppContext.training_file
				);

		AppContext.setDatasetName(
				datasetName
		);

		ExperimentResultWriter experimentResults =
				new ExperimentResultWriter();

		/*
		 * A loaded model represents one trained forest. Repeating evaluation may
		 * still be requested, but the same forest is reused each time.
		 */
		for (int repetition = 0;
			 repetition < AppContext.num_repeats;
			 repetition++) {

			clearProximityResults();

			Map<String, String> artifacts =
					new LinkedHashMap<>();

			Map<String, Double> additionalMetrics =
					new LinkedHashMap<>();

			Map<String, Long> additionalCounts =
					new LinkedHashMap<>();

			Map<String, Double> additionalTimings =
					new LinkedHashMap<>();

			if (AppContext.perform_test_imputation) {
				ProximityImputation.imputeTesting(
						test_data,
						trainData,
						forest,
						(forestForProx, test, train) ->
								computeTestTrainProximities(
										forestForProx,
										test,
										train
								)
				);
			}

			clearTestTrainProximityResults(); // these are intermediate proximities

			ProximityForestResult result =
					new ProximityForestResult(
							forest
					);

			if (AppContext.isIsolationMode()) {
				ScoreArtifact scoreArtifact =
						handleIsolationScores(
								forest,
								test_data,
								trainData.size(),
								TEST_OUTLIER_SCORES,
								repetition
						);

				artifacts.put(
						"testOutlierScores",
						relativeArtifactPath(
								scoreArtifact.path()
						)
				);

				addScoreArtifactResults(
						scoreArtifact,
						"testIsolationScore",
						"testIsolationScoringMilliseconds",
						additionalMetrics,
						additionalCounts,
						additionalTimings
				);

			} else {
				result =
						forest.test(
								test_data
						);

				if (!AppContext.perform_test_imputation) {
					result.printResults(
							datasetName,
							repetition,
							""
					);
				}

				if (AppContext.impute_test) {
					writeTestingData();
				}

				if (AppContext.get_predictions) {
					Path predictionPath =
							writePredictions(
									result,
									test_data,
									TEST_PREDICTIONS,
									repetition
							);

					artifacts.put(
							"testPredictions",
							relativeArtifactPath(
									predictionPath
							)
					);
				}

				if (AppContext.getprox) {
					Path proximityPath =
							computeAndWriteTestTrainProximities(
									forest,
									trainData,
									repetition
							);

					artifacts.put(
							"testTrainProximities",
							relativeArtifactPath(
									proximityPath
							)
					);
				}
			}

			//result.collateResults();

			experimentResults.add(
					buildExperimentResultRecord(
							result,
							datasetName,
							repetition,
							artifacts,
							additionalMetrics,
							additionalCounts,
							additionalTimings
					)
			);

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}
		}

		writeExperimentResults(
				experimentResults
		);
	}

	private ReaderOptions buildBaseReaderOptions(ReaderType readerType, String filePattern) { //(String dataPath) {

		if (readerType == null) {
			throw new IllegalArgumentException(
					"A reader type must be specified."
			);
		}

		return new ReaderOptions()
				.setReaderType(readerType)
				.setEntrySeparator(AppContext.entry_separator)
				.setArraySeparator(AppContext.array_separator)
				.setHasHeader(AppContext.csv_has_header)
				.set2D(AppContext.is2D)
				.setNumeric(AppContext.isNumeric)
				.setHasMissingValues(AppContext.hasMissingValues)
				.setTargetColumnIsFirst(AppContext.target_column_is_first)
				.setRegression(AppContext.isRegressionMode())
				.setIdColumn(AppContext.id_column)
				.setTimeColumn(AppContext.time_column)
				.setFeatureColumns(AppContext.feature_columns)
				.setLabelColumns(AppContext.label_columns)
				.setHdf5DatasetPath(AppContext.hdf5_dataset_path)
				.setHdf5LabelDatasetPath(AppContext.hdf5_label_dataset_path)
				.setFilePattern(filePattern)
				.setStandardizationStats(AppContext.standardizationStats);
	}


	private ListObjectDataset readTrainingData() throws IOException {

		ReaderOptions trainOptions =
				buildBaseReaderOptions(
						AppContext.getTrainingReaderType(),
						AppContext.getTrainingFilePattern()
				) //(AppContext.training_file)
						.setDataPath(AppContext.training_file)
						.setLabelPath(AppContext.training_labels)
						.setTest(false);

		DatasetReader trainReader =
				DatasetReaderFactory.create(trainOptions);

		return trainReader.read();
	}

	private ListObjectDataset readTestData() throws IOException {

		ReaderOptions testOptions =
				buildBaseReaderOptions(
						AppContext.getTestingReaderType(),
						AppContext.getTestingFilePattern()
				) //(AppContext.testing_file)
						.setDataPath(AppContext.testing_file)
						.setLabelPath(AppContext.testing_labels)
						.setTest(true);

		DatasetReader testReader =
				DatasetReaderFactory.create(testOptions);

		return testReader.read();
	}

	private ListObjectDataset prepareTrainingData(
			ListObjectDataset original) {

		ListObjectDataset prepared;

		if (AppContext.isClassificationMode()) {
			prepared = original.reorder_class_labels(null);
		} else {
			prepared = original;
		}

		if (AppContext.hasMissingValues && !isLazyDataset(prepared)) {
			prepared.setMissingIndices(
					MissingIndicesBuilder.buildFromDataset(prepared.getData())
			);
		}

		return prepared;
	}

	private ListObjectDataset prepareTestingData(
			ListObjectDataset original,
			Map<Object, Integer> labelMap) {

		ListObjectDataset prepared;

		if (AppContext.isClassificationMode()) {
			prepared = original.reorder_class_labels(labelMap);
		} else {
			prepared = original;
		}

		if (AppContext.hasMissingValues && !isLazyDataset(prepared)) {
			prepared.setMissingIndices(
					MissingIndicesBuilder.buildFromDataset(prepared.getData())
			);
		}

		return prepared;
	}

	private ProximityForestResult runValidationTesting(
			ProximityForest forest,
			int repetition,
			String datasetName,
			Map<String, String> artifacts,
			Map<String, Double> additionalMetrics,
			Map<String, Long> additionalCounts,
			Map<String, Double> additionalTimings
	) throws Exception {

		if (AppContext.perform_test_imputation) {
			ProximityImputation.imputeTesting(
					test_data,
					train_data,
					forest,
					(forestForProx, test, train) ->
							computeTestTrainProximities(
									forestForProx,
									test,
									train
							)
			);
		}

		if (AppContext.isIsolationMode()) {
			ScoreArtifact scoreArtifact =
					handleIsolationScores(
							forest,
							test_data,
							train_data.size(),
							VALIDATION_OUTLIER_SCORES,
							repetition
					);

			artifacts.put(
					"validationOutlierScores",
					relativeArtifactPath(
							scoreArtifact.path()
					)
			);

			addScoreArtifactResults(
					scoreArtifact,
					"validationIsolationScore",
					"validationIsolationScoringMilliseconds",
					additionalMetrics,
					additionalCounts,
					additionalTimings
			);

			return forest.getResultSet();
		}

		if (AppContext.impute_test) {
			writeTestingData();
		}

		ProximityForestResult result =
				forest.test(
						test_data
				);

		if (AppContext.get_predictions) {
			Path predictionPath =
					writePredictions(
							result,
							test_data,
							VALIDATION_PREDICTIONS,
							repetition
					);

			artifacts.put(
					"validationPredictions",
					relativeArtifactPath(
							predictionPath
					)
			);
		}

		result.printResults(
				datasetName,
				repetition,
				""
		);

		return result;
	}

	private Path saveModel(
			ProximityForest forest,
			int repetition
	) {
		try {
			AppContextSnapshot snapshot =
					AppContextUtils.captureSnapshot();

			Path modelPath =
					outputPath(
							repeatedFileName(
									AppContext.modelname + ".ser",
									repetition
							)
					);

			ModelIO.saveModel(
					modelPath.toString(),
					forest,
					train_data,
					snapshot
			);

			return modelPath;

		} catch (IOException e) {
			PrintUtilities.abort(e);

			throw new IllegalStateException(
					"Model saving failed.",
					e
			);
		}
	}

	private void writeTrainingData() throws IOException {

		GeneralUtilities.writeDelimitedData(
				train_data.getData(),
				AppContext.output_dir + AppContext.training_file,
				AppContext.array_separator,
				AppContext.entry_separator
		);
	}

	private void writeTestingData() throws IOException {

		GeneralUtilities.writeDelimitedData(
				test_data.getData(),
				AppContext.output_dir + AppContext.testing_file,
				AppContext.array_separator,
				AppContext.entry_separator
		);
	}

	private Path writePredictions(
			ProximityForestResult result,
			ListObjectDataset data,
			String baseFileName,
			int repetition
	) throws IOException {

		if (AppContext.isIsolationMode()) {
			throw new IllegalStateException(
					"Prediction output is unavailable in isolation mode."
			);
		}

		Path outputPath =
				outputPath(
						repeatedFileName(
								baseFileName,
								repetition
						)
				);

		if (AppContext.isRegressionMode()) {
			return PredictionWriter.writeRegression(
					outputPath,
					result.Predictions
			);
		}

		List<Object> predictedLabels =
				result.Predictions;

		Map<Integer, Object> newToOriginal =
				data.invertLabelMap(
						data._get_initial_class_labels()
				);

		List<Object> originalPredictions =
				predictedLabels.stream()
						.map(newToOriginal::get)
						.toList();

		return PredictionWriter.writeClassification(
				outputPath,
				originalPredictions
		);
	}

	private ScoreArtifact handleTrainingOutlierScores(
			ProximityForest forest,
			int repetition
	) throws Exception {

		if (!AppContext.get_training_outlier_scores) {
			return null;
		}

		if (AppContext.isIsolationMode()) {
			return handleIsolationScores(
					forest,
					train_data,
					train_data.size(),
					TRAINING_OUTLIER_SCORES,
					repetition
			);
		}

		if (AppContext.isRegressionMode()) {
			return null;
		}

		ensureTrainingProximities(
				forest
		);

		System.out.println(
				"Computing Training Outlier Scores..."
		);

		long start =
				System.nanoTime();

		double[] scores =
				OutlierScorer.getOutlierScores(
						AppContext.useSparseProximities,
						false,
						true,
						train_data._internal_class_array(),
						AppContext.training_proximities,
						AppContext.training_proximities_sparse
				);

		long end =
				System.nanoTime();

		Path outputPath =
				outputPath(
						repeatedFileName(
								TRAINING_OUTLIER_SCORES,
								repetition
						)
				);

		Path writtenPath =
				OutlierScoreWriter.write(
						outputPath,
						scores
				);

		return new ScoreArtifact(
				writtenPath,
				summarizeScores(
						scores
				),
				(
						end - start
				) / 1_000_000.0
		);
	}

	private Map<String, Path> handleRequestedTrainingProximities(
			ProximityForest forest,
			int repetition
	) throws IOException, ExecutionException, InterruptedException {

		Map<String, Path> artifacts =
				new LinkedHashMap<>();

		if (!AppContext.getprox) {
			return artifacts;
		}

		/*
		 * If outlier scoring was not requested, no earlier operation is
		 * guaranteed to have populated the requested proximity representation.
		 */
		ensureTrainingProximities(forest);

		Path trainingPath;

		if (AppContext.useSparseProximities) {
			if (AppContext.training_proximities_sparse == null) {
				throw new IllegalStateException(
						"Sparse training proximities were requested, but the "
								+ "sparse proximity map is null."
				);
			}

			trainingPath =
					outputPath(
							repeatedFileName(
									TRAINING_PROXIMITIES_SPARSE,
									repetition
							)
					);

			ProximityWriter.writeSparseTrainingMatrix(
					trainingPath,
					AppContext.training_proximities_sparse,
					train_data.size()
			);

		} else {
			if (AppContext.training_proximities == null) {
				throw new IllegalStateException(
						"Dense training proximities were requested, but the "
								+ "dense proximity matrix is null."
				);
			}

			trainingPath =
					outputPath(
							repeatedFileName(
									TRAINING_PROXIMITIES_DENSE,
									repetition
							)
					);

			ProximityWriter.writeDenseCsv(
					trainingPath,
					AppContext.training_proximities
			);
		}

		artifacts.put(
				"trainingProximities",
				trainingPath
		);

		if (test_data != null) {
			Path testTrainPath =
					computeAndWriteTestTrainProximities(
							forest,
							train_data,
							repetition
					);

			artifacts.put(
					"testTrainProximities",
					testTrainPath
			);
		}

		return artifacts;
	}

	private ScoreArtifact handleIsolationScores(
			ProximityForest forest,
			ListObjectDataset data,
			int normalizationSampleSize,
			String baseFileName,
			int repetition
	) throws Exception {

		long start =
				System.nanoTime();

		double[] scores =
				IsolationDepthScorer.score(
						forest,
						data,
						normalizationSampleSize
				);

		long end =
				System.nanoTime();

		Path outputPath =
				outputPath(
						repeatedFileName(
								baseFileName,
								repetition
						)
				);

		Path writtenPath =
				OutlierScoreWriter.write(
						outputPath,
						scores
				);

		return new ScoreArtifact(
				writtenPath,
				summarizeScores(
						scores
				),
				(
						end - start
				) / 1_000_000.0
		);
	}

	private Path computeAndWriteTestTrainProximities(
			ProximityForest forest,
			ListObjectDataset trainData,
			int repetition
	) throws IOException, ExecutionException, InterruptedException {

		System.out.println(
				"Computing Test/Train Proximities..."
		);

		long start =
				System.currentTimeMillis();

		computeTestTrainProximities(
				forest,
				test_data,
				trainData
		);

		long end =
				System.currentTimeMillis();

		System.out.println(
				"Done Computing Test/Train Proximities. "
						+ "Computation time: "
						+ (end - start)
						+ "ms"
		);

		if (AppContext.useSparseProximities) {
			if (AppContext.testing_training_proximities_sparse == null) {
				throw new IllegalStateException(
						"Sparse test/train proximities were requested, but "
								+ "the sparse proximity map is null."
				);
			}

			Path outputPath =
					outputPath(
							repeatedFileName(
									TEST_TRAIN_PROXIMITIES_SPARSE,
									repetition
							)
					);

			return ProximityWriter.writeSparseTestTrainMatrix(
					outputPath,
					AppContext.testing_training_proximities_sparse,
					test_data.size(),
					trainData.size()
			);
		}

		if (AppContext.testing_training_proximities == null) {
			throw new IllegalStateException(
					"Dense test/train proximities were requested, but "
							+ "the dense proximity matrix is null."
			);
		}

		Path outputPath =
				outputPath(
						repeatedFileName(
								TEST_TRAIN_PROXIMITIES_DENSE,
								repetition
						)
				);

		return ProximityWriter.writeDenseCsv(
				outputPath,
				AppContext.testing_training_proximities
		);
	}

	private ExperimentResultRecord buildExperimentResultRecord(
			ProximityForestResult result,
			String datasetName,
			int repetition,
			Map<String, String> artifacts,
			Map<String, Double> additionalMetrics,
			Map<String, Long> additionalCounts,
			Map<String, Double> additionalTimings
	) {
		result.collateResults();

		ExperimentResultRecord.Builder builder =
				ExperimentResultRecord.builder()
						.setDataset(datasetName)
						.setRepetition(repetition + 1)
						.setForestId(result.forest_id)
						.setForestMode(
								AppContext.forest_mode
						)
						.addCount(
								"trainingInstanceCount",
								train_data == null
										? 0L
										: train_data.size()
						)
						.addCount(
								"testingInstanceCount",
								test_data == null
										? 0L
										: test_data.size()
						)
						.addTimingNanoseconds(
								"trainingMilliseconds",
								Math.max(
										0L,
										result.elapsedTimeTrain
								)
						)
						.addTimingNanoseconds(
								"testingMilliseconds",
								Math.max(
										0L,
										result.elapsedTimeTest
								)
						)
						.addForestStatistic(
								"numTrees",
								result.total_num_trees
						)
						//.addConfiguration(
						//		"numTrees",
						//		AppContext.num_trees
						//)
						.addConfiguration(
								"numCandidatesPerSplit",
								AppContext.num_candidates_per_split
						)
						.addConfiguration(
								"bootstrapTrees",
								AppContext.bootstrap_trees
						)
						.addConfiguration(
								"parallelTrees",
								AppContext.parallelTrees
						)
						.addConfiguration(
								"parallelSplitAssignments",
								AppContext.parallel_split_assignments
						)
						.addConfiguration(
								"parallelSplitAssignmentThreshold",
								AppContext.parallel_split_assignment_threshold
						)
						.addForestStatistic(
								"meanNodesPerTree",
								result.mean_num_nodes_per_tree
						)
						.addForestStatistic(
								"standardDeviationNodesPerTree",
								result.sd_num_nodes_per_tree
						)
						.addForestStatistic(
								"meanDepthPerTree",
								result.mean_depth_per_tree
						)
						.addForestStatistic(
								"standardDeviationDepthPerTree",
								result.sd_depth_per_tree
						)
						.addForestStatistic(
								"meanWeightedDepthPerTree",
								result.mean_weighted_depth_per_tree
						)
						.addForestStatistic(
								"standardDeviationWeightedDepthPerTree",
								result.sd_weighted_depth_per_tree
						)
						.addConfiguration(
								"proximityType",
								AppContext.proximityType
						)
						.addConfiguration(
								"trainingReaderType",
								AppContext.getTrainingReaderType()
						)
						.addConfiguration(
								"testingReaderType",
								test_data == null
										? null
										: AppContext.getTestingReaderType()
						)
						.addConfiguration(
								"standardizationMethod",
								AppContext.standardizationConfig == null
										? null
										: AppContext.standardizationConfig.getMethod()
						)
						.addConfiguration(
								"standardizationScope",
								AppContext.standardizationConfig == null
										? null
										: AppContext.standardizationConfig.getScope()
						)
						.addMetrics(additionalMetrics)
						.addCounts(additionalCounts)
						.addArtifacts(
								artifacts
						);

		if (additionalTimings != null) {
			for (Map.Entry<String, Double> entry : additionalTimings.entrySet()) {
				builder.addTimingMilliseconds(entry.getKey(), entry.getValue());
			}
		}

		if (AppContext.isIsolationMode()) {
			builder.addConfiguration(
					"isolationNumBranches",
					AppContext.isolation_num_branches
			);

			builder.addConfiguration(
					"isolationMinLeafSize",
					AppContext.isolation_min_leaf_size
			);

			builder.addConfiguration(
					"isolationScoreMethod",
					"pathLength"
			);

			builder.addConfiguration(
					"purityMeasure",
					AppContext.purity_measure
			);
		}

		if (AppContext.isClassificationMode()) {
			int total =
					result.correct + result.errors;

			if (total > 0) {
				double accuracy =
						(double) result.correct
								/ total;

				double errorRate =
						(double) result.errors
								/ total;

				builder.addMetric(
						"accuracy",
						accuracy
				);

				builder.addMetric(
						"errorRate",
						errorRate
				);
			}

			builder.addCount(
					"correct",
					result.correct
			);

			builder.addCount(
					"errors",
					result.errors
			);

		} else if (AppContext.isRegressionMode()) {
			/*
			 * result.score is retained under a generic legacy name until the
			 * regression result implementation explicitly identifies whether
			 * it represents RMSE, MAE, R2, or another metric.
			 */
			if (Double.isFinite(result.score)) {
				builder.addMetric(
						"regressionScore",
						result.score
				);
			}
		}

		return builder.build();
	}

	private void writeExperimentResults(
			ExperimentResultWriter resultWriter
	) throws IOException {

		if (AppContext.export_level < 1
				|| resultWriter.isEmpty()) {

			return;
		}

		Path outputPath =
				resultWriter.writeToDirectory(
						Paths.get(
								AppContext.output_dir
						)
				);

		if (AppContext.verbosity > 0) {
			System.out.println(
					"Wrote experiment results to: "
							+ outputPath
			);
		}
	}

	private Path outputPath(
			String fileName
	) {
		return Paths.get(
				AppContext.output_dir,
				fileName
		).toAbsolutePath().normalize();
	}

	private String repeatedFileName(
			String baseFileName,
			int repetition
	) {
		if (AppContext.num_repeats <= 1) {
			return baseFileName;
		}

		int separatorIndex =
				baseFileName.lastIndexOf('.');

		String suffix =
				"_repeat_" + (repetition + 1);

		if (separatorIndex <= 0) {
			return baseFileName + suffix;
		}

		return baseFileName.substring(
				0,
				separatorIndex
		)
				+ suffix
				+ baseFileName.substring(
				separatorIndex
		);
	}

	private String relativeArtifactPath(
			Path artifactPath
	) {
		if (artifactPath == null) {
			return null;
		}

		Path outputDirectory =
				Paths.get(
						AppContext.output_dir
				).toAbsolutePath().normalize();

		Path normalizedArtifact =
				artifactPath.toAbsolutePath()
						.normalize();

		try {
			return outputDirectory.relativize(
					normalizedArtifact
			).toString();

		} catch (IllegalArgumentException e) {
			/*
			 * This can occur when paths are on different filesystem roots.
			 */
			return normalizedArtifact.toString();
		}
	}

	private void addArtifacts(
			Map<String, String> destination,
			Map<String, Path> source
	) {
		for (Map.Entry<String, Path> entry :
				source.entrySet()) {

			destination.put(
					entry.getKey(),
					relativeArtifactPath(
							entry.getValue()
					)
			);
		}
	}

	private void printRepetitionHeader(
			int repetition,
			String datasetName) {

		if (AppContext.verbosity > 0) {
			System.out.println(
					"-----------------Repetition No: "
							+ (repetition + 1)
							+ " ("
							+ datasetName
							+ ")   -----------------"
			);
			PrintUtilities.printMemoryUsage();
		} else if (AppContext.verbosity == 0 && repetition == 0) {
			System.out.println(
					"Repetition, Dataset, Score, TrainingTime(ms), TestingTime(ms), MeanDepthPerTree"
			);
		}
	}

	private String inferDatasetName(
			String trainingFilePath
	) {
		if (trainingFilePath == null
				|| trainingFilePath.isBlank()) {

			return "dataset";
		}

		File trainingFile =
				new File(
						trainingFilePath
				);

		return trainingFile
				.getName()
				.replaceFirst(
						"(?i)_TRAIN\\.(txt|tsv|csv)$",
						""
				);
	}

	private void validateDatasetCapabilities(
			ListObjectDataset trainData,
			ListObjectDataset testData
	) {
		boolean lazyTraining = isLazyDataset(trainData);
		boolean lazyTesting = isLazyDataset(testData);

		if (!lazyTraining && !lazyTesting) {
			return;
		}

		if (AppContext.perform_train_imputation
				|| AppContext.perform_test_imputation
				|| AppContext.impute_train
				|| AppContext.impute_test) {

			throw new UnsupportedOperationException(
					"Imputation is not currently supported for lazy datasets. "
							+ "Disable perform_train_imputation, perform_test_imputation, "
							+ "impute_train, and impute_test when using a lazy reader."
			);
		}

		StandardizationConfig standardizationConfig =
				AppContext.standardizationConfig;

		if (lazyTraining
				&& standardizationConfig != null
				&& standardizationConfig.isEnabled()
				&& AppContext.standardizationStats == null) {

			throw new UnsupportedOperationException(
					"Lazy training with standardization requires precomputed "
							+ "statistics supplied through -standardization_stats. "
							+ "Automatic lazy statistics fitting is not implemented."
			);
		}

		if (lazyTraining != lazyTesting && trainData != null && testData != null) {
			System.out.println(
					"Warning: training and testing datasets use different storage modes. "
							+ "One dataset is lazy and the other is materialized."
			);
		}
	}

	private boolean isLazyDataset(ListObjectDataset dataset) {
		if (dataset == null) {
			return false;
		}

		for (Object value : dataset.getData()) {
			if (value != null) {
				return value instanceof LazySeriesRef;
			}
		}

		return false;
	}

	// here is a stricter check (not needed if no mixing can happen)

	/*private boolean isLazyDataset(ListObjectDataset dataset) {
		if (dataset == null) {
			return false;
		}

		boolean foundLazy = false;
		boolean foundMaterialized = false;

		for (Object value : dataset.getData()) {
			if (value == null) {
				continue;
			}

			if (value instanceof LazySeriesRef) {
				foundLazy = true;
			} else {
				foundMaterialized = true;
			}

			if (foundLazy && foundMaterialized) {
				throw new IllegalStateException(
						"A ListObjectDataset cannot mix lazy references "
								+ "and materialized instances."
				);
			}
		}

		return foundLazy;
	}*/

	private void applyTrainingStandardization(
			ListObjectDataset trainingData,
			ListObjectDataset testingData
	) throws IOException {

		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			AppContext.standardizationStats =
					null;

			return;
		}

		config.requireImplemented();

		boolean lazyTraining =
				isLazyDataset(trainingData);

		boolean lazyTesting =
				isLazyDataset(testingData);

		List<String> featureNames =
				getStandardizationFeatureNames();

		StandardizationStats stats =
				AppContext.standardizationStats;

		if (config.shouldLoadStatistics()) {
			/*
			 * prepareSuppliedStandardization() already loaded and validated the
			 * statistics before the lazy or eager readers were constructed.
			 */
			if (stats == null) {
				throw new IllegalStateException(
						"Supplied standardization statistics were not loaded "
								+ "before the dataset readers were created."
				);
			}
		} else {
			if (lazyTraining) {
				throw new UnsupportedOperationException(
						"Lazy training with standardization currently requires "
								+ "precomputed statistics supplied through "
								+ "-standardization_stats. Automatic streaming fit "
								+ "for lazy training is reserved for Phase 4."
				);
			}

			stats =
					StandardizationFitter.fit(
							trainingData,
							config.getMethod(),
							config.getScope(),
							config.getVarianceConvention(),
							featureNames
					);

			config.validateStatistics(stats);

			AppContext.standardizationStats =
					stats;

			if (AppContext.verbosity > 0) {
				System.out.println(
						"Fitted standardization statistics from "
								+ "the eager training dataset."
				);
			}
		}

		/*
		 * Eager datasets must be transformed here.
		 *
		 * Lazy datasets are transformed by their PerFile*SeriesReader each time
		 * a reference is materialized.
		 */
		if (!lazyTraining) {
			Standardizer.transformInPlace(
					trainingData,
					stats,
					featureNames
			);
		}

		if (testingData != null && !lazyTesting) {
			Standardizer.transformInPlace(
					testingData,
					stats,
					featureNames
			);
		}

		if (config.shouldSaveFittedStatistics()) {
			Path outputPath =
					resolveStandardizationOutputPath(config);

			StandardizationJson.write(
					outputPath,
					stats
			);

			if (AppContext.verbosity > 0) {
				System.out.println(
						"Saved standardization statistics to: "
								+ outputPath
				);
			}
		}

		if (AppContext.verbosity > 0
				&& !config.shouldLoadStatistics()) {

			printStandardizationSummary(stats);
		}
	}

	private Path resolveStandardizationOutputPath(
			StandardizationConfig config
	) {
		String configuredPath =
				config.getStatisticsOutputPath();

		if (configuredPath != null
				&& !configuredPath.isBlank()) {

			return Paths.get(
					configuredPath
			);
		}

		return Paths.get(
				AppContext.output_dir,
				"standardization_stats.json"
		);
	}

	private List<String> getStandardizationFeatureNames() {
		if (AppContext.feature_columns == null
				|| AppContext.feature_columns.isEmpty()) {

			return List.of();
		}

		return new java.util.ArrayList<>(
				AppContext.feature_columns
		);
	}

	private void printStandardizationSummary(
			StandardizationStats stats
	) {
		System.out.println(
				"Applied "
						+ stats.getMethod()
						+ " standardization with scope "
						+ stats.getScope()
						+ " using "
						+ stats.getStatisticGroupCount()
						+ " fitted statistic group(s)."
		);

		if (AppContext.verbosity > 1) {
			System.out.println(
					"Standardization centers: "
							+ java.util.Arrays.toString(
							stats.getCenters()
					)
			);

			System.out.println(
					"Standardization scales: "
							+ java.util.Arrays.toString(
							stats.getScales()
					)
			);

			System.out.println(
					"Standardization counts: "
							+ java.util.Arrays.toString(
							stats.getCounts()
					)
			);
		}
	}

	private void applyEvaluationStandardization(
			ListObjectDataset testingData
	) {
		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			return;
		}

		config.requireImplemented();

		StandardizationStats stats =
				AppContext.standardizationStats;

		if (stats == null) {
			throw new IllegalStateException(
					"The loaded model enables standardization but does not "
							+ "contain fitted or supplied statistics."
			);
		}

		config.validateStatistics(stats);

		if (isLazyDataset(testingData)) {
			/*
			 * readTestData() constructed the lazy test reader using the statistics
			 * restored by ModelIO.applySnapshot(). Each series will be transformed
			 * immediately after materialization.
			 */
			if (AppContext.verbosity > 0) {
				System.out.println(
						"Lazy testing data will use saved "
								+ stats.getMethod()
								+ " standardization during materialization."
				);
			}

			return;
		}

		List<String> featureNames =
				getStandardizationFeatureNames();

		Standardizer.transformInPlace(
				testingData,
				stats,
				featureNames
		);

		if (AppContext.verbosity > 0) {
			System.out.println(
					"Applied saved "
							+ stats.getMethod()
							+ " standardization to the eager testing dataset."
			);
		}
	}

	private void prepareSuppliedStandardization()
			throws IOException {

		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			AppContext.standardizationStats =
					null;

			return;
		}

		config.requireImplemented();

		if (!config.shouldLoadStatistics()) {
			/*
			 * Eager training will fit statistics after the dataset is read.
			 * Lazy training without supplied statistics is rejected later.
			 */
			AppContext.standardizationStats =
					null;

			return;
		}

		List<String> featureNames =
				getStandardizationFeatureNames();

		StandardizationStats stats =
				StandardizationJson.read(
						config.getStatisticsPath(),
						config,
						featureNames
				);

		config.validateStatistics(stats);

		AppContext.standardizationStats =
				stats;

		if (AppContext.verbosity > 0) {
			System.out.println(
					"Loaded standardization statistics from: "
							+ config.getStatisticsPath()
			);

			printStandardizationSummary(stats);
		}
	}

	private void prepareTrainingStatisticsBeforeTestRead(
			ListObjectDataset trainingData
	) throws IOException {

		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			AppContext.standardizationStats =
					null;

			return;
		}

		config.requireImplemented();

		if (config.shouldLoadStatistics()) {
			/*
			 * Already loaded by prepareSuppliedStandardization().
			 */
			if (AppContext.standardizationStats == null) {
				throw new IllegalStateException(
						"Configured standardization statistics were not loaded."
				);
			}

			return;
		}

		if (isLazyDataset(trainingData)) {
			throw new UnsupportedOperationException(
					"Lazy training with standardization currently requires "
							+ "precomputed statistics supplied through "
							+ "-standardization_stats."
			);
		}

		List<String> featureNames =
				getStandardizationFeatureNames();

		StandardizationStats stats =
				StandardizationFitter.fit(
						trainingData,
						config.getMethod(),
						config.getScope(),
						config.getVarianceConvention(),
						featureNames
				);

		config.validateStatistics(stats);

		AppContext.standardizationStats =
				stats;

		if (config.shouldSaveFittedStatistics()) {
			Path outputPath =
					resolveStandardizationOutputPath(config);

			StandardizationJson.write(
					outputPath,
					stats
			);

			if (AppContext.verbosity > 0) {
				System.out.println(
						"Saved fitted standardization statistics to: "
								+ outputPath
				);
			}
		}

		if (AppContext.verbosity > 0) {
			System.out.println(
					"Fitted standardization statistics from "
							+ "the eager training dataset."
			);

			printStandardizationSummary(stats);
		}
	}

	private void applyPreparedStandardization(
			ListObjectDataset trainingData,
			ListObjectDataset testingData
	) {
		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			return;
		}

		StandardizationStats stats =
				AppContext.standardizationStats;

		if (stats == null) {
			throw new IllegalStateException(
					"Standardization is enabled but no prepared statistics "
							+ "are available."
			);
		}

		config.validateStatistics(stats);

		List<String> featureNames =
				getStandardizationFeatureNames();

		if (!isLazyDataset(trainingData)) {
			Standardizer.transformInPlace(
					trainingData,
					stats,
					featureNames
			);
		}

		if (testingData != null
				&& !isLazyDataset(testingData)) {

			Standardizer.transformInPlace(
					testingData,
					stats,
					featureNames
			);
		}
	}

	private record NumericScoreSummary(
			long count,
			long nonfiniteCount,
			double mean,
			double populationStandardDeviation,
			double minimum,
			double maximum
	) {
	}

	private record ScoreArtifact(
			Path path,
			NumericScoreSummary summary,
			double computationMilliseconds
	) {
	}

	private static NumericScoreSummary summarizeScores(
			double[] scores
	) {
		if (scores == null || scores.length == 0) {
			return new NumericScoreSummary(
					0L,
					0L,
					Double.NaN,
					Double.NaN,
					Double.NaN,
					Double.NaN
			);
		}

		long count =
				0L;

		long nonfiniteCount =
				0L;

		double mean =
				0.0;

		double m2 =
				0.0;

		double minimum =
				Double.POSITIVE_INFINITY;

		double maximum =
				Double.NEGATIVE_INFINITY;

		for (double score : scores) {
			if (!Double.isFinite(score)) {
				nonfiniteCount++;
				continue;
			}

			count++;

			double delta =
					score - mean;

			mean +=
					delta / count;

			double updatedDelta =
					score - mean;

			m2 +=
					delta * updatedDelta;

			minimum =
					Math.min(
							minimum,
							score
					);

			maximum =
					Math.max(
							maximum,
							score
					);
		}

		if (count == 0L) {
			return new NumericScoreSummary(
					0L,
					nonfiniteCount,
					Double.NaN,
					Double.NaN,
					Double.NaN,
					Double.NaN
			);
		}

		double populationVariance =
				Math.max(
						0.0,
						m2
				) / count;

		return new NumericScoreSummary(
				count,
				nonfiniteCount,
				mean,
				Math.sqrt(
						populationVariance
				),
				minimum,
				maximum
		);
	}

	private static void addScoreArtifactResults(
			ScoreArtifact scoreArtifact,
			String metricPrefix,
			String timingName,
			Map<String, Double> metrics,
			Map<String, Long> counts,
			Map<String, Double> timings
	) {
		if (scoreArtifact == null) {
			return;
		}

		NumericScoreSummary summary =
				scoreArtifact.summary();

		counts.put(
				metricPrefix + "Count",
				summary.count()
		);

		counts.put(
				metricPrefix + "NonfiniteCount",
				summary.nonfiniteCount()
		);

		timings.put(
				timingName,
				scoreArtifact.computationMilliseconds()
		);

		if (summary.count() == 0L) {
			return;
		}

		metrics.put(
				metricPrefix + "Mean",
				summary.mean()
		);

		metrics.put(
				metricPrefix
						+ "PopulationStandardDeviation",
				summary.populationStandardDeviation()
		);

		metrics.put(
				metricPrefix + "Minimum",
				summary.minimum()
		);

		metrics.put(
				metricPrefix + "Maximum",
				summary.maximum()
		);
	}

	private boolean trainingProximitiesAvailable() {
		if (AppContext.useSparseProximities) {
			return AppContext.training_proximities_sparse
					!= null;
		}

		return AppContext.training_proximities
				!= null;
	}

	private void ensureTrainingProximities(
			ProximityForest forest
	) throws IOException,
			ExecutionException,
			InterruptedException {

		if (trainingProximitiesAvailable()) {
			return;
		}

		System.out.println(
				"Computing Training Proximities..."
		);

		long start =
				System.nanoTime();

		computeTrainProximities(
				forest,
				train_data
		);

		long elapsedMilliseconds =
				(
						System.nanoTime()
								- start
				) / 1_000_000L;

		System.out.println(
				"Done Computing Training Proximities. "
						+ "Computation time: "
						+ elapsedMilliseconds
						+ "ms"
		);

		if (!trainingProximitiesAvailable()) {
			throw new IllegalStateException(
					AppContext.useSparseProximities
							? "Training proximity computation completed, but "
							+ "the sparse proximity map remains null."
							: "Training proximity computation completed, but "
							+ "the dense proximity matrix remains null."
			);
		}
	}

	private void clearProximityResults() {
		AppContext.training_proximities =
				null;

		AppContext.training_proximities_sparse =
				null;

		AppContext.testing_training_proximities =
				null;

		AppContext.testing_training_proximities_sparse =
				null;
	}

	private void clearTestTrainProximityResults() {
		AppContext.testing_training_proximities =
				null;

		AppContext.testing_training_proximities_sparse =
				null;
	}
}