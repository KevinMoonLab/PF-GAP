package trees;

import core.AppContext;
import core.ProximityForestResult;
import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;
import distance.MEASURE;
import util.PrintUtilities;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.IntStream;

/**
 * Proximity forest supporting classification, regression, and isolation
 * workflows.
 *
 * <p>Training may be parallelized across trees. Evaluation may be
 * parallelized either across test instances or across trees for one test
 * instance. The two prediction strategies are not nested.</p>
 *
 * <p>Lazy test instances are materialized once at the forest boundary and
 * reused throughout every tree traversal. Node exemplars remain controlled
 * by their splitter and may be resolved temporarily at each visited node.</p>
 */
public class ProximityForest
		implements Serializable {

	@Serial
	private static final long serialVersionUID =
			-1183368028217094381L;

	protected ProximityForestResult result;

	protected int forest_id;

	protected ProximityTree[] trees;

	public String prefix;

	/**
	 * Retained only for compatibility with older code that may inspect this
	 * field directly.
	 */
	List<Object> predictions;

	/**
	 * Prevents two callers from training the same forest simultaneously.
	 *
	 * <p>The lock does not serialize the individual trees within one
	 * training call.</p>
	 */
	private final ReentrantLock trainLock =
			new ReentrantLock();

	/**
	 * Stable seed used to derive prediction tie-breaking seeds.
	 *
	 * <p>This seed is assigned while forests are constructed sequentially.
	 * It is serialized with the model.</p>
	 */
	private final long forestSeed;

	public ProximityForest(
			int forestId,
			MEASURE... selectedDistances
	) {
		this.result =
				new ProximityForestResult(
						this
				);

		this.forest_id =
				forestId;

		this.forestSeed =
				AppContext.getRand()
						.nextLong();

		this.trees =
				new ProximityTree[
						AppContext.num_trees
						];

		MEASURE[] distances =
				selectedDistances == null
						? new MEASURE[0]
						: selectedDistances.clone();

		for (int treeIndex = 0;
			 treeIndex < trees.length;
			 treeIndex++) {

			trees[treeIndex] =
					new ProximityTree(
							treeIndex,
							this,
							distances
					);
		}
	}

	/**
	 * Trains all trees in the forest.
	 *
	 * <p>Parallel tree failures are propagated to the caller. A failed tree
	 * is never silently omitted from the forest.</p>
	 */
	public void train(
			ListObjectDataset trainData
	) throws Exception {

		Objects.requireNonNull(
				trainData,
				"Training data cannot be null."
		);

		if (trainData.size() == 0) {
			throw new IllegalArgumentException(
					"Training data cannot be empty."
			);
		}

		trainLock.lock();

		try {
			result.startTimeTrain =
					System.nanoTime();

			if (AppContext.parallelTrees
					&& trees.length > 1) {

				trainTreesParallel(
						trainData
				);
			} else {
				trainTreesSequential(
						trainData
				);
			}

			result.endTimeTrain =
					System.nanoTime();

			result.elapsedTimeTrain =
					result.endTimeTrain
							- result.startTimeTrain;

			if (AppContext.verbosity > 0) {
				System.out.println();

				PrintUtilities.printMemoryUsage();
			}
		} finally {
			trainLock.unlock();
		}
	}

	private void trainTreesSequential(
			ListObjectDataset trainData
	) throws Exception {

		for (int treeIndex = 0;
			 treeIndex < trees.length;
			 treeIndex++) {

			trees[treeIndex].train(
					trainData
			);

			reportTreeProgress(
					treeIndex
			);
		}
	}

	private void trainTreesParallel(
			ListObjectDataset trainData
	) throws Exception {

		int workerCount =
				effectiveTreeWorkerCount();

		ExecutorService executor =
				Executors.newFixedThreadPool(
						workerCount
				);

		List<Future<Void>> futures =
				new ArrayList<>(
						trees.length
				);

		try {
			for (int treeIndex = 0;
				 treeIndex < trees.length;
				 treeIndex++) {

				final int submittedTreeIndex =
						treeIndex;

				futures.add(
						executor.submit(() -> {
							trees[submittedTreeIndex].train(
									trainData
							);

							reportTreeProgress(
									submittedTreeIndex
							);

							return null;
						})
				);
			}

			waitForFutures(
					futures,
					"Parallel tree training failed."
			);
		} finally {
			shutdownExecutor(
					executor
			);
		}
	}

	private int effectiveTreeWorkerCount() {
		return Math.max(
				1,
				Math.min(
						trees.length,
						Runtime.getRuntime()
								.availableProcessors()
				)
		);
	}

	private void reportTreeProgress(
			int treeIndex
	) {
		if (AppContext.verbosity <= 0) {
			return;
		}

		synchronized (System.out) {
			System.out.print(
					treeIndex
							+ "."
			);

			if (AppContext.verbosity > 1) {
				PrintUtilities.printMemoryUsage(
						true
				);

				if ((treeIndex + 1) % 20 == 0) {
					System.out.println();
				}
			}
		}
	}

	/**
	 * Evaluates the forest on a dataset.
	 *
	 * <p>When parallelPredict is enabled, test instances are evaluated in
	 * parallel and each instance traverses the trees sequentially. This
	 * avoids nested instance-by-tree parallelism.</p>
	 *
	 * <p>When parallelPredict is disabled but parallelTrees is enabled,
	 * test instances are processed sequentially while the trees for each
	 * instance are evaluated using one executor reused throughout this test
	 * call.</p>
	 */
	public ProximityForestResult test(
			ListObjectDataset testData
	) throws Exception {

		Objects.requireNonNull(
				testData,
				"Test data cannot be null."
		);

		result.startTimeTest =
				System.nanoTime();

		int testSize =
				testData.size();

		Object[] actualLabels =
				new Object[testSize];

		Object[] predictedLabels =
				new Object[testSize];

		ProximityTree.Node[][] reachedLeaves =
				new ProximityTree.Node[
						testSize
						][trees.length];

		boolean parallelInstances =
				AppContext.parallelPredict
						&& testSize > 1;

		boolean parallelTreesPerInstance =
				!parallelInstances
						&& AppContext.parallelTrees
						&& trees.length > 1;

		ExecutorService treePredictionExecutor =
				parallelTreesPerInstance
						? Executors.newFixedThreadPool(
						effectiveTreeWorkerCount()
				)
						: null;

		try {
			if (parallelInstances) {
				evaluateInstancesParallel(
						testData,
						actualLabels,
						predictedLabels,
						reachedLeaves
				);
			} else {
				evaluateInstancesSequential(
						testData,
						actualLabels,
						predictedLabels,
						reachedLeaves,
						treePredictionExecutor
				);
			}
		} finally {
			if (treePredictionExecutor != null) {
				shutdownExecutor(
						treePredictionExecutor
				);
			}
		}

		/*
		 * TestIndices lists are ArrayLists and are not thread-safe.
		 * Membership is therefore merged only after all parallel traversal
		 * has completed.
		 */
		recordReachedLeaves(
				reachedLeaves
		);

		result.Predictions =
				new ArrayList<>(
						Arrays.asList(
								predictedLabels
						)
				);

		predictions =
				result.Predictions;

		calculateEvaluationMetrics(
				actualLabels,
				predictedLabels
		);

		result.endTimeTest =
				System.nanoTime();

		result.elapsedTimeTest =
				result.endTimeTest
						- result.startTimeTest;

		if (AppContext.verbosity > 0) {
			System.out.println();
		}

		return result;
	}

	private void evaluateInstancesSequential(
			ListObjectDataset testData,
			Object[] actualLabels,
			Object[] predictedLabels,
			ProximityTree.Node[][] reachedLeaves,
			ExecutorService treePredictionExecutor
	) throws Exception {

		for (int testIndex = 0;
			 testIndex < testData.size();
			 testIndex++) {

			evaluateOneInstance(
					testData,
					testIndex,
					actualLabels,
					predictedLabels,
					reachedLeaves,
					treePredictionExecutor
			);

			reportTestProgress(
					testIndex
			);
		}
	}

	private void evaluateInstancesParallel(
			ListObjectDataset testData,
			Object[] actualLabels,
			Object[] predictedLabels,
			ProximityTree.Node[][] reachedLeaves
	) throws Exception {

		try {
			IntStream.range(
							0,
							testData.size()
					)
					.parallel()
					.forEach(testIndex -> {
						try {
							/*
							 * Tree-level prediction is deliberately
							 * sequential inside an instance-parallel run.
							 */
							evaluateOneInstance(
									testData,
									testIndex,
									actualLabels,
									predictedLabels,
									reachedLeaves,
									null
							);

							reportTestProgress(
									testIndex
							);
						} catch (Exception e) {
							throw new PredictionExecutionException(
									e
							);
						}
					});
		} catch (PredictionExecutionException e) {
			throw unwrapPredictionException(
					e
			);
		}
	}

	private void evaluateOneInstance(
			ListObjectDataset testData,
			int testIndex,
			Object[] actualLabels,
			Object[] predictedLabels,
			ProximityTree.Node[][] reachedLeaves,
			ExecutorService treePredictionExecutor
	) throws Exception {

		actualLabels[testIndex] =
				testData.get_class(
						testIndex
				);

		Object storedQuery =
				testData.get_series(
						testIndex
				);

		Object resolvedQuery =
				resolvePredictionQuery(
						storedQuery
				);

		Object[] treePredictions =
				new Object[trees.length];

		if (treePredictionExecutor == null) {
			evaluateTreesSequential(
					resolvedQuery,
					testIndex,
					treePredictions,
					reachedLeaves[testIndex]
			);
		} else {
			evaluateTreesParallel(
					resolvedQuery,
					testIndex,
					treePredictions,
					reachedLeaves[testIndex],
					treePredictionExecutor
			);
		}

		predictedLabels[testIndex] =
				combineTreePredictions(
						treePredictions
				);
	}

	/**
	 * Resolves a lazy prediction query once at the forest boundary.
	 *
	 * <p>An eager input is returned unchanged.</p>
	 */
	private Object resolvePredictionQuery(
			Object query
	) {
		if (query == null) {
			throw new IllegalArgumentException(
					"Prediction query cannot be null."
			);
		}

		if (query instanceof LazySeriesRef reference) {
			return AppContext.readLazySeries(
					reference
			);
		}

		return query;
	}

	private void evaluateTreesSequential(
			Object resolvedQuery,
			int testIndex,
			Object[] treePredictions,
			ProximityTree.Node[] reachedLeaves
	) throws Exception {

		for (int treeIndex = 0;
			 treeIndex < trees.length;
			 treeIndex++) {

			ProximityTree.Node leaf =
					trees[treeIndex]
							.findLeafResolved(
									resolvedQuery,
									predictionRandom(
											testIndex,
											treeIndex
									)
							);

			reachedLeaves[treeIndex] =
					leaf;

			treePredictions[treeIndex] =
					leaf.label();
		}
	}

	private void evaluateTreesParallel(
			Object resolvedQuery,
			int testIndex,
			Object[] treePredictions,
			ProximityTree.Node[] reachedLeaves,
			ExecutorService executor
	) throws Exception {

		List<Future<Void>> futures =
				new ArrayList<>(
						trees.length
				);

		for (int treeIndex = 0;
			 treeIndex < trees.length;
			 treeIndex++) {

			final int submittedTreeIndex =
					treeIndex;

			futures.add(
					executor.submit(() -> {
						ProximityTree.Node leaf =
								trees[submittedTreeIndex]
										.findLeafResolved(
												resolvedQuery,
												predictionRandom(
														testIndex,
														submittedTreeIndex
												)
										);

						reachedLeaves[submittedTreeIndex] =
								leaf;

						treePredictions[submittedTreeIndex] =
								leaf.label();

						return null;
					})
			);
		}

		waitForFutures(
				futures,
				"Parallel tree prediction failed."
		);
	}

	/**
	 * Compatibility method for predicting one instance.
	 *
	 * <p>The query is resolved once. This method does not create a new
	 * executor for each prediction.</p>
	 */
	public Object predict(
			Object query,
			int index
	) throws Exception {

		Object resolvedQuery =
				resolvePredictionQuery(
						query
				);

		Object[] treePredictions =
				new Object[trees.length];

		ProximityTree.Node[] reachedLeaves =
				new ProximityTree.Node[
						trees.length
						];

		evaluateTreesSequential(
				resolvedQuery,
				index,
				treePredictions,
				reachedLeaves
		);

		recordReachedLeaves(
				new ProximityTree.Node[][]{
						reachedLeaves
				},
				index
		);

		return combineTreePredictions(
				treePredictions
		);
	}

	private Object combineTreePredictions(
			Object[] treePredictions
	) {
		if (AppContext.isRegression) {
			return combineRegressionPredictions(
					treePredictions
			);
		}

		return combineClassificationPredictions(
				treePredictions
		);
	}

	private Object combineRegressionPredictions(
			Object[] treePredictions
	) {
		double[] numericPredictions =
				new double[treePredictions.length];

		int count =
				0;

		double sum =
				0.0;

		for (Object prediction : treePredictions) {
			if (prediction instanceof Number number) {
				double value =
						number.doubleValue();

				numericPredictions[count++] =
						value;

				sum +=
						value;
			}
		}

		if (count == 0) {
			return 0.0;
		}

		if (AppContext.voting.equalsIgnoreCase(
				"mean"
		)) {
			return sum / count;
		}

		if (AppContext.voting.equalsIgnoreCase(
				"median"
		)) {
			Arrays.sort(
					numericPredictions,
					0,
					count
			);

			if ((count & 1) == 1) {
				return numericPredictions[
						count / 2
						];
			}

			return (
					numericPredictions[
							count / 2 - 1
							]
							+ numericPredictions[
							count / 2
							]
			) / 2.0;
		}

		throw new IllegalArgumentException(
				"Unknown voting method: "
						+ AppContext.voting
		);
	}

	private Object combineClassificationPredictions(
			Object[] treePredictions
	) {
		Map<Object, Integer> voteCounts =
				new HashMap<>();

		Object majority =
				null;

		int maximumCount =
				0;

		for (Object prediction : treePredictions) {
			int count =
					voteCounts.getOrDefault(
							prediction,
							0
					) + 1;

			voteCounts.put(
					prediction,
					count
			);

			if (count > maximumCount) {
				maximumCount =
						count;

				majority =
						prediction;
			}
		}

		return majority;
	}

	private void recordReachedLeaves(
			ProximityTree.Node[][] reachedLeaves
	) {
		for (int testIndex = 0;
			 testIndex < reachedLeaves.length;
			 testIndex++) {

			recordReachedLeaves(
					reachedLeaves,
					testIndex
			);
		}
	}

	private void recordReachedLeaves(
			ProximityTree.Node[][] reachedLeaves,
			int recordedTestIndex
	) {
		if (reachedLeaves.length == 0) {
			return;
		}

		ProximityTree.Node[] leavesForInstance =
				reachedLeaves[
						reachedLeaves.length == 1
								? 0
								: recordedTestIndex
						];

		for (ProximityTree.Node leaf
				: leavesForInstance) {

			if (leaf != null) {
				leaf.TestIndices.add(
						recordedTestIndex
				);
			}
		}
	}

	private void calculateEvaluationMetrics(
			Object[] actualLabels,
			Object[] predictedLabels
	) {
		int correct =
				0;

		int errors =
				0;

		int validLabelCount =
				0;

		for (int index = 0;
			 index < actualLabels.length;
			 index++) {

			Object actual =
					actualLabels[index];

			if (actual == null) {
				continue;
			}

			validLabelCount++;

			if (Objects.equals(
					actual,
					predictedLabels[index]
			)) {
				correct++;
			} else {
				errors++;
			}
		}

		result.correct =
				correct;

		result.errors =
				errors;

		if (validLabelCount == 0
				|| !AppContext.exists_testlabels) {

			result.score =
					Double.NaN;

			result.error_rate =
					Double.NaN;

			return;
		}

		if (AppContext.isRegression) {
			result.score =
					calculateRegressionScore(
							actualLabels,
							predictedLabels
					);
		} else {
			result.score =
					(double) correct
							/ validLabelCount;
		}

		result.error_rate =
				1.0
						- result.score;
	}

	/**
	 * Calculates R-squared while preserving actual/predicted alignment.
	 */
	private double calculateRegressionScore(
			Object[] actualLabels,
			Object[] predictedLabels
	) {
		int count =
				0;

		double sum =
				0.0;

		for (Object actual : actualLabels) {
			if (actual instanceof Number number) {
				sum +=
						number.doubleValue();

				count++;
			}
		}

		if (count == 0) {
			return Double.NaN;
		}

		double mean =
				sum / count;

		double totalSumOfSquares =
				0.0;

		double residualSumOfSquares =
				0.0;

		for (int index = 0;
			 index < actualLabels.length;
			 index++) {

			Object actual =
					actualLabels[index];

			if (!(actual instanceof Number actualNumber)) {
				continue;
			}

			Object predicted =
					predictedLabels[index];

			if (!(predicted instanceof Number predictedNumber)) {
				throw new IllegalStateException(
						"Regression prediction at index "
								+ index
								+ " is not numeric: "
								+ predicted
				);
			}

			double actualValue =
					actualNumber.doubleValue();

			double predictedValue =
					predictedNumber.doubleValue();

			double centered =
					actualValue - mean;

			double residual =
					actualValue - predictedValue;

			totalSumOfSquares +=
					centered * centered;

			residualSumOfSquares +=
					residual * residual;
		}

		if (totalSumOfSquares == 0.0) {
			return residualSumOfSquares == 0.0
					? 1.0
					: Double.NEGATIVE_INFINITY;
		}

		return 1.0
				- (
				residualSumOfSquares
						/ totalSumOfSquares
		);
	}

	private void reportTestProgress(
			int testIndex
	) {
		if (AppContext.verbosity <= 0) {
			return;
		}

		int interval =
				Math.max(
						1,
						AppContext
								.print_test_progress_for_each_instances
				);

		if (testIndex % interval == 0) {
			synchronized (System.out) {
				System.out.print(
						"*"
				);
			}
		}
	}

	/**
	 * Creates deterministic traversal randomness for one tree and test
	 * instance.
	 */
	private Random predictionRandom(
			int testIndex,
			int treeIndex
	) {
		long seed =
				forestSeed;

		seed =
				mixSeed(
						seed,
						forest_id
				);

		seed =
				mixSeed(
						seed,
						treeIndex
				);

		seed =
				mixSeed(
						seed,
						testIndex
				);

		return new Random(
				seed
		);
	}

	private static long mixSeed(
			long seed,
			int value
	) {
		long mixed =
				seed
						^ (
						0x9E3779B97F4A7C15L
								* (
								value + 1L
						)
				);

		mixed =
				(mixed ^ (mixed >>> 30))
						* 0xBF58476D1CE4E5B9L;

		mixed =
				(mixed ^ (mixed >>> 27))
						* 0x94D049BB133111EBL;

		return mixed
				^ (mixed >>> 31);
	}

	private static <T> void waitForFutures(
			List<Future<T>> futures,
			String failureMessage
	) throws Exception {

		try {
			for (Future<T> future : futures) {
				future.get();
			}
		} catch (InterruptedException e) {
			for (Future<T> future : futures) {
				future.cancel(
						true
				);
			}

			Thread.currentThread()
					.interrupt();

			throw e;
		} catch (ExecutionException e) {
			for (Future<T> future : futures) {
				future.cancel(
						true
				);
			}

			Throwable cause =
					e.getCause();

			if (cause instanceof Exception exception) {
				throw exception;
			}

			if (cause instanceof Error error) {
				throw error;
			}

			throw new IllegalStateException(
					failureMessage,
					cause
			);
		}
	}

	private static void shutdownExecutor(
			ExecutorService executor
	) {
		executor.shutdown();

		try {
			if (!executor.awaitTermination(
					30,
					TimeUnit.SECONDS
			)) {
				executor.shutdownNow();

				if (!executor.awaitTermination(
						30,
						TimeUnit.SECONDS
				)) {
					System.err.println(
							"Executor did not terminate cleanly."
					);
				}
			}
		} catch (InterruptedException e) {
			executor.shutdownNow();

			Thread.currentThread()
					.interrupt();
		}
	}

	private static Exception unwrapPredictionException(
			PredictionExecutionException failure
	) {
		Throwable cause =
				failure.getCause();

		if (cause instanceof Exception exception) {
			return exception;
		}

		return new IllegalStateException(
				"Parallel prediction failed.",
				cause
		);
	}

	/**
	 * Runtime wrapper used to propagate checked prediction failures through
	 * an IntStream.
	 */
	private static final class PredictionExecutionException
			extends RuntimeException {

		@Serial
		private static final long serialVersionUID =
				1L;

		private PredictionExecutionException(
				Throwable cause
		) {
			super(
					cause
			);
		}
	}

	public ProximityTree[] getTrees() {
		return trees;
	}

	public ProximityTree getTree(
			int index
	) {
		return trees[index];
	}

	public ProximityForestResult getResultSet() {
		return result;
	}

	public ProximityForestResult getForestStatCollection() {
		result.collateResults();

		return result;
	}

	public int getForestID() {
		return forest_id;
	}

	public void setForestID(
			int forestId
	) {
		this.forest_id =
				forestId;
	}
}