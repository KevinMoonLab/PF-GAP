package trees;

import core.AppContext;
import core.contracts.ObjectDataset;
import datasets.ListObjectDataset;
import distance.DistanceMeasure;

import java.io.IOException;
import java.io.Serializable;
//import java.util.ArrayList;
//import java.util.Collections;
//import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CompletionException;
//import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Selects and applies proximity-tree splits.
 *
 * <p>Candidate evaluation distinguishes between:</p>
 *
 * <ul>
 *     <li>
 *         Stored series representations, which may be eager values or lazy
 *         references.
 *     </li>
 *     <li>
 *         Temporarily materialized values used for distance computation.
 *     </li>
 * </ul>
 *
 * <p>For one candidate split, each candidate exemplar is materialized once.
 * Each query is materialized once, compared with every materialized
 * exemplar, and then released. Child datasets retain the original stored
 * query representation, preserving laziness.</p>
 *
 * <p>The optional parallel assignment path treats each query assignment as
 * an independent task:</p>
 *
 * <pre>
 * resolve query
 * compare against resolved candidate exemplars
 * store branch assignment
 * release resolved query
 * </pre>
 *
 * <p>Parallel workers use independent DistanceMeasure instances carrying
 * the same selected candidate parameters.</p>
 */
public class Splitter
		implements Serializable {

	private static final long serialVersionUID =
			1L;

	protected int num_children;

	/**
	 * Distance selected for the winning split.
	 */
	protected DistanceMeasure distance_measure;

	/**
	 * Stored representations of the winning exemplars.
	 *
	 * <p>These may be eager values or lazy references. Resolved candidate
	 * exemplars are intentionally not retained here.</p>
	 */
	protected Object[] exemplars;

	/**
	 * Distance used by the candidate currently being evaluated.
	 */
	protected DistanceMeasure temp_distance_measure;

	/**
	 * Stored representations of the current candidate exemplars.
	 */
	protected Object[] temp_exemplars;

	protected ListObjectDataset[] best_split;

	protected ProximityTree.Node node;

	public Splitter(
			ProximityTree.Node node
	) throws Exception {
		if (node == null) {
			throw new IllegalArgumentException(
					"Splitter requires a non-null tree node."
			);
		}

		this.node =
				node;
	}

	/**
	 * Evaluates one candidate split.
	 *
	 * <p>Candidate exemplars are selected and stored in
	 * {@code temp_exemplars}. They are subsequently materialized exactly once
	 * for the complete candidate assignment operation.</p>
	 *
	 * @param sample node dataset
	 * @param dataPerClass class-specific subsets for classification, or null
	 *                     for isolation and regression
	 * @return candidate child datasets
	 */
	public ListObjectDataset[] split_data(
			ObjectDataset sample,
			Map<Object, ListObjectDataset> dataPerClass
	) throws Exception {

		if (sample == null || sample.size() == 0) {
			return null;
		}

		if (temp_distance_measure == null) {
			throw new IllegalStateException(
					"Candidate distance measure has not been selected."
			);
		}

		ListObjectDataset[] splits =
				initializeCandidate(
						sample,
						dataPerClass
				);

		if (splits == null
				|| temp_exemplars == null
				|| temp_exemplars.length < 2) {

			return null;
		}

		/*
		 * The candidate exemplars are the only shared series values retained
		 * in materialized form for the duration of candidate evaluation.
		 */
		Object[] resolvedExemplars =
				temp_distance_measure.resolveSeriesArray(
						temp_exemplars
				);

		long candidateSeed =
				AppContext.getRand()
						.nextLong();

		int[] assignments;

		if (shouldUseParallelAssignments(sample)) {
			assignments =
					assignBranchesParallel(
							sample,
							resolvedExemplars,
							candidateSeed
					);
		} else {
			assignments =
					assignBranchesSequential(
							sample,
							resolvedExemplars,
							candidateSeed
					);
		}

		/*
		 * Child assembly remains sequential because ListObjectDataset uses
		 * ArrayList and LinkedHashMap internally and is not thread-safe.
		 *
		 * The original stored value is inserted, not the resolved query.
		 * LazySeriesRef objects therefore remain lazy in child nodes.
		 */
		for (int index = 0;
			 index < sample.size();
			 index++) {

			int branch =
					assignments[index];

			if (branch < 0 || branch >= splits.length) {
				throw new IllegalStateException(
						"Invalid branch assignment "
								+ branch
								+ " for instance "
								+ index
								+ ". Candidate branch count: "
								+ splits.length
								+ "."
				);
			}

			Object storedQuery =
					sample.get_series(
							index
					);

			splits[branch].add(
					sample.get_class(
							index
					),
					storedQuery,
					sample.get_index(
							index
					)
			);
		}

		return splits;
	}

	/**
	 * Selects candidate exemplars and creates empty child datasets.
	 */
	private ListObjectDataset[] initializeCandidate(
			ObjectDataset sample,
			Map<Object, ListObjectDataset> dataPerClass
	) {
		if (AppContext.isIsolationMode()) {
			int branches =
					Math.max(
							2,
							AppContext.isolation_num_branches
					);

			branches =
					Math.min(
							branches,
							sample.size()
					);

			return initializeUnsupervisedCandidate(
					sample,
					branches
			);
		}

		if (AppContext.isRegressionMode()) {
			int branches =
					Math.max(
							2,
							AppContext.regression_num_branches
					);

			branches =
					Math.min(
							branches,
							sample.size()
					);

			return initializeUnsupervisedCandidate(
					sample,
					branches
			);
		}

		return initializeClassificationCandidate(
				sample,
				dataPerClass
		);
	}

	/**
	 * Initializes an isolation or regression candidate.
	 */
	private ListObjectDataset[] initializeUnsupervisedCandidate(
			ObjectDataset sample,
			int branches
	) {
		if (branches < 2) {
			return null;
		}

		temp_exemplars =
				new Object[branches];

		ListObjectDataset[] splits =
				createEmptySplits(
						sample.size(),
						branches
				);

		int[] exemplarIndices =
				sampleDistinctIndices(
						sample.size(),
						branches
				);

		for (int branch = 0;
			 branch < branches;
			 branch++) {

			temp_exemplars[branch] =
					sample.get_series(
							exemplarIndices[branch]
					);
		}

		return splits;
	}

	/**
	 * Initializes a classification candidate containing one randomly selected
	 * exemplar from each class.
	 */
	private ListObjectDataset[] initializeClassificationCandidate(
			ObjectDataset sample,
			Map<Object, ListObjectDataset> dataPerClass
	) {
		if (dataPerClass == null
				|| dataPerClass.size() < 2) {

			return null;
		}

		int branches =
				dataPerClass.size();

		temp_exemplars =
				new Object[branches];

		ListObjectDataset[] splits =
				createEmptySplits(
						sample.size(),
						branches
				);

		int branch =
				0;

		for (Map.Entry<Object, ListObjectDataset> entry
				: dataPerClass.entrySet()) {

			ListObjectDataset classData =
					entry.getValue();

			if (classData == null
					|| classData.size() == 0) {

				return null;
			}

			int selectedIndex =
					AppContext.getRand()
							.nextInt(
									classData.size()
							);

			temp_exemplars[branch] =
					classData.get_series(
							selectedIndex
					);

			branch++;
		}

		return splits;
	}

	/**
	 * Creates empty child datasets with a reasonable initial capacity.
	 *
	 * <p>The previous implementation allocated parentSize slots in every
	 * branch. With B branches, that reserved roughly B times the required
	 * capacity in each internal list. The expected branch size is a better
	 * initial hint and the lists can still grow when needed.</p>
	 */
	private ListObjectDataset[] createEmptySplits(
			int parentSize,
			int branches
	) {
		ListObjectDataset[] splits =
				new ListObjectDataset[branches];

		int expectedBranchSize =
				Math.max(
						1,
						parentSize / branches
				);

		for (int branch = 0;
			 branch < branches;
			 branch++) {

			splits[branch] =
					new ListObjectDataset(
							expectedBranchSize
					);
		}

		return splits;
	}

	/**
	 * Sequential bounded-memory branch assignment.
	 *
	 * <p>Only one query and the candidate exemplars are materialized at a
	 * time.</p>
	 */
	private int[] assignBranchesSequential(
			ObjectDataset sample,
			Object[] resolvedExemplars,
			long candidateSeed
	) throws IOException, InterruptedException {

		int[] assignments =
				new int[sample.size()];

		for (int index = 0;
			 index < sample.size();
			 index++) {

			Object storedQuery =
					sample.get_series(
							index
					);

			int matchingExemplar =
					findStoredExemplarMatch(
							storedQuery,
							temp_exemplars
					);

			if (matchingExemplar >= 0
					&& AppContext
					.config_skip_distance_when_exemplar_matches_query) {

				assignments[index] =
						matchingExemplar;

				continue;
			}

			Object resolvedQuery =
					temp_distance_measure.resolveSeries(
							storedQuery
					);

			Random queryRandom =
					new Random(
							mixSeed(
									candidateSeed,
									stableInstanceIdentity(
											sample,
											index
									)
							)
					);

			assignments[index] =
					temp_distance_measure.findClosestResolvedNode(
							resolvedQuery,
							resolvedExemplars,
							queryRandom
					);
		}

		return assignments;
	}

	/**
	 * Parallel bounded-memory branch assignment.
	 *
	 * <p>Each active worker materializes at most one query at a time. The
	 * candidate exemplar array is materialized once and shared read-only among
	 * workers.</p>
	 *
	 * <p>Each worker thread receives an independent DistanceMeasure copy. This
	 * supports built-in and descriptor-backed distances without requiring
	 * their implementation objects to be thread-safe.</p>
	 */
	private int[] assignBranchesParallel(
			ObjectDataset sample,
			Object[] resolvedExemplars,
			long candidateSeed
	) throws Exception {

		int instanceCount =
				sample.size();

		int[] assignments =
				new int[instanceCount];

		/*
		 * Take a shallow snapshot of the stored objects. For lazy data these
		 * remain small immutable LazySeriesRef objects. The snapshot avoids
		 * repeatedly navigating a mutable dataset list from worker threads.
		 */
		Object[] storedQueries =
				sample._internal_data_list()
						.toArray();

		if (storedQueries.length != instanceCount) {
			throw new IllegalStateException(
					"Dataset size changed while preparing parallel split "
							+ "assignment."
			);
		}

		ThreadLocal<DistanceMeasure> workerDistances =
				ThreadLocal.withInitial(
						() -> createWorkerDistanceUnchecked(
								temp_distance_measure
						)
				);

		try {
			IntStream.range(
							0,
							instanceCount
					)
					.parallel()
					.forEach(index -> {
						try {
							Object storedQuery =
									storedQueries[index];

							int matchingExemplar =
									findStoredExemplarMatch(
											storedQuery,
											temp_exemplars
									);

							if (matchingExemplar >= 0
									&& AppContext
									.config_skip_distance_when_exemplar_matches_query) {

								assignments[index] =
										matchingExemplar;

								return;
							}

							DistanceMeasure workerDistance =
									workerDistances.get();

							Object resolvedQuery =
									workerDistance.resolveSeries(
											storedQuery
									);

							Random queryRandom =
									new Random(
											mixSeed(
													candidateSeed,
													stableInstanceIdentity(
															sample,
															index
													)
											)
									);

							assignments[index] =
									workerDistance.findClosestResolvedNode(
											resolvedQuery,
											resolvedExemplars,
											queryRandom
									);
						} catch (IOException
								| InterruptedException e) {

							if (e instanceof InterruptedException) {
								Thread.currentThread()
										.interrupt();
							}

							throw new CompletionException(
									e
							);
						} catch (RuntimeException e) {
							throw e;
						} catch (Exception e) {
							throw new CompletionException(
									e
							);
						}
					});
		} catch (CompletionException e) {
			Throwable cause =
					unwrapCompletionException(
							e
					);

			if (cause instanceof InterruptedException interrupted) {
				throw interrupted;
			}

			if (cause instanceof IOException ioException) {
				throw ioException;
			}

			if (cause instanceof RuntimeException runtimeException) {
				throw runtimeException;
			}

			if (cause instanceof Error error) {
				throw error;
			}

			throw new Exception(
					"Parallel candidate split assignment failed.",
					cause
			);
		} finally {
			workerDistances.remove();
		}

		return assignments;
	}

	/**
	 * Constructs a worker-local distance wrapper.
	 */
	private static DistanceMeasure createWorkerDistanceUnchecked(
			DistanceMeasure source
	) {
		try {
			return source.copyForEvaluation();
		} catch (Exception e) {
			throw new CompletionException(
					new IllegalStateException(
							"Could not create a worker-local evaluator for "
									+ source
									+ ".",
							e
					)
			);
		}
	}

	/**
	 * Determines whether this candidate should use parallel assignments.
	 *
	 * <p>Parallel tree construction and parallel within-split assignments are
	 * initially treated as mutually exclusive to avoid processor
	 * oversubscription.</p>
	 */
	private boolean shouldUseParallelAssignments(
			ObjectDataset sample
	) {
		if (!AppContext.parallel_split_assignments) {
			return false;
		}

		if (AppContext.parallelTrees) {
			throw new IllegalArgumentException(
					"parallel_trees and parallel_split_assignments cannot "
							+ "currently be enabled together. Choose the "
							+ "parallel strategy appropriate for the current "
							+ "workload."
			);
		}

		return sample.size()
				>= Math.max(
				1,
				AppContext.parallel_split_assignment_threshold
		);
	}

	/**
	 * Finds whether the stored query is itself one of the stored candidate
	 * exemplars.
	 *
	 * <p>This check must happen before independent lazy materialization.
	 * Resolving the same LazySeriesRef twice usually produces distinct array
	 * instances, so resolved reference identity would no longer detect the
	 * exemplar match.</p>
	 */
	private static int findStoredExemplarMatch(
			Object storedQuery,
			Object[] storedExemplars
	) {
		for (int branch = 0;
			 branch < storedExemplars.length;
			 branch++) {

			if (storedExemplars[branch]
					== storedQuery) {

				return branch;
			}
		}

		return -1;
	}

	/**
	 * Returns a deterministic identity for parallel tie-breaking.
	 */
	private static int stableInstanceIdentity(
			ObjectDataset sample,
			int localIndex
	) {
		Integer internalIndex =
				sample.get_index(
						localIndex
				);

		return internalIndex == null
				? localIndex
				: internalIndex;
	}

	/**
	 * Derives a deterministic per-query seed from the candidate seed and
	 * stable instance identity.
	 */
	private static long mixSeed(
			long seed,
			int instanceIdentity
	) {
		long value =
				seed
						^ (
						0x9E3779B97F4A7C15L
								* (
								instanceIdentity + 1L
						)
				);

		value =
				(value ^ (value >>> 30))
						* 0xBF58476D1CE4E5B9L;

		value =
				(value ^ (value >>> 27))
						* 0x94D049BB133111EBL;

		return value
				^ (value >>> 31);
	}

	private static Throwable unwrapCompletionException(
			Throwable failure
	) {
		Throwable current =
				failure;

		while (current instanceof CompletionException
				&& current.getCause() != null) {

			current =
					current.getCause();
		}

		return current;
	}

	/**
	 * Compatibility method accepting stored or eager inputs.
	 */
	public int find_closest_branch(
			Object query,
			DistanceMeasure distanceMeasure,
			Object[] candidateExemplars
	) throws Exception {

		return distanceMeasure.find_closest_node(
				query,
				candidateExemplars,
				true,
				node.tree.getDistance_file()
		);
	}

	/**
	 * Compatibility branch-selection method for the winning splitter.
	 *
	 * <p>This method resolves the query and node exemplars for this one call.
	 * A later prediction refactor should resolve each prediction query once at
	 * the forest boundary and call findClosestBranchResolved().</p>
	 */
	public int find_closest_branch(
			Object query
	) throws Exception {

		return distance_measure.find_closest_node(
				query,
				exemplars,
				true,
				node.tree.getDistance_file()
		);
	}

	/**
	 * Optimized branch-selection method for a query that has already been
	 * materialized.
	 *
	 * <p>The node exemplars are resolved temporarily for this traversal step.
	 * They are not retained after the method returns.</p>
	 */
	public int findClosestBranchResolved(
			Object resolvedQuery,
			Random random
	) throws IOException, InterruptedException {

		if (distance_measure == null) {
			throw new IllegalStateException(
					"Splitter has no selected distance measure."
			);
		}

		if (exemplars == null || exemplars.length == 0) {
			throw new IllegalStateException(
					"Splitter has no selected exemplars."
			);
		}

		Object[] resolvedExemplars =
				distance_measure.resolveSeriesArray(
						exemplars
				);

		return distance_measure.findClosestResolvedNode(
				resolvedQuery,
				resolvedExemplars,
				random
		);
	}

	/**
	 * Assigns every instance in a dataset to a branch of the selected split.
	 *
	 * <p>The selected node exemplars are materialized once for the complete
	 * batch. Each query is materialized once, compared with the materialized
	 * exemplars, and then released.</p>
	 *
	 * <p>The dataset itself is not modified, and lazy query references remain
	 * stored in the dataset.</p>
	 *
	 * @param data stored eager or lazy query representations
	 * @param batchSeed deterministic seed for tie-breaking
	 * @return one branch index per dataset instance
	 */
	public int[] findClosestBranches(
			ObjectDataset data,
			long batchSeed
	) throws IOException, InterruptedException {

		if (data == null) {
			throw new IllegalArgumentException(
					"Cannot route a null dataset."
			);
		}

		if (distance_measure == null) {
			throw new IllegalStateException(
					"Splitter has no selected distance measure."
			);
		}

		if (exemplars == null || exemplars.length == 0) {
			throw new IllegalStateException(
					"Splitter has no selected exemplars."
			);
		}

		Object[] resolvedExemplars =
				distance_measure.resolveSeriesArray(
						exemplars
				);

		int[] assignments =
				new int[data.size()];

		for (int index = 0;
			 index < data.size();
			 index++) {

			Object storedQuery =
					data.get_series(
							index
					);

			int matchingExemplar =
					findStoredExemplarMatch(
							storedQuery,
							exemplars
					);

			if (matchingExemplar >= 0
					&& AppContext
					.config_skip_distance_when_exemplar_matches_query) {

				assignments[index] =
						matchingExemplar;

				continue;
			}

			Object resolvedQuery =
					distance_measure.resolveSeries(
							storedQuery
					);

			Random queryRandom =
					new Random(
							mixSeed(
									batchSeed,
									stableInstanceIdentity(
											data,
											index
									)
							)
					);

			assignments[index] =
					distance_measure.findClosestResolvedNode(
							resolvedQuery,
							resolvedExemplars,
							queryRandom
					);
		}

		return assignments;
	}

	public ObjectDataset[] getBestSplits() {
		return best_split;
	}

	/**
	 * Searches candidate splits and retains the lowest-purity valid split.
	 */
	public ListObjectDataset[] find_best_split(
			ObjectDataset data
	) throws Exception {

		best_split =
				null;

		distance_measure =
				null;

		exemplars =
				null;

		temp_distance_measure =
				null;

		temp_exemplars =
				null;

		if (data == null || data.size() < 2) {
			return null;
		}

		Map<Object, ListObjectDataset> dataPerClass =
				AppContext.isRegressionMode()
						|| AppContext.isIsolationMode()
						? null
						: data.split_classes();

		double bestWeightedPurity =
				Double.POSITIVE_INFINITY;

		int parentSize =
				data.size();

		for (int candidate = 0;
			 candidate
					 < AppContext.num_candidates_per_split;
			 candidate++) {

			temp_distance_measure =
					selectDistanceMeasure(
							data
					);

			ListObjectDataset[] candidateSplits =
					split_data(
							data,
							dataPerClass
					);

			if (candidateSplits == null) {
				continue;
			}

			double weightedPurity =
					weighted_purity(
							parentSize,
							candidateSplits
					);

			if (!Double.isFinite(weightedPurity)) {
				continue;
			}

			if (weightedPurity
					< bestWeightedPurity) {

				bestWeightedPurity =
						weightedPurity;

				best_split =
						candidateSplits;

				distance_measure =
						temp_distance_measure;

				/*
				 * Retain original representations. Lazy exemplars therefore
				 * remain lazy after the split search completes.
				 */
				exemplars =
						temp_exemplars.clone();
			}
		}

		/*
		 * Candidate-only references should not remain reachable after the
		 * split search.
		 */
		temp_distance_measure =
				null;

		temp_exemplars =
				null;

		if (best_split == null) {
			return null;
		}

		num_children =
				best_split.length;

		return best_split;
	}

	private DistanceMeasure selectDistanceMeasure(
			ObjectDataset data
	) throws Exception {

		DistanceMeasure selected;

		if (node.tree.getChosen_distances().length == 0) {
			if (AppContext.random_dm_per_node) {
				int selectedIndex =
						AppContext.getRand()
								.nextInt(
										AppContext
												.enabled_distance_measures
												.length
								);

				selected =
						new DistanceMeasure(
								AppContext
										.enabled_distance_measures[
										selectedIndex
										],
								AppContext.Descriptors.get(
										selectedIndex
								)
						);
			} else {
				selected =
						node.tree.tree_distance_measure;
			}
		} else {
			if (AppContext.random_dm_per_node) {
				int selectedIndex =
						AppContext.getRand()
								.nextInt(
										node.tree
												.getChosen_distances()
												.length
								);

				selected =
						new DistanceMeasure(
								node.tree
										.getChosen_distances()[
										selectedIndex
										],
								AppContext.Descriptors.get(
										selectedIndex
								)
						);
			} else {
				selected =
						node.tree.tree_distance_measure;
			}
		}

		selected.select_random_params(
				data,
				AppContext.getRand()
		);

		return selected;
	}

	public double weighted_purity(
			int parentSize,
			ListObjectDataset[] splits
	) {
		return purity.SplitScorer.compute(
				AppContext.purity_measure,
				parentSize,
				splits
		);
	}

	/**
	 * Samples distinct indices without allocating and shuffling N boxed
	 * Integer objects.
	 *
	 * <p>This performs a partial Fisher-Yates shuffle using an int array.</p>
	 */
	private int[] sampleDistinctIndices(
			int sampleSize,
			int count
	) {
		if (sampleSize < 0) {
			throw new IllegalArgumentException(
					"sampleSize cannot be negative."
			);
		}

		count =
				Math.min(
						count,
						sampleSize
				);

		if (count < 0) {
			throw new IllegalArgumentException(
					"count cannot be negative."
			);
		}

		int[] indices =
				new int[sampleSize];

		for (int index = 0;
			 index < sampleSize;
			 index++) {

			indices[index] =
					index;
		}

		Random random =
				AppContext.getRand();

		for (int selectedIndex = 0;
			 selectedIndex < count;
			 selectedIndex++) {

			int swapIndex =
					selectedIndex
							+ random.nextInt(
							sampleSize
									- selectedIndex
					);

			int temporary =
					indices[selectedIndex];

			indices[selectedIndex] =
					indices[swapIndex];

			indices[swapIndex] =
					temporary;
		}

		int[] selected =
				new int[count];

		System.arraycopy(
				indices,
				0,
				selected,
				0,
				count
		);

		return selected;
	}
}