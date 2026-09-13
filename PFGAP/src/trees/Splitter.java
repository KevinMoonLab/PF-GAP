package trees;

import core.AppContext;
import core.contracts.ObjectDataset;
import core.parallel.ParallelRuntime;
import core.random.SeedMixer;
import datasets.ListObjectDataset;
import distance.DistanceMeasure;
import distance.MEASURE;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

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


	protected ListObjectDataset[] best_split;

	protected ProximityTree.Node node;

	/**
	 * Purpose value used to derive deterministic node-level dimension-selection
	 * randomness independently from candidate, exemplar, and tie-breaking
	 * randomness.
	 */
	private static final int DIMENSION_SELECTION_PURPOSE =
			0x44494D;
	private static final long CANDIDATE_PURPOSE = 0x43414E4449444154L;
	private static final long ASSIGNMENT_PURPOSE = 0x41535349474E4D54L;
	private static final int MINIMUM_PARALLEL_CANDIDATE_COUNT = 2;
	private static final int MINIMUM_PARALLEL_CANDIDATE_SAMPLE_SIZE = 16;
	private static final int MINIMUM_PARALLEL_ASSIGNMENT_SIZE = 64;
	private static final int MINIMUM_ASSIGNMENT_RANGE_SIZE = 16;

	/**
	 * Below this dimensionality, allocating a primitive permutation array is
	 * inexpensive enough that partial Fisher-Yates sampling is preferred.
	 */
	private static final int DENSE_DIMENSION_SAMPLING_THRESHOLD =
			4096;

	/**
	 * Selected realized dimensions for this split.
	 *
	 * <p>A null value means that every available dimension is used. Otherwise,
	 * the array contains distinct, ascending dimension indices.</p>
	 *
	 * <p>The selection is generated once per node and shared by every candidate
	 * evaluated at that node. It is serialized with the winning splitter so that
	 * prediction and loaded-model evaluation reproduce the trained routing
	 * behavior.</p>
	 */
	private int[] selectedDimensions;

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
	 * <p>Candidate exemplars are candidate-local and are materialized exactly
	 * once for the complete candidate assignment operation.</p>
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
		try (ParallelRuntime runtime = new ParallelRuntime(1)) {
			return split_data(sample, dataPerClass, runtime);
		}
	}

	public ListObjectDataset[] split_data(
			ObjectDataset sample,
			Map<Object, ListObjectDataset> dataPerClass,
			ParallelRuntime runtime
	) throws Exception {
		Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");
		if (sample == null || sample.size() == 0) {
			return null;
		}
		if (selectedDimensions == null) {
			initializeNodeDimensionSelection(sample);
		}
		CandidateSplitResult result = evaluateCandidate(
				0, sample, dataPerClass, runtime, false
		);
		return result == null ? null : result.splits();
	}

	private CandidateSplitResult evaluateCandidate(
			int candidateIndex,
			ObjectDataset sample,
			Map<Object, ListObjectDataset> dataPerClass,
			ParallelRuntime runtime,
			boolean candidatesAreParallel
	) throws Exception {
		long candidateSeed = SeedMixer.candidate(
				node.tree.getTreeSeed(),
				node.getPathIdentity(),
				CANDIDATE_PURPOSE,
				candidateIndex
		);
		Random random = new Random(candidateSeed);
		DistanceMeasure candidateDistance = selectDistanceMeasure(sample, random);
		CandidateInitialization initialization = initializeCandidate(
				sample, dataPerClass, random
		);
		if (initialization == null || initialization.exemplars().length < 2) {
			return null;
		}
		Object[] storedExemplars = initialization.exemplars();
		Object[] resolvedExemplars = candidateDistance.resolveSeriesArray(storedExemplars);
		long assignmentSeed = SeedMixer.derive(candidateSeed, ASSIGNMENT_PURPOSE);
		int[] assignments;
		if (!candidatesAreParallel && shouldUseParallelAssignments(sample, runtime)) {
			assignments = assignBranchesParallel(
					sample, storedExemplars, candidateDistance, resolvedExemplars,
					assignmentSeed, runtime
			);
		} else {
			assignments = assignBranchesSequential(
					sample, storedExemplars, candidateDistance, resolvedExemplars,
					assignmentSeed
			);
		}
		ListObjectDataset[] splits = initialization.splits();
		assembleSplits(sample, assignments, splits);
		double weightedPurity = weighted_purity(sample.size(), splits);
		if (!Double.isFinite(weightedPurity)) {
			return null;
		}
		return new CandidateSplitResult(
				candidateIndex, candidateDistance, storedExemplars, splits, weightedPurity
		);
	}

	private CandidateInitialization initializeCandidate(
			ObjectDataset sample,
			Map<Object, ListObjectDataset> dataPerClass,
			Random random
	) {
		if (AppContext.isIsolationMode()) {
			int branches = Math.min(Math.max(2, AppContext.isolation_num_branches), sample.size());
			return initializeUnsupervisedCandidate(sample, branches, random);
		}
		if (AppContext.isRegressionMode()) {
			int branches = Math.min(Math.max(2, AppContext.regression_num_branches), sample.size());
			return initializeUnsupervisedCandidate(sample, branches, random);
		}
		return initializeClassificationCandidate(sample, dataPerClass, random);
	}

	private CandidateInitialization initializeUnsupervisedCandidate(
			ObjectDataset sample, int branches, Random random
	) {
		if (branches < 2) {
			return null;
		}
		Object[] candidateExemplars = new Object[branches];
		int[] indices = sampleDistinctIndices(sample.size(), branches, random);
		for (int branch = 0; branch < branches; branch++) {
			candidateExemplars[branch] = sample.get_series(indices[branch]);
		}
		return new CandidateInitialization(
				candidateExemplars, createEmptySplits(sample.size(), branches)
		);
	}

	private CandidateInitialization initializeClassificationCandidate(
			ObjectDataset sample,
			Map<Object, ListObjectDataset> dataPerClass,
			Random random
	) {
		if (dataPerClass == null || dataPerClass.size() < 2) {
			return null;
		}
		Object[] candidateExemplars = new Object[dataPerClass.size()];
		int branch = 0;
		for (Map.Entry<Object, ListObjectDataset> entry : dataPerClass.entrySet()) {
			ListObjectDataset classData = entry.getValue();
			if (classData == null || classData.size() == 0) {
				return null;
			}
			candidateExemplars[branch++] = classData.get_series(random.nextInt(classData.size()));
		}
		return new CandidateInitialization(
				candidateExemplars, createEmptySplits(sample.size(), candidateExemplars.length)
		);
	}

	private static void assembleSplits(
			ObjectDataset sample, int[] assignments, ListObjectDataset[] splits
	) {
		for (int index = 0; index < sample.size(); index++) {
			int branch = assignments[index];
			if (branch < 0 || branch >= splits.length) {
				throw new IllegalStateException(
						"Invalid branch assignment " + branch + " for instance " + index
								+ ". Candidate branch count: " + splits.length + "."
				);
			}
			splits[branch].add(
					sample.get_class(index), sample.get_series(index), sample.get_index(index)
			);
		}
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
			Object[] storedExemplars,
			DistanceMeasure evaluator,
			Object[] resolvedExemplars,
			long assignmentSeed
	) throws IOException, InterruptedException {
		int[] assignments = new int[sample.size()];
		for (int index = 0; index < sample.size(); index++) {
			assignments[index] = assignOne(
					sample, index, sample.get_series(index), storedExemplars, evaluator,
					resolvedExemplars, assignmentSeed
			);
		}
		return assignments;
	}

	private int[] assignBranchesParallel(
			ObjectDataset sample,
			Object[] storedExemplars,
			DistanceMeasure candidateDistance,
			Object[] resolvedExemplars,
			long assignmentSeed,
			ParallelRuntime runtime
	) throws Exception {
		int count = sample.size();
		int[] assignments = new int[count];
		Object[] storedQueries = sample._internal_data_list().toArray();
		if (storedQueries.length != count) {
			throw new IllegalStateException("Dataset size changed during split assignment.");
		}
		runtime.forRanges(
				0, count, MINIMUM_ASSIGNMENT_RANGE_SIZE,
				(start, end) -> {
					DistanceMeasure workerDistance = candidateDistance.copyForEvaluation();
					for (int index = start; index < end; index++) {
						assignments[index] = assignOne(
								sample, index, storedQueries[index], storedExemplars, workerDistance,
								resolvedExemplars, assignmentSeed
						);
					}
				}
		);
		return assignments;
	}

	private int assignOne(
			ObjectDataset sample, int index, Object storedQuery,
			Object[] storedExemplars, DistanceMeasure evaluator,
			Object[] resolvedExemplars, long assignmentSeed
	) throws IOException, InterruptedException {
		int exemplarMatch = findStoredExemplarMatch(storedQuery, storedExemplars);
		if (exemplarMatch >= 0 && AppContext.config_skip_distance_when_exemplar_matches_query) {
			return exemplarMatch;
		}
		Object resolvedQuery = evaluator.resolveSeries(storedQuery);
		Random random = new Random(
				SeedMixer.instance(assignmentSeed, stableInstanceIdentity(sample, index))
		);
		return evaluator.findClosestResolvedNode(
				resolvedQuery, resolvedExemplars, random, selectedDimensions
		);
	}

	private boolean shouldUseParallelAssignments(
			ObjectDataset sample, ParallelRuntime runtime
	) {
		return runtime.isParallel()
				&& sample.size() >= MINIMUM_PARALLEL_ASSIGNMENT_SIZE;
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
	 * Compatibility branch-selection method using an explicitly supplied
	 * distance measure and exemplar array.
	 *
	 * <p>The splitter's stored node-level dimension selection is applied to the
	 * supplied distance calculation.</p>
	 */
	public int find_closest_branch(
			Object query,
			DistanceMeasure distanceMeasure,
			Object[] candidateExemplars
	) throws Exception {

		if (distanceMeasure == null) {
			throw new IllegalArgumentException(
					"Distance measure cannot be null."
			);
		}

		if (candidateExemplars == null
				|| candidateExemplars.length == 0) {

			throw new IllegalArgumentException(
					"At least one candidate exemplar is required."
			);
		}

		return distanceMeasure.find_closest_node(
				query,
				candidateExemplars,
				true,
				selectedDimensions,
				node.tree.getDistance_file()
		);
	}

	/**
	 * Compatibility branch-selection method for the winning splitter.
	 *
	 * <p>The query and stored exemplars are resolved by DistanceMeasure, and the
	 * node's trained dimension subset is applied to every comparison.</p>
	 */
	public int find_closest_branch(
			Object query
	) throws Exception {

		if (distance_measure == null) {
			throw new IllegalStateException(
					"Splitter has no selected distance measure."
			);
		}

		if (exemplars == null
				|| exemplars.length == 0) {

			throw new IllegalStateException(
					"Splitter has no selected exemplars."
			);
		}

		return distance_measure.find_closest_node(
				query,
				exemplars,
				true,
				selectedDimensions,
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
				random,
				selectedDimensions
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
							SeedMixer.instance(batchSeed, stableInstanceIdentity(
											data,
											index
									))
					);

			assignments[index] =
					distance_measure.findClosestResolvedNode(
							resolvedQuery,
							resolvedExemplars,
							queryRandom,
							selectedDimensions
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
	public ListObjectDataset[] find_best_split(ObjectDataset data) throws Exception {
		try (ParallelRuntime runtime = new ParallelRuntime(1)) {
			return find_best_split(data, runtime);
		}
	}

	public ListObjectDataset[] find_best_split(
			ObjectDataset data, ParallelRuntime runtime
	) throws Exception {
		Objects.requireNonNull(runtime, "ParallelRuntime cannot be null.");
		best_split = null;
		distance_measure = null;
		exemplars = null;
		selectedDimensions = null;
		num_children = 0;
		if (data == null || data.size() < 2) {
			return null;
		}
		initializeNodeDimensionSelection(data);
		Map<Object, ListObjectDataset> dataPerClass =
				AppContext.isRegressionMode() || AppContext.isIsolationMode()
						? null : data.split_classes();
		int count = AppContext.num_candidates_per_split;
		if (count < 1) {
			return null;
		}
		CandidateSplitResult[] results = new CandidateSplitResult[count];
		boolean parallelCandidates = runtime.isParallel()
				&& count >= MINIMUM_PARALLEL_CANDIDATE_COUNT
				&& data.size() >= MINIMUM_PARALLEL_CANDIDATE_SAMPLE_SIZE;
		if (parallelCandidates) {
			runtime.forRange(
					0, count, 1,
					i -> results[i] = evaluateCandidate(i, data, dataPerClass, runtime, true)
			);
		} else {
			for (int i = 0; i < count; i++) {
				results[i] = evaluateCandidate(i, data, dataPerClass, runtime, false);
			}
		}
		CandidateSplitResult winner = null;
		for (CandidateSplitResult candidate : results) {
			if (candidate == null) {
				continue;
			}
			if (winner == null
					|| candidate.weightedPurity() < winner.weightedPurity()
					|| (Double.compare(candidate.weightedPurity(), winner.weightedPurity()) == 0
					&& candidate.candidateIndex() < winner.candidateIndex())) {
				winner = candidate;
			}
		}
		if (winner == null) {
			return null;
		}
		best_split = winner.splits();
		distance_measure = winner.distanceMeasure();
		exemplars = winner.exemplars().clone();
		num_children = best_split.length;
		return best_split;
	}

	/**
	 * Creates an independent distance evaluator for one candidate split and
	 * selects that candidate's random distance parameters.
	 *
	 * <p>When distance selection occurs once per tree, the tree-level distance
	 * object identifies the selected distance type but must not itself be stored
	 * in a node splitter. Its parameter fields are mutable, so retaining the
	 * shared tree object would allow later nodes to overwrite parameters selected
	 * by earlier nodes.</p>
	 *
	 * <p>Every candidate therefore receives its own DistanceMeasure instance.
	 * The winning candidate's independent instance is retained by this splitter.</p>
	 */
	private DistanceMeasure selectDistanceMeasure(
			ObjectDataset data, Random random
	) throws Exception {

		DistanceMeasure selected;

		if (node.tree.getChosen_distances().length == 0) {
			if (AppContext.random_dm_per_node) {
				int selectedIndex =
						random
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
				if (node.tree.tree_distance_measure == null) {
					throw new IllegalStateException(
							"The tree-level distance measure has not been "
									+ "initialized."
					);
				}

				/*
				 * Do not return the mutable tree-level object directly.
				 * Each candidate and winning splitter must own independent
				 * parameter state.
				 */
				selected =
						node.tree.tree_distance_measure
								.copyForEvaluation();
			}
		} else {
			if (AppContext.random_dm_per_node) {
				MEASURE[] chosenDistances =
						node.tree.getChosen_distances();

				int selectedIndex =
						random
								.nextInt(
										chosenDistances.length
								);

				selected =
						new DistanceMeasure(
								chosenDistances[selectedIndex],
								AppContext.Descriptors.get(
										selectedIndex
								)
						);
			} else {
				if (node.tree.tree_distance_measure == null) {
					throw new IllegalStateException(
							"The tree-level distance measure has not been "
									+ "initialized."
					);
				}

				/*
				 * The tree chooses the distance type once, but each candidate
				 * still requires independent randomized parameter state.
				 */
				selected =
						node.tree.tree_distance_measure
								.copyForEvaluation();
			}
		}

		selected.select_random_params(
				data,
				random
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
			int sampleSize, int count, Random random
	) {
		Objects.requireNonNull(random, "Random cannot be null.");
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

	/**
	 * Selects the realized dimensions shared by every candidate at this node.
	 *
	 * <p>When dimension subsampling is disabled, the configured strategy is ALL,
	 * or the requested count includes every available dimension, the stored
	 * selection remains null. Null is the all-dimensions fast path.</p>
	 *
	 * <p>For eager data, resolving the representative instance returns the same
	 * object. For lazy data, exactly one representative instance is materialized
	 * to determine the realized dimensionality.</p>
	 */
	private void initializeNodeDimensionSelection(
			ObjectDataset data
	) throws Exception {

		if (!AppContext.subsample_dimensions
				|| AppContext.dimension_selection_strategy
				== DimensionSelectionStrategy.ALL) {

			selectedDimensions =
					null;

			return;
		}

		Object storedRepresentative =
				data.get_series(
						0
				);

		if (storedRepresentative == null) {
			throw new IllegalStateException(
					"Cannot determine selectable dimensionality from a null "
							+ "node instance."
			);
		}

		/*
		 * Use a short-lived wrapper only to apply the established eager/lazy
		 * materialization contract. No distance is computed.
		 *
		 * This avoids depending on whichever candidate distance is selected
		 * later, since all candidates must share the same node-level subset.
		 */
		Object resolvedRepresentative =
				resolveDimensionRepresentative(
						storedRepresentative
				);

		int dimensionCount =
				selectableDimensionCountOf(
						resolvedRepresentative
				);

		int selectedCount =
				resolveSelectedDimensionCount(
						dimensionCount
				);

		if (selectedCount >= dimensionCount) {
			selectedDimensions =
					null;

			return;
		}

		Random selectionRandom =
				new Random(
						node.tree.deriveNodeSeed(node.getPathIdentity(), DIMENSION_SELECTION_PURPOSE)
				);

		selectedDimensions =
				sampleDistinctDimensions(
						dimensionCount,
						selectedCount,
						selectionRandom
				);
	}

	/**
	 * Resolves one representative instance for dimensionality discovery.
	 */
	private static Object resolveDimensionRepresentative(
			Object storedRepresentative
	) {
		if (storedRepresentative
				instanceof datasets.readers.lazy.LazySeriesRef reference) {

			return AppContext
					.getLazySeriesReader(
							reference.getReaderKey()
					)
					.read(
							reference
					);
		}

		return storedRepresentative;
	}

	/**
	 * Returns the number of dimensions eligible for node-level random selection.
	 *
	 * <p>For one-dimensional arrays, each position is interpreted as a tabular
	 * feature. For two-dimensional arrays, the outer position is interpreted as
	 * a multivariate channel.</p>
	 *
	 * <p>Univariate time-point subsampling is not distinguished automatically.
	 * Users should not enable dimension subsampling for ordinary univariate time
	 * series in this initial implementation.</p>
	 */
	private static int selectableDimensionCountOf(
			Object resolvedInstance
	) {
		int dimensionCount;

		if (resolvedInstance instanceof double[] values) {
			dimensionCount =
					values.length;
		} else if (resolvedInstance instanceof Double[] values) {
			dimensionCount =
					values.length;
		} else if (resolvedInstance instanceof double[][] values) {
			dimensionCount =
					values.length;
		} else if (resolvedInstance instanceof Double[][] values) {
			dimensionCount =
					values.length;
		} else if (resolvedInstance instanceof Object[][] values) {
			dimensionCount =
					values.length;
		} else if (resolvedInstance instanceof Object[] values) {
			dimensionCount = values.length;
		} else {
			throw new UnsupportedOperationException(
					"Node-level dimension subsampling is unsupported for "
							+ "instance representation "
							+ resolvedInstance.getClass().getTypeName()
							+ ". Expected double[], Double[], double[][], "
							+ "Double[][], Object[][], or Object[]."
			);
		}

		if (dimensionCount < 1) {
			throw new IllegalArgumentException(
					"Dimension subsampling requires at least one available "
							+ "dimension."
			);
		}

		return dimensionCount;
	}

	/**
	 * Resolves the configured number of dimensions to select.
	 */
	private static int resolveSelectedDimensionCount(
			int dimensionCount
	) {
		DimensionSelectionStrategy strategy =
				AppContext.dimension_selection_strategy;

		if (strategy == null) {
			throw new IllegalStateException(
					"dimension_selection_strategy cannot be null when "
							+ "dimension subsampling is enabled."
			);
		}

		int selectedCount;

		switch (strategy) {
			case ALL:
				selectedCount =
						dimensionCount;

				break;

			case SQRT:
				selectedCount =
						(int) Math.ceil(
								Math.sqrt(
										dimensionCount
								)
						);

				break;

			case LOG2:
				/*
				 * Computes floor(log2(d)) + 1 using integer operations.
				 */
				selectedCount =
						Integer.SIZE
								- Integer.numberOfLeadingZeros(
								dimensionCount
						);

				break;

			case FIXED_COUNT:
				if (AppContext.dimension_selection_count < 1) {
					throw new IllegalArgumentException(
							"dimension_selection_count must be positive for "
									+ "FIXED_COUNT dimension selection, but received "
									+ AppContext.dimension_selection_count
									+ "."
					);
				}

				selectedCount =
						AppContext.dimension_selection_count;

				break;

			case PROPORTION:
				double proportion =
						AppContext.dimension_selection_proportion;

				if (!Double.isFinite(
						proportion
				)
						|| proportion <= 0.0
						|| proportion > 1.0) {

					throw new IllegalArgumentException(
							"dimension_selection_proportion must be finite and "
									+ "within (0, 1], but received "
									+ proportion
									+ "."
					);
				}

				selectedCount =
						(int) Math.ceil(
								proportion
										* dimensionCount
						);

				break;

			default:
				throw new IllegalStateException(
						"Unsupported dimension-selection strategy: "
								+ strategy
				);
		}

		return Math.max(
				1,
				Math.min(
						dimensionCount,
						selectedCount
				)
		);
	}

	/**
	 * Samples distinct sorted dimension indices without replacement.
	 *
	 * <p>A primitive partial Fisher-Yates shuffle is used when dimensionality is
	 * moderate or the requested selection is dense. Floyd sampling is used when
	 * dimensionality is large and the requested selection is sparse, avoiding an
	 * O(d) temporary array.</p>
	 */
	private static int[] sampleDistinctDimensions(
			int dimensionCount,
			int selectedCount,
			Random random
	) {
		if (dimensionCount < 1) {
			throw new IllegalArgumentException(
					"dimensionCount must be positive."
			);
		}

		if (selectedCount < 1
				|| selectedCount > dimensionCount) {

			throw new IllegalArgumentException(
					"selectedCount must be within [1, dimensionCount]. "
							+ "Received selectedCount="
							+ selectedCount
							+ " and dimensionCount="
							+ dimensionCount
							+ "."
			);
		}

		if (selectedCount == dimensionCount) {
			return null;
		}

		if (dimensionCount
				<= DENSE_DIMENSION_SAMPLING_THRESHOLD
				|| (long) selectedCount * 4L
				>= dimensionCount) {

			return sampleDimensionsByPartialShuffle(
					dimensionCount,
					selectedCount,
					random
			);
		}

		return sampleDimensionsByFloyd(
				dimensionCount,
				selectedCount,
				random
		);
	}

	/**
	 * Samples dimensions using a primitive partial Fisher-Yates shuffle.
	 */
	private static int[] sampleDimensionsByPartialShuffle(
			int dimensionCount,
			int selectedCount,
			Random random
	) {
		int[] available =
				new int[dimensionCount];

		for (int dimension = 0;
			 dimension < dimensionCount;
			 dimension++) {

			available[dimension] =
					dimension;
		}

		for (int output = 0;
			 output < selectedCount;
			 output++) {

			int swapIndex =
					output
							+ random.nextInt(
							dimensionCount - output
					);

			int temporary =
					available[output];

			available[output] =
					available[swapIndex];

			available[swapIndex] =
					temporary;
		}

		int[] selected =
				Arrays.copyOf(
						available,
						selectedCount
				);

		Arrays.sort(
				selected
		);

		return selected;
	}

	/**
	 * Samples dimensions with Floyd's algorithm.
	 *
	 * <p>This path uses O(k) expected temporary storage rather than allocating an
	 * array of length d. It is reserved for the sparse high-dimensional case,
	 * where avoiding the O(d) temporary array outweighs boxed-set overhead.</p>
	 */
	private static int[] sampleDimensionsByFloyd(
			int dimensionCount,
			int selectedCount,
			Random random
	) {
		int initialCapacity =
				Math.max(
						16,
						(int) Math.ceil(
								selectedCount / 0.75
						)
				);

		Set<Integer> selectedSet =
				new HashSet<>(
						initialCapacity
				);

		for (int upper = dimensionCount - selectedCount;
			 upper < dimensionCount;
			 upper++) {

			int candidate =
					random.nextInt(
							upper + 1
					);

			if (!selectedSet.add(
					candidate
			)) {
				selectedSet.add(
						upper
				);
			}
		}

		if (selectedSet.size() != selectedCount) {
			throw new IllegalStateException(
					"Dimension sampler produced "
							+ selectedSet.size()
							+ " selections; expected "
							+ selectedCount
							+ "."
			);
		}

		int[] selected =
				new int[selectedCount];

		int output =
				0;

		for (int dimension : selectedSet) {
			selected[output++] =
					dimension;
		}

		Arrays.sort(
				selected
		);

		return selected;
	}

	private record CandidateInitialization(
			Object[] exemplars, ListObjectDataset[] splits
	) {
	}

	private record CandidateSplitResult(
			int candidateIndex,
			DistanceMeasure distanceMeasure,
			Object[] exemplars,
			ListObjectDataset[] splits,
			double weightedPurity
	) {
	}

	/**
	 * Returns the node's selected realized dimensions.
	 *
	 * @return a defensive copy of the sorted selected indices, or null when the
	 *         splitter uses every available dimension
	 */
	public int[] getSelectedDimensions() {
		return selectedDimensions == null
				? null
				: selectedDimensions.clone();
	}
}