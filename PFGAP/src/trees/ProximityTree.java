package trees;

import core.AppContext;
import core.TreeStatCollector;
import core.contracts.ObjectDataset;
import datasets.ListObjectDataset;
import distance.DistanceMeasure;
import distance.MEASURE;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * One tree in a proximity forest.
 *
 * <p>Each tree owns an independent deterministic random-number generator.
 * The original training dataset is treated as read-only. Bootstrap and
 * out-of-bag subsets preserve the original stored series representations,
 * including LazySeriesRef values.</p>
 */
public class ProximityTree
		implements Serializable {

	@Serial
	private static final long serialVersionUID =
			1L;

	protected int forest_id;

	private int tree_id;

	protected Node root;

	protected int node_counter =
			0;

	/**
	 * Reconstructed from treeSeed when training begins.
	 */
	protected transient Random rand;

	/**
	 * Stable seed assigned when the tree is constructed.
	 *
	 * <p>This avoids sharing one mutable Random among concurrently trained
	 * trees.</p>
	 */
	private final long treeSeed;

	public TreeStatCollector stats;

	protected ArrayList<Node> leaves;

	private final MEASURE[] chosen_distances;

	private String[] distance_file;

	/**
	 * Used when random_dm_per_node is false.
	 */
	protected DistanceMeasure tree_distance_measure;

	public ProximityTree(
			int treeId,
			ProximityForest forest,
			MEASURE... chosenDistances
	) {
		if (forest == null) {
			throw new IllegalArgumentException(
					"ProximityTree requires a non-null forest."
			);
		}

		this.forest_id =
				forest.forest_id;

		this.tree_id =
				treeId;

		/*
		 * Trees are constructed sequentially by ProximityForest, so each tree
		 * receives one deterministic seed before parallel training begins.
		 */
		this.treeSeed =
				AppContext.getRand()
						.nextLong();

		this.rand =
				new Random(
						treeSeed
				);

		this.stats =
				new TreeStatCollector(
						forest_id,
						tree_id
				);

		this.leaves =
				new ArrayList<>();

		this.chosen_distances =
				chosenDistances == null
						? new MEASURE[0]
						: chosenDistances.clone();

		initializeDistanceFile();
	}

	private void initializeDistanceFile() {
		distance_file =
				new String[]{""};

		for (MEASURE measure : chosen_distances) {
			if (measure == MEASURE.maple) {
				distance_file =
						new String[]{
								"MapleDistance.mpl"
						};
			} else if (measure == MEASURE.python) {
				distance_file =
						new String[]{
								"PythonDistance.py"
						};
			}
		}
	}

	public Node getRootNode() {
		return root;
	}

	public MEASURE[] getChosen_distances() {
		return chosen_distances.clone();
	}

	public String[] getDistance_file() {
		return distance_file == null
				? new String[]{""}
				: distance_file.clone();
	}

	public ArrayList<Node> getLeaves() {
		return leaves;
	}

	/**
	 * Trains this tree.
	 *
	 * <p>The supplied dataset is treated as read-only. In particular, this
	 * method does not replace the source dataset's internal index list.</p>
	 */
	public void train(
			ListObjectDataset data
	) throws Exception {

		if (data == null || data.size() == 0) {
			throw new IllegalArgumentException(
					"Cannot train ProximityTree on empty data."
			);
		}

		resetTrainingState();

		initializeTreeDistanceMeasure();

		root =
				new Node(
						null,
						null,
						++node_counter,
						this
				);

		BootstrapSample bootstrapSample =
				createBootstrapSample(
						data.size()
				);

		populateRootBootstrapMetadata(
				bootstrapSample
		);

		ListObjectDataset inBagData =
				createDatasetFromPositions(
						data,
						bootstrapSample.sampledPositions
				);

		ListObjectDataset outOfBagData =
				createDatasetFromPositions(
						data,
						bootstrapSample.outOfBagPositions
				);

		root.train(
				inBagData,
				outOfBagData
		);
	}

	private void resetTrainingState() {
		rand =
				new Random(
						treeSeed
				);

		node_counter =
				0;

		root =
				null;

		tree_distance_measure =
				null;

		leaves =
				new ArrayList<>();
	}

	/**
	 * Selects the distance used by this tree when distance selection occurs
	 * once per tree.
	 *
	 * <p>Distance parameters are still selected per node by Splitter.</p>
	 */
	private void initializeTreeDistanceMeasure()
			throws Exception {

		if (AppContext.random_dm_per_node) {
			return;
		}

		if (chosen_distances.length == 0) {
			if (AppContext.enabled_distance_measures == null
					|| AppContext.enabled_distance_measures.length == 0) {

				throw new IllegalStateException(
						"No enabled distance measures are available."
				);
			}

			int selectedIndex =
					rand.nextInt(
							AppContext
									.enabled_distance_measures
									.length
					);

			tree_distance_measure =
					new DistanceMeasure(
							AppContext.enabled_distance_measures[
									selectedIndex
									],
							descriptorAt(
									selectedIndex
							)
					);

			return;
		}

		int selectedIndex =
				rand.nextInt(
						chosen_distances.length
				);

		tree_distance_measure =
				new DistanceMeasure(
						chosen_distances[selectedIndex],
						descriptorAt(
								selectedIndex
						)
				);
	}

	private String[] descriptorAt(
			int index
	) {
		if (AppContext.Descriptors == null
				|| index < 0
				|| index >= AppContext.Descriptors.size()) {

			return new String[0];
		}

		String[] descriptors =
				AppContext.Descriptors.get(
						index
				);

		return descriptors == null
				? new String[0]
				: descriptors.clone();
	}

	/**
	 * Creates the bootstrap draw, multiplicities, and OOB positions in linear
	 * time.
	 */
	private BootstrapSample createBootstrapSample(
			int size
	) {
		int[] sampledPositions =
				new int[size];

		int[] multiplicities =
				new int[size];

		if (AppContext.bootstrap_trees) {
			for (int draw = 0;
				 draw < size;
				 draw++) {

				int position =
						rand.nextInt(
								size
						);

				sampledPositions[draw] =
						position;

				multiplicities[position]++;
			}
		} else {
			for (int position = 0;
				 position < size;
				 position++) {

				sampledPositions[position] =
						position;

				multiplicities[position] =
						1;
			}
		}

		int outOfBagCount =
				0;

		for (int position = 0;
			 position < size;
			 position++) {

			if (multiplicities[position] == 0) {
				outOfBagCount++;
			}
		}

		int[] outOfBagPositions =
				new int[outOfBagCount];

		int outputIndex =
				0;

		for (int position = 0;
			 position < size;
			 position++) {

			if (multiplicities[position] == 0) {
				outOfBagPositions[outputIndex++] =
						position;
			}
		}

		return new BootstrapSample(
				sampledPositions,
				multiplicities,
				outOfBagPositions
		);
	}

	/**
	 * Preserves the current root-node metadata representation for
	 * compatibility with existing proximity and OOB code.
	 */
	private void populateRootBootstrapMetadata(
			BootstrapSample bootstrapSample
	) {
		root.InBagIndices.ensureCapacity(
				bootstrapSample.sampledPositions.length
		);

		for (int position
				: bootstrapSample.sampledPositions) {

			root.InBagIndices.add(
					position
			);
		}

		root.OutOfBagIndices.ensureCapacity(
				bootstrapSample.outOfBagPositions.length
		);

		for (int position
				: bootstrapSample.outOfBagPositions) {

			root.OutOfBagIndices.add(
					position
			);
		}

		root.multiplicities =
				new HashMap<>();

		for (int position = 0;
			 position < bootstrapSample.multiplicities.length;
			 position++) {

			int count =
					bootstrapSample.multiplicities[position];

			if (count > 0) {
				root.multiplicities.put(
						position,
						count
				);
			}
		}
	}

	/**
	 * Builds a dataset subset from source-list positions.
	 *
	 * <p>The source series object is copied by reference. LazySeriesRef
	 * instances therefore remain lazy. The subset's internal index is the
	 * original dataset index, not merely the source-list position.</p>
	 */
	private ListObjectDataset createDatasetFromPositions(
			ObjectDataset source,
			int[] positions
	) {
		ListObjectDataset subset =
				new ListObjectDataset(
						positions.length
				);

		subset.setLength(
				source.getLength()
		);

		for (int position : positions) {
			subset.add(
					source.get_class(
							position
					),
					source.get_series(
							position
					),
					source.get_index(
							position
					)
			);
		}

		return subset;
	}

	/**
	 * Legacy helper retained for compatibility with callers that may still
	 * use it indirectly.
	 */
	private int[] sampleTrainingIndices(
			int size
	) {
		return createBootstrapSample(
				size
		).sampledPositions;
	}

	/**
	 * Predicts one query using its stored or eager representation.
	 *
	 * <p>This compatibility path may resolve a lazy query at each traversed
	 * node. Evaluation will later be refactored to resolve the query once and
	 * call predictResolved().</p>
	 */
	public Object predict(
			Object query,
			int index
	) throws Exception {

		Node leaf =
				findLeaf(
						query
				);

		leaf.TestIndices.add(
				index
		);

		return leaf.label();
	}

	/**
	 * Finds the leaf reached by a stored or eager query.
	 */
	public Node findLeaf(
			Object query
	) throws Exception {

		if (root == null) {
			throw new IllegalStateException(
					"Cannot predict with an untrained tree."
			);
		}

		Node current =
				root;

		while (!current.is_leaf()) {
			int branch =
					current.splitter
							.find_closest_branch(
									query
							);

			current =
					current.children[branch];
		}

		return current;
	}

	/**
	 * Traverses the tree using an already materialized query.
	 *
	 * <p>Node exemplars are resolved temporarily by Splitter. The query is
	 * not reread at every node.</p>
	 */
	public Object predictResolved(
			Object resolvedQuery,
			int index,
			Random predictionRandom
	) throws Exception {

		Node leaf =
				findLeafResolved(
						resolvedQuery,
						predictionRandom
				);

		leaf.TestIndices.add(
				index
		);

		return leaf.label();
	}

	/**
	 * Pure traversal helper for an already materialized query.
	 *
	 * <p>This method does not itself update TestIndices. The overload above
	 * retains existing behavior, while a later forest refactor can use this
	 * method and record test membership outside the traversal hot path.</p>
	 */
	public Node findLeafResolved(
			Object resolvedQuery,
			Random predictionRandom
	) throws Exception {

		if (root == null) {
			throw new IllegalStateException(
					"Cannot predict with an untrained tree."
			);
		}

		if (resolvedQuery == null) {
			throw new IllegalArgumentException(
					"Resolved prediction query cannot be null."
			);
		}

		if (predictionRandom == null) {
			throw new IllegalArgumentException(
					"Prediction Random cannot be null."
			);
		}

		Node current =
				root;

		while (!current.is_leaf()) {
			int branch =
					current.splitter
							.findClosestBranchResolved(
									resolvedQuery,
									predictionRandom
							);

			current =
					current.children[branch];
		}

		return current;
	}

	public int getTreeID() {
		return tree_id;
	}

	long deriveSeed(
			int nodeId,
			int purpose
	) {
		long seed =
				treeSeed;

		seed =
				mixSeed(
						seed,
						nodeId
				);

		return mixSeed(
				seed,
				purpose
		);
	}

	public TreeStatCollector getTreeStatCollection() {
		stats.collateResults(
				this
		);

		return stats;
	}

	public int get_num_nodes() {
		int actualCount =
				get_num_nodes(
						root
				);

		if (node_counter != actualCount) {
			System.out.println(
					"Error: node counter is "
							+ node_counter
							+ " but recursive count is "
							+ actualCount
							+ "."
			);

			return -1;
		}

		return node_counter;
	}

	public int get_num_nodes(
			Node node
	) {
		if (node == null) {
			return 0;
		}

		if (node.children == null) {
			return 1;
		}

		int count =
				1;

		for (Node child : node.children) {
			count +=
					get_num_nodes(
							child
					);
		}

		return count;
	}

	public int get_num_leaves() {
		return get_num_leaves(
				root
		);
	}

	public int get_num_leaves(
			Node node
	) {
		if (node == null) {
			return 0;
		}

		if (node.children == null) {
			return 1;
		}

		int count =
				0;

		for (Node child : node.children) {
			count +=
					get_num_leaves(
							child
					);
		}

		return count;
	}

	public int get_num_internal_nodes() {
		return get_num_internal_nodes(
				root
		);
	}

	public int get_num_internal_nodes(
			Node node
	) {
		if (node == null
				|| node.children == null) {

			return 0;
		}

		int count =
				1;

		for (Node child : node.children) {
			count +=
					get_num_internal_nodes(
							child
					);
		}

		return count;
	}

	public int get_height() {
		return get_height(
				root
		);
	}

	public int get_height(
			Node node
	) {
		if (node == null
				|| node.children == null) {

			return 0;
		}

		int maximumDepth =
				0;

		for (Node child : node.children) {
			maximumDepth =
					Math.max(
							maximumDepth,
							get_height(
									child
							)
					);
		}

		return maximumDepth + 1;
	}

	public int get_min_depth(
			Node node
	) {
		if (node == null
				|| node.children == null) {

			return 0;
		}

		int minimumDepth =
				Integer.MAX_VALUE;

		for (Node child : node.children) {
			minimumDepth =
					Math.min(
							minimumDepth,
							get_min_depth(
									child
							)
					);
		}

		return minimumDepth + 1;
	}

	/**
	 * Immutable bootstrap construction result.
	 */
	private static final class BootstrapSample {

		private final int[] sampledPositions;
		private final int[] multiplicities;
		private final int[] outOfBagPositions;

		private BootstrapSample(
				int[] sampledPositions,
				int[] multiplicities,
				int[] outOfBagPositions
		) {
			this.sampledPositions =
					sampledPositions;

			this.multiplicities =
					multiplicities;

			this.outOfBagPositions =
					outOfBagPositions;
		}
	}

	/**
	 * Tree node.
	 */
	public class Node
			implements Serializable {

		@Serial
		private static final long serialVersionUID =
				1L;

		protected ArrayList<Integer> InBagIndices;

		protected ArrayList<Integer> OutOfBagIndices;

		public ArrayList<Integer> TestIndices;

		protected Map<Integer, Integer> multiplicities;

		protected Node parent;

		protected ProximityTree tree;

		protected int node_id;

		protected int node_depth =
				0;

		protected boolean is_leaf =
				false;

		protected Object label;

		protected Node[] children;

		protected Splitter splitter;

		public Node(
				Node parent,
				Integer branchLabel,
				int nodeId,
				ProximityTree tree
		) {
			this.parent =
					parent;

			this.node_id =
					nodeId;

			this.tree =
					tree;

			this.InBagIndices =
					new ArrayList<>();

			this.OutOfBagIndices =
					new ArrayList<>();

			this.TestIndices =
					new ArrayList<>();

			this.multiplicities =
					null;

			if (parent != null) {
				node_depth =
						parent.node_depth + 1;
			}
		}

		public boolean is_leaf() {
			return is_leaf;
		}

		public Object label() {
			return label;
		}

		public Node[] get_children() {
			return children;
		}

		public void setInBagIndices(
				ArrayList<Integer> indices
		) {
			this.InBagIndices =
					indices == null
							? new ArrayList<>()
							: indices;
		}

		public ArrayList<Integer> getInBagIndices() {
			return InBagIndices;
		}

		public void setOutOfBagIndices(
				ArrayList<Integer> indices
		) {
			this.OutOfBagIndices =
					indices == null
							? new ArrayList<>()
							: indices;
		}

		public ArrayList<Integer> getOutOfBagIndices() {
			return OutOfBagIndices;
		}

		public void setMultiplicities(
				Map<Integer, Integer> multiplicities
		) {
			this.multiplicities =
					multiplicities;
		}

		public Map<Integer, Integer> getMultiplicities() {
			return multiplicities;
		}

		@Override
		public String toString() {
			return "Node{"
					+ "nodeId="
					+ node_id
					+ ", depth="
					+ node_depth
					+ ", leaf="
					+ is_leaf
					+ '}';
		}

		/**
		 * Computes the response attached to a leaf.
		 */
		public static Object computeLeafLabel(
				List<Object> labels
		) {
			if (AppContext.isIsolationMode()) {
				return null;
			}

			if (labels == null || labels.isEmpty()) {
				return null;
			}

			if (AppContext.isRegressionMode()) {
				return computeRegressionLeafValue(
						labels
				);
			}

			return computeClassificationLeafValue(
					labels
			);
		}

		private static Object computeRegressionLeafValue(
				List<Object> labels
		) {
			double[] numericValues =
					new double[labels.size()];

			int numericCount =
					0;

			double sum =
					0.0;

			for (Object value : labels) {
				if (value instanceof Number number) {
					double numericValue =
							number.doubleValue();

					numericValues[numericCount++] =
							numericValue;

					sum +=
							numericValue;
				}
			}

			if (numericCount == 0) {
				return 0.0;
			}

			if (AppContext.voting.equalsIgnoreCase(
					"mean"
			)) {
				return sum / numericCount;
			}

			if (AppContext.voting.equalsIgnoreCase(
					"median"
			)) {
				Arrays.sort(
						numericValues,
						0,
						numericCount
				);

				if ((numericCount & 1) == 1) {
					return numericValues[
							numericCount / 2
							];
				}

				return (
						numericValues[
								numericCount / 2 - 1
								]
								+ numericValues[
								numericCount / 2
								]
				) / 2.0;
			}

			throw new IllegalArgumentException(
					"Unknown voting method: "
							+ AppContext.voting
			);
		}

		private static Object computeClassificationLeafValue(
				List<Object> labels
		) {
			Map<Object, Integer> frequencies =
					new LinkedHashMap<>();

			Object mostFrequentLabel =
					null;

			int maximumFrequency =
					0;

			for (Object currentLabel : labels) {
				int frequency =
						frequencies.getOrDefault(
								currentLabel,
								0
						) + 1;

				frequencies.put(
						currentLabel,
						frequency
				);

				if (frequency > maximumFrequency) {
					maximumFrequency =
							frequency;

					mostFrequentLabel =
							currentLabel;
				}
			}

			return mostFrequentLabel;
		}

		/**
		 * Recursively trains this node.
		 */
		public void train(
				ObjectDataset data,
				ObjectDataset oobData
		) throws Exception {

			if (data == null || data.size() == 0) {
				throw new IllegalStateException(
						"Cannot train an empty tree node."
				);
			}

			if (shouldStopForIsolationSize(
					data
			)) {
				makeLeaf(
						null
				);

				return;
			}

			if (shouldStopForPureClassification(
					data
			)) {
				makeLeaf(
						onlyClassLabel(
								data
						)
				);

				return;
			}

			if (shouldStopForPurity(
					data
			)) {
				makeLeaf(
						computeLeafLabel(
								data._internal_class_list()
						)
				);

				return;
			}

			if (shouldStopForDepth()) {
				makeLeaf(
						AppContext.isIsolationMode()
								? null
								: computeLeafLabel(
								data._internal_class_list()
						)
				);

				return;
			}

			splitter =
					new Splitter(
							this
					);

			ObjectDataset[] bestSplits =
					splitter.find_best_split(
							data
					);

			if (hasEmptySplit(
					bestSplits
			)) {
				makeLeaf(
						AppContext.isIsolationMode()
								? null
								: computeLeafLabel(
								data._internal_class_list()
						)
				);

				return;
			}

			initializeChildren(
					bestSplits
			);

			ObjectDataset[] oobSplits =
					createEmptyOobSplits(
							bestSplits.length,
							oobData
					);

			routeOutOfBagData(
					oobData,
					oobSplits
			);

			for (int branch = 0;
				 branch < bestSplits.length;
				 branch++) {

				children[branch].train(
						bestSplits[branch],
						oobSplits[branch]
				);
			}
		}

		private boolean shouldStopForIsolationSize(
				ObjectDataset data
		) {
			return AppContext.isIsolationMode()
					&& data.size()
					<= AppContext.isolation_min_leaf_size;
		}

		/**
		 * Avoids a full Gini or entropy scan for an already pure
		 * classification node.
		 */
		private boolean shouldStopForPureClassification(
				ObjectDataset data
		) {
			return !AppContext.isIsolationMode()
					&& !AppContext.isRegressionMode()
					&& data.get_num_classes() <= 1;
		}

		private Object onlyClassLabel(
				ObjectDataset data
		) {
			Set<Object> classes =
					data.get_unique_classes_as_set();

			if (classes == null || classes.isEmpty()) {
				return computeLeafLabel(
						data._internal_class_list()
				);
			}

			return classes.iterator()
					.next();
		}

		private boolean shouldStopForPurity(
				ObjectDataset data
		) {
			return !AppContext.isIsolationMode()
					&& data.purity(
					AppContext.purity_measure
			)
					<= AppContext.purity_threshold;
		}

		private boolean shouldStopForDepth() {
			return AppContext.max_depth != 0
					&& node_depth
					>= AppContext.max_depth;
		}

		private void makeLeaf(
				Object leafLabel
		) {
			label =
					leafLabel;

			is_leaf =
					true;

			children =
					null;

			tree.leaves.add(
					this
			);
		}

		private boolean hasEmptySplit(
				ObjectDataset[] splits
		) {
			if (splits == null
					|| splits.length == 0) {

				return true;
			}

			for (ObjectDataset split : splits) {
				if (split == null
						|| split.size() == 0) {

					return true;
				}
			}

			return false;
		}

		private void initializeChildren(
				ObjectDataset[] bestSplits
		) {
			children =
					new Node[bestSplits.length];

			for (int branch = 0;
				 branch < children.length;
				 branch++) {

				children[branch] =
						new Node(
								this,
								branch,
								++tree.node_counter,
								tree
						);

				children[branch].setInBagIndices(
						new ArrayList<>(
								bestSplits[branch]
										._internal_indices_list()
						)
				);
			}
		}

		private ObjectDataset[] createEmptyOobSplits(
				int branchCount,
				ObjectDataset oobData
		) {
			ObjectDataset[] oobSplits =
					new ObjectDataset[branchCount];

			int expectedBranchSize =
					oobData == null
							? 0
							: Math.max(
							1,
							oobData.size()
									/ branchCount
					);

			for (int branch = 0;
				 branch < branchCount;
				 branch++) {

				ListObjectDataset split =
						new ListObjectDataset(
								expectedBranchSize
						);

				if (oobData != null) {
					split.setLength(
							oobData.getLength()
					);
				}

				oobSplits[branch] =
						split;
			}

			return oobSplits;
		}

		/**
		 * Routes OOB instances through the selected split.
		 *
		 * <p>The selected node exemplars are resolved once for the entire OOB batch,
		 * rather than once for every OOB instance.</p>
		 */
		private void routeOutOfBagData(
				ObjectDataset oobData,
				ObjectDataset[] oobSplits
		) throws Exception {

			if (oobData == null
					|| oobData.size() == 0) {

				return;
			}

			int[] branches =
					splitter.findClosestBranches(
							oobData,
							tree.deriveSeed(
									node_id,
									0x4F4F42
							)
					);

			if (branches.length != oobData.size()) {
				throw new IllegalStateException(
						"OOB branch-assignment count does not match "
								+ "the OOB dataset size."
				);
			}

			for (int index = 0;
				 index < oobData.size();
				 index++) {

				int branch =
						branches[index];

				if (branch < 0
						|| branch >= children.length) {

					throw new IllegalStateException(
							"Invalid OOB branch "
									+ branch
									+ " for node "
									+ node_id
									+ "."
					);
				}

				Integer originalIndex =
						oobData.get_index(
								index
						);

				Object currentLabel =
						oobData.get_class(
								index
						);

				Object storedSeries =
						oobData.get_series(
								index
						);

				children[branch]
						.OutOfBagIndices
						.add(
								originalIndex
						);

				oobSplits[branch].add(
						currentLabel,
						storedSeries,
						originalIndex
				);
			}
		}

		public Splitter getSplitter() {
			return splitter;
		}
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
}