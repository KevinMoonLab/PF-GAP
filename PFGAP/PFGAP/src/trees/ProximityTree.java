package trees;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.*;

import core.AppContext;
import core.TreeStatCollector;
//import core.contracts.Dataset;
import core.contracts.ObjectDataset;
//import datasets.ListDataset;
import datasets.ListObjectDataset;
import distance.DistanceMeasure;
import distance.MEASURE;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

//public class Proximity Tree {
public class ProximityTree implements Serializable {
	protected int forest_id;	
	private int tree_id;
	protected Node root;
	protected int node_counter = 0;
	
	protected transient Random rand;
	public TreeStatCollector stats;
	protected ArrayList<Node> leaves;
	private MEASURE[] chosen_distances;
	private String[] distance_file;
	
	protected DistanceMeasure tree_distance_measure; //only used if AppContext.random_dm_per_node == false

	public ProximityTree(int tree_id, ProximityForest forest, MEASURE... chosen_distances) {
		this.forest_id = forest.forest_id;
		this.tree_id = tree_id;
		this.rand = AppContext.getRand();
		stats = new TreeStatCollector(forest_id, tree_id);
		this.leaves = new ArrayList<Node>();
		//this.distance_file = distance_file;
		//if (distance_file[0].contains(".mpl")){
			//System.out.println("Ah, so you're a Maple user, eh?");
		//	this.chosen_distances = new MEASURE[]{MEASURE.maple};
		//}
		//if (distance_file[0].contains(".py")){
			//System.out.println("Ah, so you're a Python user, eh?");
		//	this.chosen_distances = new MEASURE[]{MEASURE.python};
		//}
		this.chosen_distances = chosen_distances;
		if (chosen_distances.length > 0){
			if (Arrays.toString(chosen_distances).contains("maple")){
				this.distance_file = new String[]{"MapleDistance.mpl"};
			}
			if (Arrays.toString(chosen_distances).contains("python")){
				this.distance_file = new String[]{"PythonDistance.py"};
			}
		} else {
			this.distance_file = new String[]{""};
		}

	}

	public Node getRootNode() {
		return this.root;
	}

	public MEASURE[] getChosen_distances(){
		return this.chosen_distances;
	}

	public String[] getDistance_file(){
		return this.distance_file;
	}

	public ArrayList<Node> getLeaves() {return this.leaves;}
	
	//public void train(Dataset data) throws Exception {
	public void train(ListObjectDataset data) throws Exception {
		//System.out.println("Training a tree");
		if (this.chosen_distances.length == 0){
			if (AppContext.random_dm_per_node ==  false) {	//DM is selected once per tree
				int r = AppContext.getRand().nextInt(AppContext.enabled_distance_measures.length);
				tree_distance_measure = new DistanceMeasure(AppContext.enabled_distance_measures[r]);
				//params selected per node in the splitter class
			}
		}
		else {
			if (AppContext.random_dm_per_node ==  false) {	//DM is selected once per tree
				int r = AppContext.getRand().nextInt(this.chosen_distances.length);
				tree_distance_measure = new DistanceMeasure(this.chosen_distances[r]);
				//params selected per node in the splitter class
			}
		}
		
		this.root = new Node(null, null, ++node_counter, this);
		//Try putting the in- and out-of-bag stuff here.
		ObjectDataset inbagData = new ListObjectDataset(); //ListDataset();
		ObjectDataset oobData = new ListObjectDataset(); //ListDataset();

		//First we get the in-bag indices.
		int dummySize = data.size();
		int[]  randomIntsArray = IntStream.generate(() -> new Random().nextInt(dummySize)).limit(data.size()).toArray();
		//InBagIndices = new ArrayList<Integer>();
		for(int i=0; i<randomIntsArray.length; i++){
			this.getRootNode().InBagIndices.add(randomIntsArray[i]);
		}
		//int[] num = IntStream.range(0, data.size()).toArray();
		int[] distinctInBag = Arrays.stream(randomIntsArray).distinct().toArray();
		//setInBagIndices(distinctInBag);
		//setInBagIndices(randomIntsArray);

		this.getRootNode().multiplicities = new HashMap<Integer,Integer>();
		//now we compute the multiplicities (number of times the indices occur in the in-bag sample)
		for (int i=0; i<this.getRootNode().InBagIndices.size(); i++){
			if (this.getRootNode().multiplicities.containsKey((this.getRootNode().InBagIndices.get(i)))){
				this.getRootNode().multiplicities.put(this.getRootNode().InBagIndices.get(i),this.getRootNode().multiplicities.get(this.getRootNode().InBagIndices.get(i))+1);
			}
			else{
				this.getRootNode().multiplicities.put(this.getRootNode().InBagIndices.get(i),1);
			}
		}

		//now we need to get the out-of-bag indices.
		ArrayList<Integer> result = new ArrayList<Integer>(); //out-of-bag
		//ArrayList<Integer> resultA = (ArrayList<Integer>) InBagIndices; //new ArrayList<Integer>(); //for converting inbag
		for (int i = 0; i < randomIntsArray.length; i++){
			Boolean isInBag = ArrayUtils.contains(distinctInBag, i);
			if (!isInBag){result.add(i);}
			//else{resultA.add(i);}
		}
		int[] result2 = result.stream().mapToInt(i -> i).toArray();
		//OutOfBagIndices = new ArrayList<Integer>();
		for(int i=0; i<result2.length; i++){
			this.getRootNode().OutOfBagIndices.add(result2[i]);
		}
		//setOutOfBagIndices(result2);
		data.set_indices(this.getRootNode().InBagIndices);

		// Now we need to get the sub sample corresponding to the in-bag indices.
		ObjectDataset data2 = new ListObjectDataset(); //ListDataset(); //data;
		for (int index : this.getRootNode().InBagIndices){
			//System.out.println(Arrays.toString(data.get_series(index)));
			//System.out.println(data.get_index(index));
			data2.add(data.get_class(index), data.get_series(index), index);

		}
		//data = data2; //we're only training on in-bag samples.
		inbagData = data2;
		// Now we need to get the sub sample corresponding to the out-of-bag indices.
		ObjectDataset data3 = new ListObjectDataset(); //new ListDataset();
		for (int index : this.getRootNode().OutOfBagIndices){
			//System.out.println(Arrays.toString(data.get_series(index)));
			//System.out.println(data.get_index(index));
			data3.add(data.get_class(index), data.get_series(index), index);

		}
		oobData = data3;
		//System.out.println(Arrays.toString(data.get_series(0)));
		//System.out.println(data.get_index(0));


		//Integer[] result = s1.toArray(new Integer[s1.size()]);



		this.root.train(inbagData, oobData);
	}
	
	//public Integer predict(double[] query) throws Exception {
	public Object predict(Object query, int index) throws Exception {
		Node node = this.root;

		while(!node.is_leaf()) {
			node = node.children[node.splitter.find_closest_branch(query)];
		}
		node.TestIndices.add(index);
		return node.label();
	}	

	
	public int getTreeID() {
		return tree_id;
	}

	
	//************************************** START stats -- development/debug code
	public TreeStatCollector getTreeStatCollection() {
		
		stats.collateResults(this);
		
		return stats;
	}	
	
	public int get_num_nodes() {
		if (node_counter != get_num_nodes(root)) {
			System.out.println("Error: error in node counter!");
			return -1;
		}else {
			return node_counter;
		}
	}	

	public int get_num_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_nodes(n.children[i]);
		}
		
		return count+1;
	}
	
	public int get_num_leaves() {
		return get_num_leaves(root);
	}	
	
	public int get_num_leaves(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_leaves(n.children[i]);
		}
		
		return count;
	}
	
	public int get_num_internal_nodes() {
		return get_num_internal_nodes(root);
	}
	
	public int get_num_internal_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 0;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_internal_nodes(n.children[i]);
		}
		
		return count+1;
	}
	
	public int get_height() {
		return get_height(root);
	}
	
	public int get_height(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}

		for (int i = 0; i < n.children.length; i++) {
			max_depth = Math.max(max_depth, get_height(n.children[i]));
		}
		
		return max_depth+1;
	}
	
	public int get_min_depth(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}

		for (int i = 0; i < n.children.length; i++) {
			max_depth = Math.min(max_depth, get_height(n.children[i]));
		}
		
		return max_depth+1;
	}
	
//	public double get_weighted_depth() {
//		return printTreeComplexity(root, 0, root.data.size());
//	}
//	
//	// high deep and unbalanced
//	// low is shallow and balanced?
//	public double printTreeComplexity(Node n, int depth, int root_size) {
//		double ratio = 0;
//		
//		if (n.is_leaf) {
//			double r = (double)n.data.size()/root_size * (double)depth;
////			System.out.format("%d: %d/%d*%d/%d + %f + ", n.label, 
////					n.data.size(),root_size, depth, max_depth, r);
//			
//			return r;
//		}
//		
//		for (int i = 0; i < n.children.length; i++) {
//			ratio += printTreeComplexity(n.children[i], depth+1, root_size);
//		}
//		
//		return ratio;
//	}		
	
	
	//**************************** END stats -- development/debug code
	
	
	
	
	
	
	
	//public class Node{
	public class Node implements Serializable {
	
		protected ArrayList<Integer> InBagIndices; //int[] InBagIndices;
		protected ArrayList<Integer> OutOfBagIndices; //ArrayList<Integer> OutOfBagIndices;
		public ArrayList<Integer> TestIndices;
		protected Map<Integer, Integer> multiplicities;
		//protected transient Node parent;	//dont need this, but it helps to debug
		//protected transient ProximityTree tree;
		protected Node parent;	//dont need this, but it helps to debug
		protected ProximityTree tree;
		
		protected int node_id;
		protected int node_depth = 0;

		protected boolean is_leaf = false;
		//protected Integer label;
		protected Object label; // for classification...

//		protected transient Dataset data;				
		protected Node[] children;
		protected Splitter splitter;
		//System.out.print()
		
		public Node(Node parent, Integer label, int node_id, ProximityTree tree) {
			this.parent = parent;
//			this.data = new ListDataset();
			this.node_id = node_id;
			this.tree = tree;
			this.InBagIndices = new ArrayList<>();
			this.OutOfBagIndices = new ArrayList<>();
			this.TestIndices = new ArrayList<>();
			this.multiplicities = null;
			
			if (parent != null) {
				node_depth = parent.node_depth + 1;
			}
		}
		
		public boolean is_leaf() {
			return this.is_leaf;
		}
		
		public Object label() {
			return this.label;
		}	
		
		public Node[] get_children() {
			return this.children;
		}

		public void setInBagIndices(ArrayList<Integer> indices) { this.InBagIndices=indices;}

		public ArrayList<Integer> getInBagIndices() {return this.InBagIndices;}

		public void setOutOfBagIndices(ArrayList<Integer> indices) { this.OutOfBagIndices=indices;}

		//public ArrayList<Integer> getOutOfBagIndices() {return this.OutOfBagIndices;}
		public ArrayList<Integer> getOutOfBagIndices() {return this.OutOfBagIndices;}

		public void setMultiplicities(Map<Integer, Integer> multiplicities) { this.multiplicities=multiplicities;}

		public Map<Integer, Integer> getMultiplicities() {return this.multiplicities;}
//		public Dataset get_data() {
//			return this.data;
//		}		
		
		public String toString() {
			return "d: ";// + this.data.toString();
		}


		public static Object computeLeafLabel(List<Object> labels) {
			if (labels == null || labels.isEmpty()) return null;

			if (AppContext.isRegression) {
				// Collect numeric values
				List<Double> numericLabels = new ArrayList<>();
				for (Object label : labels) {
					if (label instanceof Number) {
						numericLabels.add(((Number) label).doubleValue());
					}
				}

				if (numericLabels.isEmpty()) return 0.0;

				if (AppContext.voting.equalsIgnoreCase("mean")) {
					return numericLabels.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
				} else if (AppContext.voting.equalsIgnoreCase("median")) {
					Collections.sort(numericLabels);
					int n = numericLabels.size();
					return (n % 2 == 1)
							? numericLabels.get(n / 2)
							: (numericLabels.get(n / 2 - 1) + numericLabels.get(n / 2)) / 2.0;
				} else {
					throw new IllegalArgumentException("Unknown voting method: " + AppContext.voting);
				}

			} else {
				// Classification: vote for majority label
				Map<Object, Integer> frequencyMap = new HashMap<>();
				for (Object label : labels) {
					frequencyMap.put(label, frequencyMap.getOrDefault(label, 0) + 1);
				}

				Object mostFrequentLabel = null;
				int maxFrequency = 0;
				for (Map.Entry<Object, Integer> entry : frequencyMap.entrySet()) {
					if (entry.getValue() > maxFrequency) {
						maxFrequency = entry.getValue();
						mostFrequentLabel = entry.getKey();
					}
				}

				return mostFrequentLabel;
			}
		}

//		public void train(Dataset data) throws Exception {
//			this.data = data;
//			this.train();
//		}		
		
		//public void train(Dataset data, Dataset oobData) throws Exception {
		public void train(ObjectDataset data, ObjectDataset oobData) throws Exception {
//			System.out.println(this.node_depth + ":   " + (this.parent == null ? "r" : this.parent.node_id)  +"->"+ this.node_id +":"+ data.toString());
			//System.out.println("The train method was called on a node");
			//Debugging check
			if (data == null || data.size() == 0) {
				throw new Exception("possible bug: empty node found");
//				this.is_leaf = true;
//				return;
			}

			if (!AppContext.isRegression && data.purity(AppContext.purity_measure) <= AppContext.purity_threshold) {
				this.label = computeLeafLabel(data._internal_class_list());
				this.is_leaf = true;
				this.tree.leaves.add(this);
				return;
			}

			if (AppContext.isRegression && data.purity(AppContext.purity_measure) <= AppContext.purity_threshold) {
				this.label = computeLeafLabel(data._internal_class_list());
				this.is_leaf = true;
				this.tree.leaves.add(this);
				return;
			}

			/*if (AppContext.max_depth != 0 && this.tree.get_height() >= AppContext.max_depth){ //0 means no max depth.
				//first, get a count of each class present in the node. Then find the majority class.
				//int[] classes = data.get_unique_classes();
				Object[] classes = data.get_unique_classes();
				Map<Integer, Integer> frequencyMap = new HashMap<>();

				// Count frequencies of each element
				for (int num : classes) {
					frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
				}

				int mostFrequentElement = classes[0]; // Initialize with the first element
				int maxFrequency = 0;

				// Find the element with the maximum frequency
				for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
					if (entry.getValue() > maxFrequency) {
						maxFrequency = entry.getValue();
						mostFrequentElement = entry.getKey();
					}
				}
				// Now that we have the most Frequent class, we consider the node to be this class and terminate.
				this.label = mostFrequentElement;
				this.is_leaf = true;
				this.tree.leaves.add(this);
				return;
			}*/

			if (AppContext.max_depth != 0 && this.tree.get_height() >= AppContext.max_depth) {

				this.label = computeLeafLabel(data._internal_class_list());

				this.is_leaf = true;
				this.tree.leaves.add(this);
				return;
			}

			this.splitter = new Splitter(this);


			//Dataset[] best_splits = splitter.find_best_split(data);
			ObjectDataset[] best_splits = splitter.find_best_split(data);

			// check to see if any would-be child nodes are empty. We don't want that.
			if (best_splits == null || Arrays.stream(best_splits).anyMatch(split -> split.size() == 0)) {
				this.is_leaf = true;
				this.label = computeLeafLabel(data._internal_class_list());
				this.tree.leaves.add(this);
			} else {

				//Dataset[] oob_splits = new Dataset[best_splits.length];
				ObjectDataset[] oob_splits = new ObjectDataset[best_splits.length];
				this.children = new Node[best_splits.length];
				for (int i = 0; i < children.length; i++) {
					this.children[i] = new Node(this, i, ++tree.node_counter, tree);
					this.children[i].setInBagIndices(best_splits[i]._internal_indices_list());
					oob_splits[i] = new ListObjectDataset();
				}
				//Now we need to let the oob indices trickle down (set the oob indices for the children).
				//System.out.println(oobData.size());
				for (int i = 0; i < oobData.size(); i++) {
					int ind = oobData.get_index(i);
					//int label = oobData.get_class(i);
					Object label = oobData.get_class(i);
					//double[] series = oobData.get_series(i);
					Object series = oobData.get_series(i);
					int branch = splitter.find_closest_branch(oobData.get_series(i));
					//if (this.children[branch] != null){
					//	this.children[branch].OutOfBagIndices.add(ind);
					//}
					this.children[branch].OutOfBagIndices.add(ind);
					oob_splits[branch].add(label, series, ind);
				}

				// Now train on the children.
				for (int i = 0; i < best_splits.length; i++) {

					//this.children[i].train(best_splits[i]);
					this.children[i].train(best_splits[i], oob_splits[i]);
				}
			}
		}

		public Splitter getSplitter() {
			return splitter;
		}



	}
	
}
