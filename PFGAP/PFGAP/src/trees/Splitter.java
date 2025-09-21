package trees;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;

import core.AppContext;
//import core.contracts.Dataset;
import core.contracts.ObjectDataset;
//import datasets.ListDataset;
import datasets.ListObjectDataset;
import distance.DistanceMeasure;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */
// public class Splitter{
public class Splitter implements Serializable {
	
	protected int num_children; //may be move to splitter?
	protected DistanceMeasure distance_measure;
	//protected double[][] exemplars;
	protected Object[] exemplars;
	
	protected DistanceMeasure temp_distance_measure;
	//protected double[][] temp_exemplars;
	protected Object[] temp_exemplars;
	
	//ListDataset[] best_split = null;
	ListObjectDataset[] best_split = null;
	ProximityTree.Node node;
	
	public Splitter(ProximityTree.Node node) throws Exception {
		this.node = node;	
	}
	
	//public ListDataset[] split_data(Dataset sample, Map<Integer, ListDataset> data_per_class) throws Exception {
	public ListObjectDataset[] split_data(ObjectDataset sample, Map<Integer, ListObjectDataset> data_per_class) throws Exception {
//		num_children = sample.get_num_classes();
		ListObjectDataset[] splits = new ListObjectDataset[sample.get_num_classes()];
		temp_exemplars = new Object[sample.get_num_classes()][]; //double[sample.get_num_classes()][];

		int branch = 0;
		for (Map.Entry<Integer, ListObjectDataset> entry : data_per_class.entrySet()) {
			int r = AppContext.getRand().nextInt(entry.getValue().size());
			
			splits[branch] = new ListObjectDataset(sample.size()); //, sample.length());
			//splits[branch] = new ListDataset(sample.size(), sample.length(), sample.length());
			//use key just in case iteration order is not consistent
			temp_exemplars[branch] = entry.getValue().get_series(r);
			branch++;
		}
		
		int sample_size = sample.size();
		int closest_branch = -1;
		for (int j = 0; j < sample_size; j++) {
			closest_branch = this.find_closest_branch(sample.get_series(j), 
					temp_distance_measure, temp_exemplars);
			if (closest_branch == -1) {
				assert false;
			}
			//splits[closest_branch].add(sample.get_class(j), sample.get_series(j));
			splits[closest_branch].add(sample.get_class(j), sample.get_series(j), sample._internal_indices_list().get(j));
		}

		return splits;
	}	

	//public int find_closest_branch(double[] query, DistanceMeasure dm, double[][] e) throws Exception{
	public int find_closest_branch(Object query, DistanceMeasure dm, Object[] e) throws Exception{
		return dm.find_closest_node(query, e, true, this.node.tree.getDistance_file());
	}	
	
	//public int find_closest_branch(double[] query) throws Exception{
	public int find_closest_branch(Object query) throws Exception{
		return this.distance_measure.find_closest_node(query, exemplars, true, this.node.tree.getDistance_file());
	}		
	
	//public Dataset[] getBestSplits() {
	public ObjectDataset[] getBestSplits() {
		return this.best_split;
	}
	
	//public ListDataset[] find_best_split(Dataset data) throws Exception {
	//public ListDataset[] find_best_split(ObjectDataset data) throws Exception {
	public ListObjectDataset[] find_best_split(ObjectDataset data) throws Exception {
				
		Map<Integer, ListObjectDataset> data_per_class = data.split_classes();
		
		double weighted_gini = Double.POSITIVE_INFINITY;
		double best_weighted_gini = Double.POSITIVE_INFINITY;
		ListObjectDataset[] splits = null;
		int parent_size = data.size();
	
		for (int i = 0; i < AppContext.num_candidates_per_split; i++) {

			if (this.node.tree.getChosen_distances().length==0){
				if (AppContext.random_dm_per_node) {
					int r = AppContext.getRand().nextInt(AppContext.enabled_distance_measures.length);
					temp_distance_measure = new DistanceMeasure(AppContext.enabled_distance_measures[r]);
				}else {
					//NOTE: num_candidates_per_split has no effect if random_dm_per_node == false (if DM is selected once per tree)
					//after experiments we found that DM selection per node is better since it diversifies the ensemble
					temp_distance_measure = node.tree.tree_distance_measure;
				}

				temp_distance_measure.select_random_params(data, AppContext.getRand());
			} else{
				if (AppContext.random_dm_per_node) {
					int r = AppContext.getRand().nextInt(this.node.tree.getChosen_distances().length);
					temp_distance_measure = new DistanceMeasure(this.node.tree.getChosen_distances()[r]);
				}else {
					//NOTE: num_candidates_per_split has no effect if random_dm_per_node == false (if DM is selected once per tree)
					//after experiments we found that DM selection per node is better since it diversifies the ensemble
					temp_distance_measure = node.tree.tree_distance_measure;
				}

				temp_distance_measure.select_random_params(data, AppContext.getRand());
			}


							
			splits = split_data(data, data_per_class);
			weighted_gini = weighted_gini(parent_size, splits);

			if (weighted_gini <  best_weighted_gini) {
				best_weighted_gini = weighted_gini;
				best_split = splits;
				distance_measure = temp_distance_measure;
				exemplars = temp_exemplars;
			}
		}

		this.num_children = best_split.length;
		
		return this.best_split;
	}
	
	//public double weighted_gini(int parent_size, ListDataset[] splits) {
	public double weighted_gini(int parent_size, ListObjectDataset[] splits) {
		double wgini = 0.0;
		
		for (int i = 0; i < splits.length; i++) {
			wgini = wgini + ((double) splits[i].size() / parent_size) * splits[i].gini();
		}

		return wgini;
	}	
	
}
