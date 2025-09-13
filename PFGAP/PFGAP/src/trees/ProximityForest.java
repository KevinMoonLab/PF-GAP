package trees;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.locks.ReentrantLock;

import core.AppContext;
import core.ProximityForestResult;
//import core.contracts.Dataset;
import core.contracts.ObjectDataset;
//import datasets.ListDataset;
import datasets.ListObjectDataset;
import distance.DistanceMeasure;
import distance.MEASURE;
import util.PrintUtilities;
/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ProximityForest implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -1183368028217094381L;
	//private final int r; //is this needed? number of candidates to consider.
	//protected transient ProximityForestResult result;
	protected ProximityForestResult result;
	protected int forest_id;
	protected ProximityTree trees[];
	public String prefix;
	
	int[] num_votes;
	List<Integer> max_voted_classes;

	private final ReentrantLock trainLock = new ReentrantLock();
	private final ReentrantLock predictLock = new ReentrantLock();


	public ProximityForest(int forest_id, MEASURE... selected_distances) {
		this.result = new ProximityForestResult(this);
		System.out.println(this.result);
		
		this.forest_id = forest_id;
		this.trees = new ProximityTree[AppContext.num_trees];
		
		for (int i = 0; i < AppContext.num_trees; i++) {
			trees[i] = new ProximityTree(i, this, selected_distances);
		}

	}
	// This was used as a debugging tool:
	/*public ProximityForest(int num_trees, int r) {
		this.result = new ProximityForestResult(this);
		this.r = r;
		this.forest_id = forest_id;
		this.trees = new ProximityTree[num_trees]; //ProximityTree[AppContext.num_trees];

		for (int i = 0; i < num_trees; i++) {
			trees[i] = new ProximityTree(i, this);
		}

	}*/

	// The following is the previous "train" method before the parallel option was made.
	/*public void train(Dataset train_data) throws Exception {
	//public void train(ListDataset train_data) throws Exception {
		result.startTimeTrain = System.nanoTime();

		for (int i = 0; i < this.trees.length; i++) {
			trees[i].train(train_data);
			
			if (AppContext.verbosity > 0) {
				System.out.print(i+".");
				if (AppContext.verbosity > 1) {
					PrintUtilities.printMemoryUsage(true);	
					if ((i+1) % 20 == 0) {
						System.out.println();
					}
				}		
			}

		}
		
		result.endTimeTrain = System.nanoTime();
		result.elapsedTimeTrain = result.endTimeTrain - result.startTimeTrain;
		
		if (AppContext.verbosity > 0) {
			System.out.print("\n");				
		}
		
//		System.gc();
		if (AppContext.verbosity > 0) {
			PrintUtilities.printMemoryUsage();	
		}
	
	}*/

	//public void train(Dataset train_data) throws Exception {
	public void train(ListObjectDataset train_data) throws Exception {
		trainLock.lock();
		try {
			result.startTimeTrain = System.nanoTime();

			if (AppContext.parallelTrees) {
				ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
				List<Future<?>> futures = new ArrayList<>();

				for (int i = 0; i < trees.length; i++) {
					final int index = i;
					futures.add(executor.submit(() -> {
						try {
							trees[index].train(train_data);
						} catch (Exception e) {
							e.printStackTrace();
						}
						if (AppContext.verbosity > 0) {
							synchronized (System.out) {
								System.out.print(index + ".");
								if (AppContext.verbosity > 1) {
									PrintUtilities.printMemoryUsage(true);
									if ((index + 1) % 20 == 0) {
										System.out.println();
									}
								}
							}
						}
					}));
				}

				for (Future<?> future : futures) {
					future.get();
				}

				executor.shutdown();
			} else {
				for (int i = 0; i < trees.length; i++) {
					trees[i].train(train_data);
					if (AppContext.verbosity > 0) {
						System.out.print(i + ".");
						if (AppContext.verbosity > 1) {
							PrintUtilities.printMemoryUsage(true);
							if ((i + 1) % 20 == 0) {
								System.out.println();
							}
						}
					}
				}
			}

			result.endTimeTrain = System.nanoTime();
			result.elapsedTimeTrain = result.endTimeTrain - result.startTimeTrain;

			if (AppContext.verbosity > 0) {
				System.out.print("\n");
				PrintUtilities.printMemoryUsage();
			}
		} finally {
			trainLock.unlock();
		}
	}



	//This is the previous test() method code before parallelization.
	//ASSUMES CLASS labels HAVE BEEN reordered to start from 0 and contiguous
	//public ProximityForestResult test(Dataset test_data) throws Exception {
	public ProximityForestResult test(ListObjectDataset test_data) throws Exception {
		result.startTimeTest = System.nanoTime();
		//result.Predictions = new ArrayList<>();
		num_votes = new int[test_data._get_initial_class_labels().size()]; //new int[test_data.length()]; //new int[test_data._get_initial_class_labels().size()];
		max_voted_classes = new ArrayList<Integer>();
		//ArrayList<Integer> Predictions = new ArrayList<>();
		
		int predicted_class;
		int actual_class;
		int size = test_data.size();
		
		for (int i=0; i < size; i++){
			actual_class = test_data.get_class(i);
			predicted_class = predict(test_data.get_series(i));
			result.Predictions.add(predicted_class);
			if (actual_class != predicted_class){
				result.errors++;
			}else{
				result.correct++;
			}
			
			if (AppContext.verbosity > 0) {
				if (i % AppContext.print_test_progress_for_each_instances == 0) {
					System.out.print("*");
				}				
			}
		}
		
		result.endTimeTest = System.nanoTime();
		result.elapsedTimeTest = result.endTimeTest - result.startTimeTest;
		
		if (AppContext.verbosity > 0) {
			System.out.println();
		}
		
		
		assert test_data.size() == result.errors + result.correct;		
		result.accuracy  = ((double) result.correct) / test_data.size();
		result.error_rate = 1 - result.accuracy;

        return result;
	}

	//This is the work-in-progress parallel test method.
	/*public ProximityForestResult test(Dataset test_data) throws Exception {
		result.startTimeTest = System.nanoTime();
		int size = test_data.size();

		ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		List<Future<Integer>> futures = new ArrayList<>();

		for (int i = 0; i < size; i++) {
			final int index = i;
			futures.add(executor.submit(() -> {
				double[] query = test_data.get_series(index);
				return predict(query);
			}));
		}

		for (int i = 0; i < size; i++) {
			int predicted_class = futures.get(i).get();
			int actual_class = test_data.get_class(i);

			result.Predictions.add(predicted_class);
			if (actual_class != predicted_class) {
				result.errors++;
			} else {
				result.correct++;
			}

			if (AppContext.verbosity > 0 && i % AppContext.print_test_progress_for_each_instances == 0) {
				System.out.print("*");
			}
		}

		executor.shutdown();

		result.endTimeTest = System.nanoTime();
		result.elapsedTimeTest = result.endTimeTest - result.startTimeTest;

		if (AppContext.verbosity > 0) {
			System.out.println();
		}

		assert test_data.size() == result.errors + result.correct;
		result.accuracy = ((double) result.correct) / test_data.size();
		result.error_rate = 1 - result.accuracy;

		return result;
	}*/




	//public Integer predict(double[] query) throws Exception {
	public Integer predict(Object query) throws Exception {
		predictLock.lock();
		try {
			int max_vote_count = -1;
			int temp_count;

			if (AppContext.parallelTrees) {
				AtomicIntegerArray voteCounts = new AtomicIntegerArray(num_votes.length);
				ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
				List<Future<?>> futures = new ArrayList<>();

				for (int i = 0; i < trees.length; i++) {
					final int index = i;
					futures.add(executor.submit(() -> {
						int label = 0;
						try {
							label = trees[index].predict(query);
						} catch (Exception e) {
							e.printStackTrace();
						}
						voteCounts.incrementAndGet(label);
					}));
				}

				for (Future<?> future : futures) {
					future.get();
				}

				executor.shutdown();

				max_voted_classes.clear();
				for (int i = 0; i < voteCounts.length(); i++) {
					temp_count = voteCounts.get(i);
					if (temp_count > max_vote_count) {
						max_vote_count = temp_count;
						max_voted_classes.clear();
						max_voted_classes.add(i);
					} else if (temp_count == max_vote_count) {
						max_voted_classes.add(i);
					}
				}
			} else {
				int label;
				for (int i = 0; i < num_votes.length; i++) {
					num_votes[i] = 0;
				}
				max_voted_classes.clear();

				for (int i = 0; i < trees.length; i++) {
					label = trees[i].predict(query);
					num_votes[label]++;
				}

				for (int i = 0; i < num_votes.length; i++) {
					temp_count = num_votes[i];
					if (temp_count > max_vote_count) {
						max_vote_count = temp_count;
						max_voted_classes.clear();
						max_voted_classes.add(i);
					} else if (temp_count == max_vote_count) {
						max_voted_classes.add(i);
					}
				}
			}

			int r = AppContext.getRand().nextInt(max_voted_classes.size());

			if (max_voted_classes.size() > 1) {
				this.result.majority_vote_match_count++;
			}

			return max_voted_classes.get(r);
		} finally {
			predictLock.unlock();
		}
	}



	//This is the previous code for the predict method before the parallel option.
	/*public Integer predict(double[] query) throws Exception {
		//ASSUMES CLASSES HAVE BEEN REMAPPED, start from 0
		int label;
		int max_vote_count = -1;
		int temp_count = 0;
		
		for (int i = 0; i < num_votes.length; i++) {
			num_votes[i] = 0;
		}
		max_voted_classes.clear();

		for (int i = 0; i < trees.length; i++) {
			label = trees[i].predict(query);
			
			num_votes[label]++;
		}
		
//			System.out.println("vote counting using uni dist");
			
		for (int i = 0; i < num_votes.length; i++) {
			temp_count = num_votes[i];
			
			if (temp_count > max_vote_count) {
				max_vote_count = temp_count;
				max_voted_classes.clear();
				max_voted_classes.add(i);
			}else if (temp_count == max_vote_count) {
				max_voted_classes.add(i);
			}
		}
		
		int r = AppContext.getRand().nextInt(max_voted_classes.size());
		
		//collecting some stats
		if (max_voted_classes.size() > 1) {
			this.result.majority_vote_match_count++;
		}
		
		return max_voted_classes.get(r);
	}*/
	
	public ProximityTree[] getTrees() {
		return this.trees;
	}
	
	public ProximityTree getTree(int i) {
		return this.trees[i];
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

	public void setForestID(int forest_id) {
		this.forest_id = forest_id;
	}




	
}
