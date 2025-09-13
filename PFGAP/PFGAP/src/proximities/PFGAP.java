package proximities;

import core.AppContext;
import datasets.ListObjectDataset;
import trees.ProximityForest;
import trees.ProximityTree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class PFGAP{

    //Here is the original code.
    public static ArrayList<ProximityTree> getSi(Integer i, ProximityForest pf){
        ArrayList<ProximityTree> Si = new ArrayList<ProximityTree>();
        ProximityTree[] trees = pf.getTrees();
        for(ProximityTree tree:trees){
            ArrayList<Integer> oob = tree.getRootNode().getOutOfBagIndices();
            if(oob.contains(i)){
                Si.add(tree);
            }
        }
        return Si;
    }

    //Here is the parallel code.
    /*public static ArrayList<ProximityTree> getSi(Integer i, ProximityForest pf) {
        ArrayList<ProximityTree> Si = new ArrayList<>();
        ProximityTree[] trees = pf.getTrees();

        if (AppContext.parallelProx) {
            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            List<Future<ProximityTree>> futures = new ArrayList<>();

            for (ProximityTree tree : trees) {
                futures.add(executor.submit(() -> {
                    if (tree.getRootNode().getOutOfBagIndices().contains(i)) {
                        return tree;
                    }
                    return null;
                }));
            }

            for (Future<ProximityTree> future : futures) {
                try {
                    ProximityTree result = future.get();
                    if (result != null) {
                        Si.add(result);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            executor.shutdown();
        } else {
            for (ProximityTree tree : trees) {
                if (tree.getRootNode().getOutOfBagIndices().contains(i)) {
                    Si.add(tree);
                }
            }
        }

        return Si;
    }*/


    //public static ProximityTree.Node getJiLeaf(Integer i, ProximityTree t){
    //Here is the original code
    public static ArrayList<Integer> getJi(Integer i, ProximityTree t){
        ArrayList<Integer> Ji = new ArrayList<>();
        ArrayList<ProximityTree.Node> leaves = t.getLeaves();
        //System.out.println(leaves.size());
        for (ProximityTree.Node leaf : leaves){
            ArrayList<Integer> inbags = leaf.getInBagIndices();
            ArrayList<Integer> oob = leaf.getOutOfBagIndices();
            //System.out.println(oob);
            //System.out.println(i);
            if(oob.contains(i)){
                Ji = inbags;
                //System.out.print(Ji);
            }
        }
        return Ji;
    }

    // Here is the parallel code.
    /*public static ArrayList<Integer> getJi(Integer i, ProximityTree t) {
        ArrayList<Integer> Ji = new ArrayList<>();
        ArrayList<ProximityTree.Node> leaves = t.getLeaves();

        if (AppContext.parallelProx) {
            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            List<Future<ArrayList<Integer>>> futures = new ArrayList<>();

            for (ProximityTree.Node leaf : leaves) {
                futures.add(executor.submit(() -> {
                    if (leaf.getOutOfBagIndices().contains(i)) {
                        return leaf.getInBagIndices();
                    }
                    return null;
                }));
            }

            for (Future<ArrayList<Integer>> future : futures) {
                try {
                    ArrayList<Integer> result = future.get();
                    if (result != null) {
                        Ji = result; // Overwrite with the last matching leaf
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            executor.shutdown();
        } else {
            for (ProximityTree.Node leaf : leaves) {
                if (leaf.getOutOfBagIndices().contains(i)) {
                    Ji = leaf.getInBagIndices();
                }
            }
        }

        return Ji;
    }*/


    //Below is the original code
    public static Double ForestProximity(Integer i, Integer j, ProximityForest pf){
        ArrayList<ProximityTree> Si = getSi(i,pf);
        //Double[] terms = new Double[]{};
        ArrayList<Double> terms = new ArrayList<>();
        for (ProximityTree t : Si){
            Integer cj = t.getRootNode().getMultiplicities().get(j);
            ArrayList<Integer> Mi = getJi(i,t);
            if (Mi.contains(j)){
                double Cj = (double) cj;
                terms.add((Cj/Mi.size())/ Si.size());
            }
            else{
                terms.add((double) 0);
            }
        }
        double sum = 0;
        for (Double term : terms){
            sum += term;
        }
        return sum;
    }

    // Here is the parallel code.
    /*public static Double ForestProximity(Integer i, Integer j, ProximityForest pf) {
        ArrayList<ProximityTree> Si = getSi(i, pf);
        ArrayList<Double> terms = new ArrayList<>();

        if (AppContext.parallelProx) {
            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            List<Future<Double>> futures = new ArrayList<>();

            for (ProximityTree t : Si) {
                futures.add(executor.submit(() -> {
                    Integer cj = t.getRootNode().getMultiplicities().get(j);
                    ArrayList<Integer> Mi = getJi(i, t);
                    if (Mi.contains(j)) {
                        double Cj = (double) cj;
                        return (Cj / Mi.size()) / Si.size();
                    } else {
                        return 0.0;
                    }
                }));
            }

            for (Future<Double> future : futures) {
                try {
                    terms.add(future.get());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            executor.shutdown();
        } else {
            for (ProximityTree t : Si) {
                Integer cj = t.getRootNode().getMultiplicities().get(j);
                ArrayList<Integer> Mi = getJi(i, t);
                if (Mi.contains(j)) {
                    double Cj = (double) cj;
                    terms.add((Cj / Mi.size()) / Si.size());
                } else {
                    terms.add(0.0);
                }
            }
        }

        double sum = 0;
        for (Double term : terms) {
            sum += term;
        }

        return sum;
    }*/


    public static void computeTrainProximities(ProximityForest forest, ListObjectDataset train_data) throws ExecutionException, InterruptedException {
        int N = train_data.size();

        if (AppContext.useSparseProximities) {
            Map<Integer, Map<Integer, Double>> sparseP = new HashMap<>();

            if (AppContext.parallelProx) {
                ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                List<Future<?>> futures = new ArrayList<>();

                for (int k = 0; k < N; k++) {
                    final int finalK = k;
                    futures.add(executor.submit(() -> {
                        Map<Integer, Double> rowMap = new HashMap<>();
                        for (int j = 0; j < N; j++) {
                            double prox = ForestProximity(finalK, j, forest);
                            if (prox > 1e-6) {
                                rowMap.put(j, prox);
                            }
                        }
                        synchronized (sparseP) {
                            sparseP.put(finalK, rowMap);
                        }
                    }));
                }

                for (Future<?> future : futures) {
                    future.get();
                }

                executor.shutdown();

            } else {
                for (int k = 0; k < N; k++) {
                    Map<Integer, Double> rowMap = new HashMap<>();
                    for (int j = 0; j < N; j++) {
                        double prox = ForestProximity(k, j, forest);
                        if (prox > 1e-6) {
                            rowMap.put(j, prox);
                        }
                    }
                    sparseP.put(k, rowMap);
                }
            }

            AppContext.training_proximities_sparse = sparseP;

        } else {
            double[][] PFGAP = new double[N][N];

            if (AppContext.parallelProx) {
                ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                List<Future<?>> futures = new ArrayList<>();

                for (int k = 0; k < N; k++) {
                    final int finalK = k;
                    futures.add(executor.submit(() -> {
                        for (int j = 0; j < N; j++) {
                            double prox = ForestProximity(finalK, j, forest);
                            PFGAP[finalK][j] = prox;
                        }
                    }));
                }

                for (Future<?> future : futures) {
                    future.get();
                }

                executor.shutdown();

            } else {
                for (int k = 0; k < N; k++) {
                    for (int j = 0; j < N; j++) {
                        double prox = ForestProximity(k, j, forest);
                        PFGAP[k][j] = prox;
                    }
                }
            }

            AppContext.training_proximities = PFGAP;
        }
    }



    /*public static void computeTrainProximities(ProximityForest forest, ListObjectDataset train_data) throws ExecutionException, InterruptedException {
        double[][] PFGAP = new double[train_data.size()][train_data.size()];
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
        AppContext.training_proximities = PFGAP;
    }*/


}