package core;

import trees.ProximityForest;
import trees.ProximityTree;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class PFGAP{

    //Here is the original code.
    /*public static ArrayList<ProximityTree> getSi(Integer i, ProximityForest pf){
        ArrayList<ProximityTree> Si = new ArrayList<ProximityTree>();
        ProximityTree[] trees = pf.getTrees();
        for(ProximityTree tree:trees){
            ArrayList<Integer> oob = tree.getRootNode().getOutOfBagIndices();
            if(oob.contains(i)){
                Si.add(tree);
            }
        }
        return Si;
    }*/

    public static ArrayList<ProximityTree> getSi(Integer i, ProximityForest pf) {
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
    }


    //public static ProximityTree.Node getJiLeaf(Integer i, ProximityTree t){
    //Here is the original code
    /*public static ArrayList<Integer> getJi(Integer i, ProximityTree t){
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
    }*/

    public static ArrayList<Integer> getJi(Integer i, ProximityTree t) {
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
    }


    //Below is the original code
    /*public static Double ForestProximity(Integer i, Integer j, ProximityForest pf){
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
    }*/

    public static Double ForestProximity(Integer i, Integer j, ProximityForest pf) {
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
    }


}