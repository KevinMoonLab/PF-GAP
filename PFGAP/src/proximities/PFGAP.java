package proximities;

import core.AppContext;
import datasets.ListObjectDataset;
import trees.ProximityForest;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class PFGAP {

    public static double computeProximity(
            int i,
            int j,
            ProximityForest forest) {

        switch (AppContext.proximityType) {

            case BREIMAN:
                return BreimanProximity.compute(
                        i,
                        j,
                        forest);

            case DEPTH_WEIGHTED:
                return DepthWeightedProximity.compute(
                        i,
                        j,
                        forest
                );

            case PFGAP:
            default:
                return PFGAPProximity.compute(
                        i,
                        j,
                        forest);
        }
    }

    public static double computeTestTrainProximity(
            int testIndex,
            int trainIndex,
            ProximityForest forest) {

        switch (AppContext.proximityType) {

            case BREIMAN:
                return BreimanProximity.computeTestTrain(
                        testIndex,
                        trainIndex,
                        forest);

            case DEPTH_WEIGHTED:
                return DepthWeightedProximity.computeTestTrain(
                        testIndex,
                        trainIndex,
                        forest
                );
            case PFGAP:
            default:
                return PFGAPProximity.computeTestTrain(
                        testIndex,
                        trainIndex,
                        forest);
        }
    }

    public static void computeTrainProximities(
            ProximityForest forest,
            ListObjectDataset train_data)
            throws ExecutionException, InterruptedException {

        int N = train_data.size();

        if (AppContext.useSparseProximities) {

            Map<Integer, Map<Integer, Double>> sparseP =
                    new HashMap<>();

            if (AppContext.parallelProx) {

                ExecutorService executor =
                        Executors.newFixedThreadPool(
                                Runtime.getRuntime()
                                        .availableProcessors());

                List<Future<?>> futures =
                        new ArrayList<>();

                for (int k = 0; k < N; k++) {

                    final int finalK = k;

                    futures.add(executor.submit(() -> {

                        Map<Integer, Double> rowMap =
                                new HashMap<>();

                        for (int j = 0; j < N; j++) {

                            double prox =
                                    computeProximity(
                                            finalK,
                                            j,
                                            forest);

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

                    Map<Integer, Double> rowMap =
                            new HashMap<>();

                    for (int j = 0; j < N; j++) {

                        double prox =
                                computeProximity(
                                        k,
                                        j,
                                        forest);

                        if (prox > 1e-6) {
                            rowMap.put(j, prox);
                        }
                    }

                    sparseP.put(k, rowMap);
                }
            }

            AppContext.training_proximities_sparse =
                    sparseP;

        } else {

            double[][] proximities =
                    new double[N][N];

            if (AppContext.parallelProx) {

                ExecutorService executor =
                        Executors.newFixedThreadPool(
                                Runtime.getRuntime()
                                        .availableProcessors());

                List<Future<?>> futures =
                        new ArrayList<>();

                for (int k = 0; k < N; k++) {

                    final int finalK = k;

                    futures.add(executor.submit(() -> {

                        for (int j = 0; j < N; j++) {

                            double prox =
                                    computeProximity(
                                            finalK,
                                            j,
                                            forest);

                            proximities[finalK][j] =
                                    prox;
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

                        double prox =
                                computeProximity(
                                        k,
                                        j,
                                        forest);

                        proximities[k][j] = prox;
                    }
                }
            }

            AppContext.training_proximities =
                    proximities;
        }
    }

    public static void computeTestTrainProximities(
            ProximityForest forest,
            ListObjectDataset test_data,
            ListObjectDataset train_data)
            throws ExecutionException, InterruptedException {

        int N = train_data.size();
        int K = test_data.size();

        if (AppContext.useSparseProximities) {

            Map<Integer, Map<Integer, Double>> sparseP =
                    new HashMap<>();

            if (AppContext.parallelProx) {

                ExecutorService executor =
                        Executors.newFixedThreadPool(
                                Runtime.getRuntime()
                                        .availableProcessors());

                List<Future<?>> futures =
                        new ArrayList<>();

                for (int k = 0; k < K; k++) {

                    final int finalK = k;

                    futures.add(executor.submit(() -> {

                        Map<Integer, Double> rowMap =
                                new HashMap<>();

                        for (int j = 0; j < N; j++) {

                            double prox =
                                    computeTestTrainProximity(
                                            finalK,
                                            j,
                                            forest);

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

                for (int k = 0; k < K; k++) {

                    Map<Integer, Double> rowMap =
                            new HashMap<>();

                    for (int j = 0; j < N; j++) {

                        double prox =
                                computeTestTrainProximity(
                                        k,
                                        j,
                                        forest);

                        if (prox > 1e-6) {
                            rowMap.put(j, prox);
                        }
                    }

                    sparseP.put(k, rowMap);
                }
            }

            AppContext.testing_training_proximities_sparse =
                    sparseP;

        } else {

            double[][] proximities =
                    new double[K][N];

            if (AppContext.parallelProx) {

                ExecutorService executor =
                        Executors.newFixedThreadPool(
                                Runtime.getRuntime()
                                        .availableProcessors());

                List<Future<?>> futures =
                        new ArrayList<>();

                for (int k = 0; k < K; k++) {

                    final int finalK = k;

                    futures.add(executor.submit(() -> {

                        for (int j = 0; j < N; j++) {

                            double prox =
                                    computeTestTrainProximity(
                                            finalK,
                                            j,
                                            forest);

                            proximities[finalK][j] =
                                    prox;
                        }

                    }));
                }

                for (Future<?> future : futures) {
                    future.get();
                }

                executor.shutdown();

            } else {

                for (int k = 0; k < K; k++) {

                    for (int j = 0; j < N; j++) {

                        double prox =
                                computeTestTrainProximity(
                                        k,
                                        j,
                                        forest);

                        proximities[k][j] = prox;
                    }
                }
            }

            AppContext.testing_training_proximities =
                    proximities;
        }
    }
}