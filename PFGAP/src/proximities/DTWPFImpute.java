package proximities;

import core.AppContext;
import datasets.ListObjectDataset;
import distance.elastic.DTWWithPath;
import distance.multiTS.DTW_D;
import imputation.MissingIndices;
import util.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DTWPFImpute {
    private static final double EPSILON = 1e-6;

    // Assumes you have a cache or method to retrieve alignment paths
    public static Map<Integer, Map<Integer, List<Pair<Integer, Integer>>>> alignmentPaths;

    public static void trainNumericImpute(ListObjectDataset dataToUpdate) {
        List<Object> rawData = dataToUpdate.getData();
        if (dataToUpdate.getMissingIndices().is2D()) {
            List<List<List<Integer>>> missingIndices2D = dataToUpdate.getMissingIndices().indices2D;
            List<Object> updated = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(n -> {
                        double[][] matrix = (double[][]) rawData.get(n);
                        List<List<Integer>> missingRows = missingIndices2D.get(n);
                        double[][] updatedMatrix = new double[matrix.length][];
                        for (int i = 0; i < matrix.length; i++) {
                            double[] row = matrix[i];
                            List<Integer> missing = missingRows.get(i);
                            double[] updatedRow = Arrays.copyOf(row, row.length);
                            for (int k : missing) {
                                double weightedSum = 0;
                                double totalWeight = 0;
                                Map<Integer, Double> neighbors = AppContext.useSparseProximities
                                        ? AppContext.training_proximities_sparse.get(n)
                                        : convertDenseRowToMap(AppContext.training_proximities[n]);

                                for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                    int j = entry.getKey();
                                    double weight = entry.getValue();
                                    double[][] otherMatrix = (double[][]) rawData.get(j);

                                    List<Pair<Integer, Integer>> path = getAlignmentPath(n, j);
                                    for (Pair<Integer, Integer> pair : path) {
                                        if (pair.getKey() == i && pair.getValue() == k) {
                                            int alignedRow = pair.getKey();
                                            int alignedCol = pair.getValue();
                                            if (!missingIndices2D.get(j).get(alignedRow).contains(alignedCol)) {
                                                weightedSum += weight * otherMatrix[alignedRow][alignedCol];
                                                totalWeight += weight;
                                            }
                                        }
                                    }
                                }
                                updatedRow[k] = totalWeight > 0 ? weightedSum / totalWeight : row[k];
                            }
                            updatedMatrix[i] = updatedRow;
                        }
                        return (Object) updatedMatrix;
                    })
                    .collect(Collectors.toList());
            dataToUpdate.setData(updated);
        } else {
            List<List<Integer>> missingIndices1D = dataToUpdate.getMissingIndices().indices1D;
            List<Object> updated = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(n -> {
                        double[] row = (double[]) rawData.get(n);
                        List<Integer> missing = missingIndices1D.get(n);
                        double[] updatedRow = Arrays.copyOf(row, row.length);
                        for (int k : missing) {
                            double weightedSum = 0;
                            double totalWeight = 0;
                            Map<Integer, Double> neighbors = AppContext.useSparseProximities
                                    ? AppContext.training_proximities_sparse.get(n)
                                    : convertDenseRowToMap(AppContext.training_proximities[n]);

                            for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                int j = entry.getKey();
                                double weight = entry.getValue();
                                double[] otherRow = (double[]) rawData.get(j);

                                List<Pair<Integer, Integer>> path = getAlignmentPath(n, j);
                                for (Pair<Integer, Integer> pair : path) {
                                    if (pair.getKey() == k) {
                                        int alignedIndex = pair.getValue();
                                        if (!missingIndices1D.get(j).contains(alignedIndex)) {
                                            weightedSum += weight * otherRow[alignedIndex];
                                            totalWeight += weight;
                                        }
                                    }
                                }
                            }
                            updatedRow[k] = totalWeight > 0 ? weightedSum / totalWeight : row[k];
                        }
                        return (Object) updatedRow;
                    })
                    .collect(Collectors.toList());
            dataToUpdate.setData(updated);
        }
    }

    public static void testNumericImpute(ListObjectDataset testData, ListObjectDataset trainData) {
        List<Object> testRaw = testData.getData();
        List<Object> trainRaw = trainData.getData();

        if (testData.getMissingIndices().is2D()) {
            List<List<List<Integer>>> missingIndices2D = testData.getMissingIndices().indices2D;
            List<Object> updated = IntStream.range(0, testRaw.size())
                    .parallel()
                    .mapToObj(j -> {
                        double[][] matrix = (double[][]) testRaw.get(j);
                        List<List<Integer>> missingRows = missingIndices2D.get(j);
                        double[][] updatedMatrix = new double[matrix.length][];
                        for (int i = 0; i < matrix.length; i++) {
                            double[] row = matrix[i];
                            List<Integer> missing = missingRows.get(i);
                            double[] updatedRow = Arrays.copyOf(row, row.length);
                            for (int k : missing) {
                                double weightedSum = 0;
                                double totalWeight = 0;
                                Map<Integer, Double> neighbors = AppContext.useSparseProximities
                                        ? AppContext.testing_training_proximities_sparse.get(j)
                                        : convertDenseRowToMap(AppContext.testing_training_proximities[j]);

                                for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                    int n = entry.getKey(); // train index
                                    double weight = entry.getValue();
                                    double[][] trainMatrix = (double[][]) trainRaw.get(n);

                                    List<Pair<Integer, Integer>> path = getAlignmentPath(j, n);
                                    for (Pair<Integer, Integer> pair : path) {
                                        if (pair.getKey() == i && pair.getValue() == k) {
                                            int alignedRow = pair.getKey();
                                            int alignedCol = pair.getValue();
                                            if (!trainData.getMissingIndices().indices2D.get(n).get(alignedRow).contains(alignedCol)) {
                                                weightedSum += weight * trainMatrix[alignedRow][alignedCol];
                                                totalWeight += weight;
                                            }
                                        }
                                    }
                                }
                                updatedRow[k] = totalWeight > 0 ? weightedSum / totalWeight : row[k];
                            }
                            updatedMatrix[i] = updatedRow;
                        }
                        return (Object) updatedMatrix;
                    })
                    .collect(Collectors.toList());
            testData.setData(updated);
        } else {
            List<List<Integer>> missingIndices1D = testData.getMissingIndices().indices1D;
            List<Object> updated = IntStream.range(0, testRaw.size())
                    .parallel()
                    .mapToObj(j -> {
                        double[] row = (double[]) testRaw.get(j);
                        List<Integer> missing = missingIndices1D.get(j);
                        double[] updatedRow = Arrays.copyOf(row, row.length);
                        for (int k : missing) {
                            double weightedSum = 0;
                            double totalWeight = 0;
                            Map<Integer, Double> neighbors = AppContext.useSparseProximities
                                    ? AppContext.testing_training_proximities_sparse.get(j)
                                    : convertDenseRowToMap(AppContext.testing_training_proximities[j]);

                            for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                int n = entry.getKey(); // train index
                                double weight = entry.getValue();
                                double[] trainRow = (double[]) trainRaw.get(n);

                                List<Pair<Integer, Integer>> path = getAlignmentPath(j, n);
                                for (Pair<Integer, Integer> pair : path) {
                                    if (pair.getKey() == k) {
                                        int alignedIndex = pair.getValue();
                                        if (!trainData.getMissingIndices().indices1D.get(n).contains(alignedIndex)) {
                                            weightedSum += weight * trainRow[alignedIndex];
                                            totalWeight += weight;
                                        }
                                    }
                                }
                            }
                            updatedRow[k] = totalWeight > 0 ? weightedSum / totalWeight : row[k];
                        }
                        return (Object) updatedRow;
                    })
                    .collect(Collectors.toList());
            testData.setData(updated);
        }
    }

    private static Map<Integer, Double> convertDenseRowToMap(double[] row) {
        Map<Integer, Double> map = new HashMap<>();
        for (int i = 0; i < row.length; i++) {
            if (row[i] > EPSILON) {
                map.put(i, row[i]);
            }
        }
        return map;
    }

    private static List<Pair<Integer, Integer>> getAlignmentPath(int i, int j) {
        return alignmentPaths.getOrDefault(i, Collections.emptyMap()).getOrDefault(j, Collections.emptyList());
    }

    public static void buildAlignmentPathCache(
            ListObjectDataset dataA,
            ListObjectDataset dataB,
            double[][] proximities,
            boolean is2D,
            int windowSize
    ) {
        int sizeA = dataA.size();
        int sizeB = dataB.size();

        DTWWithPath dtw1D = new DTWWithPath();
        DTW_D dtw2D = new DTW_D();

        alignmentPaths = new HashMap<>();

        for (int i = 0; i < sizeA; i++) {
            for (int j = 0; j < sizeB; j++) {
                if (i == j && dataA == dataB) continue;
                if (proximities[i][j] <= EPSILON) continue;

                Object s1 = dataA.get_series(i);
                Object s2 = dataB.get_series(j);

                List<Pair<Integer, Integer>> path;
                if (is2D) {
                    path = dtw2D.getAlignmentPath((double[][]) s1, (double[][]) s2, windowSize);
                } else {
                    path = dtw1D.getAlignmentPath((double[]) s1, (double[]) s2, windowSize);
                }

                alignmentPaths
                        .computeIfAbsent(i, k -> new HashMap<>())
                        .put(j, path);
            }
        }
    }
}