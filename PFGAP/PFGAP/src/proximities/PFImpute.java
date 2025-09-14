package proximities;

import core.AppContext;
import datasets.ListObjectDataset;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PFImpute {

    private static final double EPSILON = 1e-6;

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

                                if (AppContext.useSparseProximities) {
                                    Map<Integer, Double> neighbors = AppContext.training_proximities_sparse.get(n);
                                    for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                        int j = entry.getKey();
                                        double weight = entry.getValue();
                                        double[][] otherMatrix = (double[][]) rawData.get(j);
                                        if (!missingIndices2D.get(j).get(i).contains(k)) {
                                            weightedSum += weight * otherMatrix[i][k];
                                            totalWeight += weight;
                                        }
                                    }
                                } else {
                                    double[][] P = AppContext.training_proximities;
                                    for (int j = 0; j < rawData.size(); j++) {
                                        if (j == n) continue;
                                        double weight = P[n][j];
                                        if (weight > EPSILON && !missingIndices2D.get(j).get(i).contains(k)) {
                                            double[][] otherMatrix = (double[][]) rawData.get(j);
                                            weightedSum += weight * otherMatrix[i][k];
                                            totalWeight += weight;
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

                            if (AppContext.useSparseProximities) {
                                Map<Integer, Double> neighbors = AppContext.training_proximities_sparse.get(n);
                                for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                    int j = entry.getKey();
                                    double weight = entry.getValue();
                                    double[] otherRow = (double[]) rawData.get(j);
                                    if (!missingIndices1D.get(j).contains(k)) {
                                        weightedSum += weight * otherRow[k];
                                        totalWeight += weight;
                                    }
                                }
                            } else {
                                double[][] P = AppContext.training_proximities;
                                for (int j = 0; j < rawData.size(); j++) {
                                    if (j == n) continue;
                                    double weight = P[n][j];
                                    if (weight > EPSILON && !missingIndices1D.get(j).contains(k)) {
                                        double[] otherRow = (double[]) rawData.get(j);
                                        weightedSum += weight * otherRow[k];
                                        totalWeight += weight;
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

                                if (AppContext.useSparseProximities) {
                                    Map<Integer, Double> neighbors = AppContext.testing_training_proximities_sparse.get(j);
                                    for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                        int n = entry.getKey();
                                        double weight = entry.getValue();
                                        double[][] trainMatrix = (double[][]) trainRaw.get(n);
                                        if (!trainData.getMissingIndices().indices2D.get(n).get(i).contains(k)) {
                                            weightedSum += weight * trainMatrix[i][k];
                                            totalWeight += weight;
                                        }
                                    }
                                } else {
                                    double[][] P = AppContext.testing_training_proximities;
                                    for (int n = 0; n < trainRaw.size(); n++) {
                                        double weight = P[j][n];
                                        if (weight > EPSILON &&
                                                !trainData.getMissingIndices().indices2D.get(n).get(i).contains(k)) {
                                            double[][] trainMatrix = (double[][]) trainRaw.get(n);
                                            weightedSum += weight * trainMatrix[i][k];
                                            totalWeight += weight;
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

                            if (AppContext.useSparseProximities) {
                                Map<Integer, Double> neighbors = AppContext.testing_training_proximities_sparse.get(j);
                                for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {
                                    int n = entry.getKey();
                                    double weight = entry.getValue();
                                    double[] trainRow = (double[]) trainRaw.get(n);
                                    if (!trainData.getMissingIndices().indices1D.get(n).contains(k)) {
                                        weightedSum += weight * trainRow[k];
                                        totalWeight += weight;
                                    }
                                }
                            } else {
                                double[][] P = AppContext.testing_training_proximities;
                                for (int n = 0; n < trainRaw.size(); n++) {
                                    double weight = P[j][n];
                                    if (weight > EPSILON &&
                                            !trainData.getMissingIndices().indices1D.get(n).contains(k)) {
                                        double[] trainRow = (double[]) trainRaw.get(n);
                                        weightedSum += weight * trainRow[k];
                                        totalWeight += weight;
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


    public static void trainCategoricalImpute(ListObjectDataset dataToUpdate) {
        // To be implemented later
    }

    public static void testCategoricalImpute(ListObjectDataset dataToUpdate) {
        // To be implemented later
    }

    public static Map<Integer, Map<Integer, Double>> buildSparseProximityMap(double[][] P, double epsilon) {
        Map<Integer, Map<Integer, Double>> sparseMap = new HashMap<>();
        for (int i = 0; i < P.length; i++) {
            Map<Integer, Double> rowMap = new HashMap<>();
            for (int j = 0; j < P[i].length; j++) {
                if (P[i][j] > epsilon) {
                    rowMap.put(j, P[i][j]);
                }
            }
            sparseMap.put(i, rowMap);
        }
        return sparseMap;
    }
}
