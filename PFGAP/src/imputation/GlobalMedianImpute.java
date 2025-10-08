package imputation;

import datasets.ListObjectDataset;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GlobalMedianImpute extends Imputer {

    public void Impute(ListObjectDataset missingDS) {
        List<Object> rawData = missingDS.getData();
        MissingIndices mi = missingDS.getMissingIndices();

        if (mi.is2D()) {
            int numRows = ((Double[][]) rawData.get(0)).length;
            int numCols = ((Double[][]) rawData.get(0))[0].length;

            List<Double>[][] values = new List[numRows][numCols];
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    values[i][j] = new ArrayList<>();
                }
            }

            for (Object instance : rawData) {
                Double[][] matrix = (Double[][]) instance;
                for (int i = 0; i < numRows; i++) {
                    for (int j = 0; j < numCols; j++) {
                        Double val = matrix[i][j];
                        if (val != null) {
                            values[i][j].add(val);
                        }
                    }
                }
            }

            double[][] medians = new double[numRows][numCols];
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    medians[i][j] = computeMedian(values[i][j]);
                }
            }

            List<Object> result = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(i -> {
                        Double[][] matrix = (Double[][]) rawData.get(i);
                        double[][] imputed = new double[numRows][numCols];
                        for (int r = 0; r < numRows; r++) {
                            for (int c = 0; c < numCols; c++) {
                                imputed[r][c] = matrix[r][c] != null ? matrix[r][c] : medians[r][c];
                            }
                        }
                        return (Object) imputed;
                    })
                    .collect(Collectors.toList());

            missingDS.setData(result);

        } else {
            int numFeatures = ((Double[]) rawData.get(0)).length;
            List<Double>[] values = new List[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                values[i] = new ArrayList<>();
            }

            for (Object instance : rawData) {
                Double[] row = (Double[]) instance;
                for (int i = 0; i < row.length; i++) {
                    if (row[i] != null) {
                        values[i].add(row[i]);
                    }
                }
            }

            double[] medians = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                medians[i] = computeMedian(values[i]);
            }

            List<Object> result = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(i -> {
                        Double[] row = (Double[]) rawData.get(i);
                        double[] imputed = new double[row.length];
                        for (int j = 0; j < row.length; j++) {
                            imputed[j] = row[j] != null ? row[j] : medians[j];
                        }
                        return (Object) imputed;
                    })
                    .collect(Collectors.toList());

            missingDS.setData(result);
        }
    }

    private static double computeMedian(List<Double> values) {
        if (values.isEmpty()) return 0.0;
        Collections.sort(values);
        int n = values.size();
        return (n % 2 == 1)
                ? values.get(n / 2)
                : (values.get(n / 2 - 1) + values.get(n / 2)) / 2.0;
    }
}