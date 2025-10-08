package imputation;

import datasets.ListObjectDataset;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GlobalMeanImpute extends Imputer {

    public void Impute(ListObjectDataset missingDS) {
        List<Object> rawData = missingDS.getData();
        MissingIndices mi = missingDS.getMissingIndices();

        if (mi.is2D()) {
            // Step 1: Compute column-wise means for each row position
            int numRows = ((Double[][]) rawData.get(0)).length;
            int numCols = ((Double[][]) rawData.get(0))[0].length;

            double[][] sums = new double[numRows][numCols];
            int[][] counts = new int[numRows][numCols];

            for (Object instance : rawData) {
                Double[][] matrix = (Double[][]) instance;
                for (int i = 0; i < numRows; i++) {
                    for (int j = 0; j < numCols; j++) {
                        Double val = matrix[i][j];
                        if (val != null) {
                            sums[i][j] += val;
                            counts[i][j]++;
                        }
                    }
                }
            }

            double[][] means = new double[numRows][numCols];
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    means[i][j] = counts[i][j] > 0 ? sums[i][j] / counts[i][j] : 0.0;
                }
            }

            // Step 2: Impute missing values using global means
            List<Object> result = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(i -> {
                        Double[][] matrix = (Double[][]) rawData.get(i);
                        List<List<Integer>> missing = mi.indices2D.get(i);
                        double[][] imputed = new double[numRows][numCols];

                        for (int r = 0; r < numRows; r++) {
                            for (int c = 0; c < numCols; c++) {
                                imputed[r][c] = matrix[r][c] != null ? matrix[r][c] : means[r][c];
                            }
                        }

                        return (Object) imputed;
                    })
                    .collect(Collectors.toList());

            missingDS.setData(result);

        } else {
            // 1D case (same as before)
            int numFeatures = ((Double[]) rawData.get(0)).length;
            double[] sums = new double[numFeatures];
            int[] counts = new int[numFeatures];

            for (Object instance : rawData) {
                Double[] row = (Double[]) instance;
                for (int i = 0; i < row.length; i++) {
                    if (row[i] != null) {
                        sums[i] += row[i];
                        counts[i]++;
                    }
                }
            }

            double[] means = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                means[i] = counts[i] > 0 ? sums[i] / counts[i] : 0.0;
            }

            List<Object> result = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(i -> {
                        Double[] row = (Double[]) rawData.get(i);
                        double[] imputed = new double[row.length];
                        for (int j = 0; j < row.length; j++) {
                            imputed[j] = row[j] != null ? row[j] : means[j];
                        }
                        return (Object) imputed;
                    })
                    .collect(Collectors.toList());

            missingDS.setData(result);
        }
    }
}