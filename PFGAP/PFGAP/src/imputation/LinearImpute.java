package imputation;

import datasets.ListObjectDataset;
import core.AppContext;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LinearImpute extends Imputer {

    private static final int CHUNK_SIZE = 1000;


    public void Impute(ListObjectDataset missingDS) {
        List<Object> rawData = missingDS.getData();
        MissingIndices mi = missingDS.getMissingIndices();

        if (mi.is2D()) {
            List<Object> result = new ArrayList<>(rawData.size());

            for (int i = 0; i < rawData.size(); i += CHUNK_SIZE) {
                int end = Math.min(i + CHUNK_SIZE, rawData.size());
                List<Object> chunk = rawData.subList(i, end);
                List<List<List<Integer>>> missingChunk = mi.indices2D.subList(i, end);

                List<Object> processedChunk = IntStream.range(0, chunk.size())
                        .parallel()
                        .mapToObj(j -> {
                            Double[][] matrix = (Double[][]) chunk.get(j);
                            List<List<Integer>> missing = missingChunk.get(j);
                            double[][] imputed = convert2DPrimitive(matrix, missing);
                            return (Object) imputed;
                        })
                        .collect(Collectors.toList());

                result.addAll(processedChunk);
                for (int j = i; j < end; j++) rawData.set(j, null);
                System.gc();
            }

            missingDS.setData(result);
        } else {
            List<Object> result = new ArrayList<>(rawData.size());

            for (int i = 0; i < rawData.size(); i += CHUNK_SIZE) {
                int end = Math.min(i + CHUNK_SIZE, rawData.size());
                List<Object> chunk = rawData.subList(i, end);
                List<List<Integer>> missingChunk = mi.indices1D.subList(i, end);

                List<Object> processedChunk = IntStream.range(0, chunk.size())
                        .parallel()
                        .mapToObj(j -> {
                            Double[] row = (Double[]) chunk.get(j);
                            List<Integer> missing = missingChunk.get(j);
                            double[] imputed = convert1DPrimitive(row, missing);
                            return (Object) imputed;
                        })
                        .collect(Collectors.toList());

                result.addAll(processedChunk);
                for (int j = i; j < end; j++) rawData.set(j, null);
                System.gc();
            }

            missingDS.setData(result);
        }
    }

    public static double[] convert1DPrimitive(Double[] row, List<Integer> missingIndices) {
        double[] result = new double[row.length];
        for (int i = 0; i < row.length; i++) {
            result[i] = row[i] != null ? row[i] : Double.NaN;
        }

        // Linear interpolation
        for (int i = 0; i < result.length; i++) {
            if (Double.isNaN(result[i])) {
                int left = i - 1;
                while (left >= 0 && Double.isNaN(result[left])) left--;

                int right = i + 1;
                while (right < result.length && Double.isNaN(result[right])) right++;

                if (left >= 0 && right < result.length) {
                    result[i] = result[left] + (result[right] - result[left]) * (i - left) / (right - left);
                } else if (left >= 0) {
                    result[i] = result[left];
                } else if (right < result.length) {
                    result[i] = result[right];
                } else {
                    result[i] = 0.0;
                }
            }
        }

        return result;
    }

    public static double[][] convert2DPrimitive(Double[][] matrix, List<List<Integer>> missingIndices) {
        double[][] result = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            Double[] row = matrix[i];
            List<Integer> missing = missingIndices.get(i);
            result[i] = convert1DPrimitive(row, missing);
        }
        return result;
    }
}