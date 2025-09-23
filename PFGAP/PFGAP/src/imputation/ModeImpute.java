package imputation;

import datasets.ListObjectDataset;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ModeImpute extends Imputer {

    private static final int CHUNK_SIZE = 1000;


    public static void Impute(ListObjectDataset missingDS) {
        List<Object> rawData = missingDS.getData();
        var mi = missingDS.getMissingIndices();

        if (mi.is2D()) {
            List<Object> result = new ArrayList<>(rawData.size());

            for (int i = 0; i < rawData.size(); i += CHUNK_SIZE) {
                int end = Math.min(i + CHUNK_SIZE, rawData.size());
                List<Object> chunk = rawData.subList(i, end);
                List<List<List<Integer>>> missingChunk = mi.indices2D.subList(i, end);

                List<Object> processedChunk = IntStream.range(0, chunk.size())
                        .parallel()
                        .mapToObj(j -> {
                            Object[][] matrix = (Object[][]) chunk.get(j);
                            List<List<Integer>> missing = missingChunk.get(j);
                            Object[][] imputed = convert2D(matrix, missing);
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
                            Object[] row = (Object[]) chunk.get(j);
                            List<Integer> missing = missingChunk.get(j);
                            Object[] imputed = convert1D(row, missing);
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

    public static Object[] convert1D(Object[] row, List<Integer> missingIndices) {
        Object[] result = Arrays.copyOf(row, row.length);
        Map<Object, Integer> freq = new HashMap<>();

        for (Object val : row) {
            if (val != null) {
                freq.put(val, freq.getOrDefault(val, 0) + 1);
            }
        }

        Object mode = freq.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);

        for (int i : missingIndices) {
            result[i] = mode;
        }

        return result;
    }

    public static Object[][] convert2D(Object[][] matrix, List<List<Integer>> missingIndices) {
        Object[][] result = new Object[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            Object[] row = matrix[i];
            List<Integer> missing = missingIndices.get(i);
            result[i] = convert1D(row, missing);
        }
        return result;
    }
}