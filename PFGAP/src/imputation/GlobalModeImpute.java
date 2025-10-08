package imputation;

import datasets.ListObjectDataset;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GlobalModeImpute extends Imputer {
    //per column imputation as opposed to per-instance imputation.

    public void Impute(ListObjectDataset missingDS) {
        List<Object> rawData = missingDS.getData();
        var mi = missingDS.getMissingIndices();

        if (mi.is2D()) {
            int numRows = ((Object[][]) rawData.get(0)).length;
            int numCols = ((Object[][]) rawData.get(0))[0].length;

            Map<Object, Integer>[][] freqTables = new Map[numRows][numCols];
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    freqTables[i][j] = new HashMap<>();
                }
            }

            for (Object instance : rawData) {
                Object[][] matrix = (Object[][]) instance;
                for (int i = 0; i < numRows; i++) {
                    for (int j = 0; j < numCols; j++) {
                        Object val = matrix[i][j];
                        if (val != null) {
                            freqTables[i][j].put(val, freqTables[i][j].getOrDefault(val, 0) + 1);
                        }
                    }
                }
            }

            Object[][] modes = new Object[numRows][numCols];
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    modes[i][j] = freqTables[i][j].entrySet().stream()
                            .max(Map.Entry.comparingByValue())
                            .map(Map.Entry::getKey)
                            .orElse(null);
                }
            }

            List<Object> result = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(i -> {
                        Object[][] matrix = (Object[][]) rawData.get(i);
                        Object[][] imputed = new Object[numRows][numCols];
                        for (int r = 0; r < numRows; r++) {
                            for (int c = 0; c < numCols; c++) {
                                imputed[r][c] = matrix[r][c] != null ? matrix[r][c] : modes[r][c];
                            }
                        }
                        return (Object) imputed;
                    })
                    .collect(Collectors.toList());

            missingDS.setData(result);

        } else {
            int numFeatures = ((Object[]) rawData.get(0)).length;
            Map<Object, Integer>[] freqTables = new Map[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                freqTables[i] = new HashMap<>();
            }

            for (Object instance : rawData) {
                Object[] row = (Object[]) instance;
                for (int i = 0; i < row.length; i++) {
                    Object val = row[i];
                    if (val != null) {
                        freqTables[i].put(val, freqTables[i].getOrDefault(val, 0) + 1);
                    }
                }
            }

            Object[] modes = new Object[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                modes[i] = freqTables[i].entrySet().stream()
                        .max(Map.Entry.comparingByValue())
                        .map(Map.Entry::getKey)
                        .orElse(null);
            }

            List<Object> result = IntStream.range(0, rawData.size())
                    .parallel()
                    .mapToObj(i -> {
                        Object[] row = (Object[]) rawData.get(i);
                        Object[] imputed = new Object[row.length];
                        for (int j = 0; j < row.length; j++) {
                            imputed[j] = row[j] != null ? row[j] : modes[j];
                        }
                        return (Object) imputed;
                    })
                    .collect(Collectors.toList());

            missingDS.setData(result);
        }
    }
}