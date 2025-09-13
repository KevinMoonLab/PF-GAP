package imputation;

import core.AppContext;
import datasets.ListObjectDataset;
import util.Statistics;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

// We need to be working with List<Double[]> or List<Double[][]>

public class MeanImpute extends Imputer {

    private static final int CHUNK_SIZE = 1000;

    public static void Impute(ListObjectDataset missingDS) {
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

                // Erase original data
                for (int j = i; j < end; j++) {
                    rawData.set(j, null);
                }

                System.gc(); // Hint GC
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

                // Erase original data
                for (int j = i; j < end; j++) {
                    rawData.set(j, null);
                }

                System.gc(); // Hint GC
            }

            missingDS.setData(result);
        }
    }

    public static double[] convert1DPrimitive(Double[] row, List<Integer> missingIndices) {
        double[] result = new double[row.length];
        double sum = 0;
        int count = 0;

        Set<Integer> missingSet = new HashSet<>(missingIndices);

        for (int i = 0; i < row.length; i++) {
            if (!missingSet.contains(i)) {
                Double val = row[i];
                if (val != null) {
                    sum += val;
                    count++;
                    result[i] = val;
                }
            }
        }

        double mean = count > 0 ? sum / count : 0;

        for (int i : missingIndices) {
            result[i] = mean;
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


/*public class ListMeanImpute extends Imputer{


    public static void Impute(ListObjectDataset missingDS){
        List<Object> rawData = missingDS.getData();
        missingDS.setData(rawData);
    }

    // assumes we would want to work with ListDataset--given Double[], need double[].

    public static double[] convertToPrimitive(Object Input) {
        Double[] input = (Double[]) Input;
        int len = input.length;
        double[] result = new double[len];
        double sum = 0;
        int count = 0;

        for (int i = 0; i < len; i++) {
            Double val = input[i];
            if (val != null) {
                sum += val;
                count++;
                result[i] = val;
            }
        }

        double mean = count > 0 ? sum / count : 0;

        for (int i = 0; i < len; i++) {
            if (input[i] == null) {
                result[i] = mean;
            }
        }

        return result;
    }



    public static List<double[]> convertList(List<Object> inputList) {
        return inputList.parallelStream()
                .map(ListMeanImpute::convertToPrimitive)
                .collect(Collectors.toList());
    }



    // this is the non parallel version:
    //public static List<double[]> convertList(List<Double[]> inputList) {
    //    List<double[]> resultList = new ArrayList<>();
    //    for (Double[] row : inputList) {
    //        resultList.add(convertToPrimitive(row));
    //    }
    //    return resultList;
    //}

}*/
