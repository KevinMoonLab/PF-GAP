package imputation;

import datasets.ListObjectDataset;
import distance.DistanceRegistry;
import distance.MEASURE;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KNNImputer extends Imputer {

    private final MEASURE[] measures;
    private final int k;

    public KNNImputer(MEASURE[] measures, int k) {
        this.measures = measures;
        this.k = k;
    }

    @Override
    public void Impute(ListObjectDataset missingDS) {
        List<Object> rawData = missingDS.getData();
        MissingIndices mi = missingDS.getMissingIndices();

        List<MaskedDistance> distanceFunctions = Arrays.stream(measures)
                .map(DistanceRegistry::getMaskedDistance)
                .collect(Collectors.toList());

        List<Object> result;

        if (mi.is2D()) {
            result = impute2D(rawData, mi.indices2D, distanceFunctions);
        } else {
            Object first = rawData.get(0);
            if (first instanceof Double[]) {
                result = impute1DNumeric(rawData, mi.indices1D, distanceFunctions);
            } else {
                result = impute1DCategorical(rawData, mi.indices1D, distanceFunctions);
            }
        }

        missingDS.setData(result);  // Replaces the entire data list
    }


    private List<Object> impute1DNumeric(List<Object> data, List<List<Integer>> missingIndices, List<MaskedDistance> distances) {
        return IntStream.range(0, data.size())
                .parallel()
                .mapToObj(i -> {
                    Double[] target = (Double[]) data.get(i);
                    List<Integer> missing = missingIndices.get(i);

                    // Always convert to double[] form
                    double[] imputed = new double[target.length];

                    if (missing.isEmpty()) {
                        // Just unbox everything
                        for (int j = 0; j < target.length; j++) {
                            // Maybe check for null, decide what to do; here we assume no nulls
                            imputed[j] = target[j];
                        }
                        return (Object) imputed;
                    }

                    // If there *are* missing entries:
                    List<Neighbor> neighbors = findKNearest(target, data, i, distances);

                    for (int j = 0; j < target.length; j++) {
                        if (target[j] != null) {
                            imputed[j] = target[j];
                        } else {
                            int finalJ = j;
                            List<Double> values = neighbors.stream()
                                    .map(n -> ((Double[]) data.get(n.index))[finalJ])
                                    .filter(Objects::nonNull)
                                    .collect(Collectors.toList());
                            imputed[j] = values.isEmpty()
                                    ? 0.0
                                    : values.stream().mapToDouble(d -> d).average().orElse(0.0);
                        }
                    }
                    return (Object) imputed;
                })
                .collect(Collectors.toList());
    }



    private List<Object> impute1DCategorical(List<Object> data, List<List<Integer>> missingIndices, List<MaskedDistance> distances) {
        return IntStream.range(0, data.size())
                .parallel()
                .mapToObj(i -> {
                    Object[] target = (Object[]) data.get(i);
                    List<Integer> missing = missingIndices.get(i);

                    if (missing.isEmpty()) return target;

                    List<Neighbor> neighbors = findKNearest(target, data, i, distances);

                    Object[] imputed = Arrays.copyOf(target, target.length);
                    for (int j : missing) {
                        Map<Object, Integer> freq = new HashMap<>();
                        for (Neighbor n : neighbors) {
                            Object val = ((Object[]) data.get(n.index))[j];
                            if (val != null) {
                                freq.put(val, freq.getOrDefault(val, 0) + 1);
                            }
                        }
                        Object mode = freq.entrySet().stream()
                                .max(Map.Entry.comparingByValue())
                                .map(Map.Entry::getKey)
                                .orElse(null);
                        imputed[j] = mode;
                    }

                    return (Object) imputed;
                })
                .collect(Collectors.toList());
    }


    private List<Object> impute2D(List<Object> data, List<List<List<Integer>>> missingIndices, List<MaskedDistance> distances) {
        return IntStream.range(0, data.size())
                .parallel()
                .mapToObj(i -> {
                    Double[][] target = (Double[][]) data.get(i);
                    List<List<Integer>> missing = missingIndices.get(i);

                    // Always produce a primitive double[][]
                    double[][] imputed = new double[target.length][target[0].length];

                    if (missing.stream().allMatch(List::isEmpty)) {
                        for (int r = 0; r < target.length; r++) {
                            for (int c = 0; c < target[r].length; c++) {
                                imputed[r][c] = target[r][c];
                            }
                        }
                        return (Object) imputed;
                    }

                    List<Neighbor> neighbors = findKNearest(target, data, i, distances);

                    for (int r = 0; r < target.length; r++) {
                        for (int c = 0; c < target[r].length; c++) {
                            if (target[r][c] != null) {
                                imputed[r][c] = target[r][c];
                            } else {
                                int finalR = r;
                                int finalC = c;
                                List<Double> values = neighbors.stream()
                                        .map(n -> ((Double[][]) data.get(n.index))[finalR][finalC])
                                        .filter(Objects::nonNull)
                                        .collect(Collectors.toList());
                                imputed[r][c] = values.isEmpty()
                                        ? 0.0
                                        : values.stream().mapToDouble(d -> d).average().orElse(0.0);
                            }
                        }
                    }
                    return (Object) imputed;
                })
                .collect(Collectors.toList());
    }




    private List<Neighbor> findKNearest(Object target, List<Object> data, int targetIndex, List<MaskedDistance> distances) {
        List<Neighbor> neighbors = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            if (i == targetIndex) continue;

            double totalDist = 0;
            for (MaskedDistance dist : distances) {
                double d = dist.compute(target, data.get(i));
                if (Double.isInfinite(d)) {
                    totalDist = Double.POSITIVE_INFINITY;
                    break;
                }
                totalDist += d;
            }

            if (!Double.isInfinite(totalDist)) {
                neighbors.add(new Neighbor(i, totalDist));
            }
        }

        return neighbors.stream()
                .sorted(Comparator.comparingDouble(n -> n.distance))
                .limit(k)
                .collect(Collectors.toList());
    }

    private static class Neighbor {
        int index;
        double distance;

        Neighbor(int index, double distance) {
            this.index = index;
            this.distance = distance;
        }
    }


}