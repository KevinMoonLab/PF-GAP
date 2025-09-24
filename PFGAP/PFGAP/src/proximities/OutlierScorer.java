package proximities;

import java.util.*;
import java.util.stream.*;

public class OutlierScorer {

    public static double[] getOutlierScores(
            boolean useSparse,
            boolean symmetrize,
            boolean parallel,
            Object[] ytrain,
            double[][] denseProximities,
            Map<Integer, Map<Integer, Double>> sparseProximities
    ) {
        if (useSparse) {
            Map<Integer, Map<Integer, Double>> P = symmetrize ? symmetrizeSparse(sparseProximities) : sparseProximities;
            return computeOutlierScoresSparse(P, ytrain, parallel);
        } else {
            double[][] P = symmetrize ? symmetrizeDense(denseProximities) : denseProximities;
            return computeOutlierScoresDense(P, ytrain, parallel);
        }
    }

    private static double[][] symmetrizeDense(double[][] P) {
        int n = P.length;
        double[][] sym = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sym[i][j] = 0.5 * (P[i][j] + P[j][i]);
            }
        }
        return sym;
    }

    private static Map<Integer, Map<Integer, Double>> symmetrizeSparse(Map<Integer, Map<Integer, Double>> P) {
        Map<Integer, Map<Integer, Double>> sym = new HashMap<>();
        for (Map.Entry<Integer, Map<Integer, Double>> entry : P.entrySet()) {
            int i = entry.getKey();
            for (Map.Entry<Integer, Double> inner : entry.getValue().entrySet()) {
                int j = inner.getKey();
                double pij = inner.getValue();
                double pji = P.getOrDefault(j, Collections.emptyMap()).getOrDefault(i, 0.0);
                double avg = 0.5 * (pij + pji);
                sym.computeIfAbsent(i, k -> new HashMap<>()).put(j, avg);
                sym.computeIfAbsent(j, k -> new HashMap<>()).put(i, avg);
            }
        }
        return sym;
    }

    private static double[] computeOutlierScoresDense(double[][] P, Object[] ytrain, boolean parallel) {
        int n = ytrain.length;
        Map<Object, List<Integer>> labelToIndices = groupByLabel(ytrain);

        Double[] rawScores = (parallel ? IntStream.range(0, n).parallel() : IntStream.range(0, n))
                .mapToObj(i -> {
                    List<Integer> sameClass = labelToIndices.get(ytrain[i]);
                    double sum = 0.0;
                    for (int j : sameClass) {
                        sum += P[i][j] * P[i][j];
                    }
                    if (sum == 0.0) sum = 1e-6;
                    return n / sum;
                }).toArray(Double[]::new);

        return normalizeScores(rawScores, ytrain, labelToIndices);
    }

    private static double[] computeOutlierScoresSparse(Map<Integer, Map<Integer, Double>> P, Object[] ytrain, boolean parallel) {
        int n = ytrain.length;
        Map<Object, List<Integer>> labelToIndices = groupByLabel(ytrain);

        Double[] rawScores = (parallel ? IntStream.range(0, n).parallel() : IntStream.range(0, n))
                .mapToObj(i -> {
                    List<Integer> sameClass = labelToIndices.get(ytrain[i]);
                    double sum = 0.0;
                    Map<Integer, Double> row = P.getOrDefault(i, Collections.emptyMap());
                    for (int j : sameClass) {
                        double val = row.getOrDefault(j, 0.0);
                        sum += val * val;
                    }
                    if (sum == 0.0) sum = 1e-6;
                    return n / sum;
                }).toArray(Double[]::new);

        return normalizeScores(rawScores, ytrain, labelToIndices);
    }

    private static double[] normalizeScores(Double[] scores, Object[] ytrain, Map<Object, List<Integer>> labelToIndices) {
        double[] normalized = new double[scores.length];
        Map<Object, Double> medians = new HashMap<>();
        Map<Object, Double> mads = new HashMap<>();

        for (Map.Entry<Object, List<Integer>> entry : labelToIndices.entrySet()) {
            Object label = entry.getKey();
            List<Double> labelScores = entry.getValue().stream().map(i -> scores[i]).collect(Collectors.toList());

            double median = computeMedian(labelScores);
            double mad = labelScores.stream().mapToDouble(s -> Math.abs(s - median)).average().orElse(1e-6);

            medians.put(label, median);
            mads.put(label, mad);
        }

        for (int i = 0; i < scores.length; i++) {
            double median = medians.get(ytrain[i]);
            double mad = mads.get(ytrain[i]);
            normalized[i] = Math.abs(scores[i] - median) / mad;
        }

        return normalized;
    }

    private static double computeMedian(List<Double> values) {
        Collections.sort(values);
        int n = values.size();
        if (n % 2 == 0) {
            return 0.5 * (values.get(n / 2 - 1) + values.get(n / 2));
        } else {
            return values.get(n / 2);
        }
    }

    private static Map<Object, List<Integer>> groupByLabel(Object[] ytrain) {
        Map<Object, List<Integer>> labelToIndices = new HashMap<>();
        for (int i = 0; i < ytrain.length; i++) {
            labelToIndices.computeIfAbsent(ytrain[i], k -> new ArrayList<>()).add(i);
        }
        return labelToIndices;
    }
}
