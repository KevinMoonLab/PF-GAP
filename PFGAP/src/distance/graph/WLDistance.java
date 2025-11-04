package distance.graph;

import java.io.Serializable;
import java.util.*;

import distance.api.DistanceFunction;

public class WLDistance implements DistanceFunction, Serializable {
    private static final int WL_ITERATIONS = 3;
    private static final int BIN_COUNT = 10; // For numeric feature binning

    @Override
    public double compute(Object a, Object b) {
        Object[][] graphA = (Object[][]) a;
        Object[][] graphB = (Object[][]) b;

        int N1 = graphA.length, N2 = graphB.length;

        double[][] adjA = extractAdjacency(graphA);
        double[][] adjB = extractAdjacency(graphB);

        Object[][] featuresA = extractObjectNodeFeatures(graphA);
        Object[][] featuresB = extractObjectNodeFeatures(graphB);

        Map<String, Integer> histogramA = computeWLHistogram(adjA, featuresA);
        Map<String, Integer> histogramB = computeWLHistogram(adjB, featuresB);

        return histogramDistance(histogramA, histogramB);
    }

    private double[][] extractAdjacency(Object[][] graph) {
        int N = graph.length;
        double[][] adj = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                adj[i][j] = ((Number) graph[i][j]).doubleValue();
            }
        }
        return adj;
    }

    private Object[][] extractObjectNodeFeatures(Object[][] graph) {
        int N = graph.length;
        int totalCols = graph[0].length;
        int F = totalCols - N;
        if (F <= 0) return null;

        Object[][] features = new Object[N][F];
        for (int i = 0; i < N; i++) {
            System.arraycopy(graph[i], N, features[i], 0, F);
        }
        return features;
    }

    private Map<String, Integer> computeWLHistogram(double[][] adj, Object[][] features) {
        int N = adj.length;
        String[] labels = new String[N];

        // Step 1: Initialize node labels
        for (int i = 0; i < N; i++) {
            if (features != null) {
                StringBuilder sb = new StringBuilder();
                for (Object f : features[i]) {
                    if (f instanceof Number) {
                        int bin = ((Number) f).intValue() % BIN_COUNT;
                        sb.append("num").append(bin).append("_");
                    } else if (f != null) {
                        sb.append(f.toString()).append("_");
                    }
                }
                labels[i] = sb.toString();
            } else {
                labels[i] = "init";
            }
        }

        Map<String, Integer> histogram = new HashMap<>();

        // Step 2: Perform WL iterations
        for (int it = 0; it < WL_ITERATIONS; it++) {
            Map<String, String> compressionMap = new HashMap<>();
            int counter = 0;
            String[] newLabels = new String[N];

            for (int i = 0; i < N; i++) {
                List<String> multiset = new ArrayList<>();
                for (int j = 0; j < N; j++) {
                    if (adj[i][j] != 0) {
                        multiset.add(labels[j]);
                    }
                }
                Collections.sort(multiset);
                String combined = labels[i] + "_" + String.join("_", multiset);

                // Compression step: relabel to compact strings
                String compressed;
                if (compressionMap.containsKey(combined)) {
                    compressed = compressionMap.get(combined);
                } else {
                    compressed = "h" + counter++;
                    compressionMap.put(combined, compressed);
                }

                newLabels[i] = compressed;
                histogram.put(compressed, histogram.getOrDefault(compressed, 0) + 1);
            }

            labels = newLabels;
        }

        return histogram;
    }

    private double histogramDistance(Map<String, Integer> h1, Map<String, Integer> h2) {
        Set<String> keys = new HashSet<>(h1.keySet());
        keys.addAll(h2.keySet());

        double sum = 0.0;
        for (String key : keys) {
            int v1 = h1.getOrDefault(key, 0);
            int v2 = h2.getOrDefault(key, 0);
            sum += Math.pow(v1 - v2, 2);
        }

        return Math.sqrt(sum); // L2 distance
    }
}

