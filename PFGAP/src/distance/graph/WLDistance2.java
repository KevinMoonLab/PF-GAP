package distance.graph;

import java.io.Serializable;
import java.util.*;
import distance.api.DistanceFunction;

public class WLDistance2 implements DistanceFunction, Serializable {

    private static final int WL_ITERATIONS = 3;
    private static final int BIN_COUNT = 10;
    private static final boolean USE_HASHING = false;

    @Override
    public double compute(Object a, Object b) {
        // Directly cast inputs to double[][] arrays
        double[][] graphA = (double[][]) a;
        double[][] graphB = (double[][]) b;

        double[][] adjA = extractAdjacency(graphA);
        double[][] adjB = extractAdjacency(graphB);

        double[][] featuresA = extractNodeFeatures(graphA);
        double[][] featuresB = extractNodeFeatures(graphB);

        Map<String, Double> histA = computeWLHistogram(adjA, featuresA);
        Map<String, Double> histB = computeWLHistogram(adjB, featuresB);

        return cosineDistance(histA, histB);
    }

    /**
     * Extract adjacency matrix from the top-left N×N block.
     */
    private double[][] extractAdjacency(double[][] graph) {
        int N = graph.length;
        double[][] adj = new double[N][N];
        for (int i = 0; i < N; i++) {
            System.arraycopy(graph[i], 0, adj[i], 0, N);
        }
        return adj;
    }

    /**
     * Extract node features from the right-side N×F block.
     */
    private double[][] extractNodeFeatures(double[][] graph) {
        int N = graph.length;
        int totalCols = graph[0].length;
        int F = totalCols - N;
        if (F <= 0) return null;

        double[][] features = new double[N][F];
        for (int i = 0; i < N; i++) {
            System.arraycopy(graph[i], N, features[i], 0, F);
        }
        return features;
    }

    /**
     * Compute Weisfeiler–Lehman histogram for numeric features.
     */
    private Map<String, Double> computeWLHistogram(double[][] adj, double[][] features) {
        int N = adj.length;
        String[] labels = new String[N];

        // Initialize node labels from features
        for (int i = 0; i < N; i++) {
            if (features != null) {
                StringBuilder sb = new StringBuilder();
                for (double val : features[i]) {
                    // Clamp features to [0, 1]
                    if (val > 1.0) val = 1.0;
                    if (val < 0.0) val = 0.0;

                    // For binary or continuous features, create compact bin label
                    int bin = (val == 0.0 || val == 1.0)
                            ? (int) val
                            : (int) Math.floor(val * (BIN_COUNT - 1));

                    sb.append("num").append(bin).append("_");
                }
                labels[i] = sb.toString();
            } else {
                labels[i] = "init";
            }
        }

        Map<String, Double> hist = new HashMap<>();

        // Count initial labels
        for (String label : labels) {
            hist.put(label, hist.getOrDefault(label, 0.0) + 1);
        }

        // WL iterations
        for (int it = 0; it < WL_ITERATIONS; it++) {
            String[] newLabels = new String[N];

            for (int i = 0; i < N; i++) {
                List<String> multiset = new ArrayList<>();
                for (int j = 0; j < N; j++) {
                    if (adj[i][j] > 0.0) { // treat nonzero edges as neighbors
                        multiset.add(labels[j]);
                    }
                }
                Collections.sort(multiset);
                String combined = labels[i] + "_" + String.join("_", multiset);
                String finalLabel = USE_HASHING
                        ? Integer.toHexString(combined.hashCode())
                        : combined;

                newLabels[i] = finalLabel;
                hist.put(finalLabel, hist.getOrDefault(finalLabel, 0.0) + 1);
            }

            labels = newLabels;
        }

        // Normalize histogram (L1)
        double total = hist.values().stream().mapToDouble(Double::doubleValue).sum();
        if (total > 0) {
            for (Map.Entry<String, Double> e : hist.entrySet()) {
                hist.put(e.getKey(), e.getValue() / total);
            }
        }

        return hist;
    }

    /**
     * Compute cosine distance (1 - cosine similarity).
     */
    private double cosineDistance(Map<String, Double> h1, Map<String, Double> h2) {
        Set<String> keys = new HashSet<>(h1.keySet());
        keys.addAll(h2.keySet());

        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;

        for (String key : keys) {
            double v1 = h1.getOrDefault(key, 0.0);
            double v2 = h2.getOrDefault(key, 0.0);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        if (norm1 == 0 || norm2 == 0) return 1.0;
        double sim = dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
        return 1.0 - sim;
    }
}




//previous implementation
/*package distance.graph;

import java.io.Serializable;
import java.util.*;

import distance.api.DistanceFunction;

public class WLDistance2 implements DistanceFunction, Serializable {
    private static final int WL_ITERATIONS = 3;
    private static final int BIN_COUNT = 10;
    private static final boolean USE_HASHING = false;

    @Override
    public double compute(Object a, Object b) {
        Object[][] graphA = (Object[][]) a;
        Object[][] graphB = (Object[][]) b;

        double[][] adjA = extractAdjacency(graphA);
        double[][] adjB = extractAdjacency(graphB);

        Object[][] featuresA = extractObjectNodeFeatures(graphA);
        Object[][] featuresB = extractObjectNodeFeatures(graphB);

        Map<String, Double> histogramA = computeWLHistogram(adjA, featuresA);
        Map<String, Double> histogramB = computeWLHistogram(adjB, featuresB);

        return cosineDistance(histogramA, histogramB);
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

    private Map<String, Double> computeWLHistogram(double[][] adj, Object[][] features) {
        int N = adj.length;
        String[] labels = new String[N];

        // Step 1: Initialize labels from features
        for (int i = 0; i < N; i++) {
            if (features != null) {
                StringBuilder sb = new StringBuilder();
                for (Object f : features[i]) {
                    if (f instanceof Number) {
                        double val = ((Number) f).doubleValue();
                        //int bin = (int) Math.floor(val * BIN_COUNT); // assuming features are normalized 0-1
                        //bin = Math.max(0, Math.min(BIN_COUNT - 1, bin));
                        //sb.append("num").append(bin).append("_");
                        if (val > 1.0) val = 1.0;
                        if (val < 0.0) val = 0.0;

                        // For binary features, just cast directly
                        int bin = (val == 0.0 || val == 1.0) ? (int) val : (int) Math.floor(val * BIN_COUNT);
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

        Map<String, Double> histogram = new HashMap<>();

        // Count initial labels
        for (String label : labels) {
            histogram.put(label, histogram.getOrDefault(label, 0.0) + 1);
        }

        // Step 2: WL iterations
        for (int it = 0; it < WL_ITERATIONS; it++) {
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

                // Optional: Hash to shorten labels
                String finalLabel = USE_HASHING ? Integer.toHexString(combined.hashCode()) : combined;

                newLabels[i] = finalLabel;
                histogram.put(finalLabel, histogram.getOrDefault(finalLabel, 0.0) + 1);
            }

            labels = newLabels;
        }

        // Normalize histogram to unit L1 norm
        double total = histogram.values().stream().mapToDouble(Double::doubleValue).sum();
        if (total > 0) {
            for (Map.Entry<String, Double> entry : histogram.entrySet()) {
                histogram.put(entry.getKey(), entry.getValue() / total);
            }
        }

        return histogram;
    }

    private double cosineDistance(Map<String, Double> h1, Map<String, Double> h2) {
        Set<String> keys = new HashSet<>(h1.keySet());
        keys.addAll(h2.keySet());

        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;

        for (String key : keys) {
            double v1 = h1.getOrDefault(key, 0.0);
            double v2 = h2.getOrDefault(key, 0.0);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        if (norm1 == 0 || norm2 == 0) return 1.0; // Max distance if one is empty
        double cosineSimilarity = dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
        return 1.0 - cosineSimilarity; // Convert similarity to distance
    }
}*/
