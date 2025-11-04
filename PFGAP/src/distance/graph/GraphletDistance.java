package distance.graph;

import distance.api.DistanceFunction;

import java.io.Serializable;
import java.util.*;

public class GraphletDistance implements DistanceFunction, Serializable {
    private static final int NUM_SAMPLES = 500; // tune for speed/accuracy trade-off
    private static final int MAX_ATTEMPTS = 5000;

    @Override
    public double compute(Object a, Object b) {
        Object[][] graphA = (Object[][]) a;
        Object[][] graphB = (Object[][]) b;

        double[][] adjA = extractAdjacency(graphA);
        double[][] adjB = extractAdjacency(graphB);

        Map<String, Integer> histA = sampleGraphletHistogram(adjA, NUM_SAMPLES);
        Map<String, Integer> histB = sampleGraphletHistogram(adjB, NUM_SAMPLES);

        return histogramDistance(histA, histB);
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

    private Map<String, Integer> sampleGraphletHistogram(double[][] adj, int numSamples) {
        int N = adj.length;
        Map<String, Integer> histogram = new HashMap<>();
        Random rand = new Random();

        int samplesCollected = 0;
        int attempts = 0;

        while (samplesCollected < numSamples && attempts < MAX_ATTEMPTS) {
            int i = rand.nextInt(N);
            int j = rand.nextInt(N);
            int k = rand.nextInt(N);

            if (i == j || i == k || j == k) {
                attempts++;
                continue;
            }

            int[] nodes = {i, j, k};
            Arrays.sort(nodes);
            i = nodes[0]; j = nodes[1]; k = nodes[2];

            // Count edges among i, j, k
            int edges = 0;
            if (adj[i][j] != 0) edges++;
            if (adj[i][k] != 0) edges++;
            if (adj[j][k] != 0) edges++;

            String type;
            switch (edges) {
                case 3:
                    type = "triangle";
                    break;
                case 2:
                    type = "wedge";
                    break;
                case 1:
                    type = "edge";
                    break;
                default:
                    attempts++;
                    continue; // skip null graphlets
            }

            histogram.put(type, histogram.getOrDefault(type, 0) + 1);
            samplesCollected++;
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

        return Math.sqrt(sum);
    }
}

