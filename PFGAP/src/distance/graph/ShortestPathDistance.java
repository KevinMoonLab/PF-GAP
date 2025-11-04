package distance.graph;

import java.io.Serializable;
import java.util.*;

import distance.api.DistanceFunction;

public class ShortestPathDistance implements DistanceFunction, Serializable {
    private static final double INF = Double.POSITIVE_INFINITY;

    @Override
    public double compute(Object a, Object b) {
        Object[][] graphA = (Object[][]) a;
        Object[][] graphB = (Object[][]) b;

        double[][] adjA = extractAdjacency(graphA);
        double[][] adjB = extractAdjacency(graphB);

        double[][] spA = floydWarshall(adjA);
        double[][] spB = floydWarshall(adjB);

        Map<Double, Integer> histA = buildPathLengthHistogram(spA);
        Map<Double, Integer> histB = buildPathLengthHistogram(spB);

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

    private double[][] floydWarshall(double[][] graph) {
        int N = graph.length;
        double[][] dist = new double[N][N];

        for (int i = 0; i < N; i++) {
            Arrays.fill(dist[i], INF);
            dist[i][i] = 0;
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (graph[i][j] != 0) {
                    dist[i][j] = graph[i][j];
                }
            }
        }

        for (int k = 0; k < N; k++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        return dist;
    }

    private Map<Double, Integer> buildPathLengthHistogram(double[][] shortestPaths) {
        Map<Double, Integer> histogram = new HashMap<>();
        int N = shortestPaths.length;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double d = shortestPaths[i][j];
                if (i != j && d < INF) {
                    double rounded = Math.round(d * 100.0) / 100.0; // Round for stability
                    histogram.put(rounded, histogram.getOrDefault(rounded, 0) + 1);
                }
            }
        }

        return histogram;
    }

    private double histogramDistance(Map<Double, Integer> h1, Map<Double, Integer> h2) {
        Set<Double> keys = new HashSet<>(h1.keySet());
        keys.addAll(h2.keySet());

        double sum = 0.0;
        for (double key : keys) {
            int v1 = h1.getOrDefault(key, 0);
            int v2 = h2.getOrDefault(key, 0);
            sum += Math.pow(v1 - v2, 2);
        }

        return Math.sqrt(sum); // L2 distance between histograms
    }
}

