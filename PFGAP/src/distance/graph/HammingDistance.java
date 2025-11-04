package distance.graph;

import distance.api.DistanceFunction;

import java.io.Serializable;

public class HammingDistance implements DistanceFunction, Serializable {

    @Override
    public double compute(Object a, Object b) {
        Object[][] graphA = (Object[][]) a;
        Object[][] graphB = (Object[][]) b;

        double[][] adjA = extractAdjacency(graphA);
        double[][] adjB = extractAdjacency(graphB);

        if (adjA.length != adjB.length) {
            throw new IllegalArgumentException("Graphs must be of the same size for Hamming distance.");
        }

        return computeHammingDistance(adjA, adjB);
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

    private double computeHammingDistance(double[][] adjA, double[][] adjB) {
        int N = adjA.length;
        double distance = 0.0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                distance += Math.abs(adjA[i][j] - adjB[i][j]);
            }
        }

        return distance;
    }
}

