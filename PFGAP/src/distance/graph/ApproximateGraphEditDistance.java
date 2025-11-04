package distance.graph;

import java.io.Serializable;
import java.util.*;

import distance.api.DistanceFunction;

public class ApproximateGraphEditDistance implements DistanceFunction, Serializable {
    private static final double NODE_INSERTION_DELETION_COST = 1.0;
    private static final double EDGE_INSERTION_DELETION_COST = 1.0;
    private static final double EDGE_SUBSTITUTION_COST_WEIGHTED = 1.0;

    @Override
    public double compute(Object a, Object b) {
        Object[][] graphA = (Object[][]) a;
        Object[][] graphB = (Object[][]) b;

        int N1 = graphA.length;
        int N2 = graphB.length;

        double[][] adjA = extractAdjacency(graphA);
        double[][] adjB = extractAdjacency(graphB);

        Object[][] featuresA = extractObjectNodeFeatures(graphA);
        Object[][] featuresB = extractObjectNodeFeatures(graphB);

        return approximateGED(adjA, adjB, featuresA, featuresB);
    }

    private double approximateGED(
        double[][] adjA,
        double[][] adjB,
        Object[][] featuresA,
        Object[][] featuresB
    ) {
        int N1 = adjA.length;
        int N2 = adjB.length;

        boolean[] matchedB = new boolean[N2];
        Map<Integer, Integer> mapping = new HashMap<>();

        // Step 1: Greedy node matching
        for (int i = 0; i < N1; i++) {
            double minCost = Double.MAX_VALUE;
            int bestMatch = -1;
            for (int j = 0; j < N2; j++) {
                if (matchedB[j]) continue;
                double cost = nodeSubstitutionCost(adjA, adjB, featuresA, featuresB, i, j);
                if (cost < minCost) {
                    minCost = cost;
                    bestMatch = j;
                }
            }
            if (bestMatch != -1) {
                mapping.put(i, bestMatch);
                matchedB[bestMatch] = true;
            }
        }

        // Step 2: Compute edge cost for matched nodes
        double edgeCost = 0.0;
        for (Map.Entry<Integer, Integer> entry1 : mapping.entrySet()) {
            int i1 = entry1.getKey();
            int j1 = entry1.getValue();
            for (Map.Entry<Integer, Integer> entry2 : mapping.entrySet()) {
                int i2 = entry2.getKey();
                int j2 = entry2.getValue();
                if (i1 == i2) continue;

                boolean edgeA = adjA[i1][i2] != 0;
                boolean edgeB = adjB[j1][j2] != 0;

                if (edgeA && edgeB) {
                    edgeCost += EDGE_SUBSTITUTION_COST_WEIGHTED *
                                Math.abs(adjA[i1][i2] - adjB[j1][j2]);
                } else if (edgeA || edgeB) {
                    edgeCost += EDGE_INSERTION_DELETION_COST;
                }
            }
        }

        // Step 3: Account for unmatched nodes
        int unmatchedA = N1 - mapping.size();
        int unmatchedB = N2 - mapping.size();
        double nodeEditCost = (unmatchedA + unmatchedB) * NODE_INSERTION_DELETION_COST;

        return edgeCost + nodeEditCost;
    }

    private double nodeSubstitutionCost(
        double[][] g1, double[][] g2,
        Object[][] f1, Object[][] f2,
        int i, int j
    ) {
        if (f1 != null && f2 != null) {
            int len = Math.min(f1[i].length, f2[j].length);
            double cost = 0.0;

            for (int k = 0; k < len; k++) {
                Object v1 = f1[i][k];
                Object v2 = f2[j][k];

                if (v1 instanceof Number && v2 instanceof Number) {
                    double d1 = ((Number) v1).doubleValue();
                    double d2 = ((Number) v2).doubleValue();
                    cost += Math.pow(d1 - d2, 2);
                } else {
                    if (!Objects.equals(v1, v2)) {
                        cost += 1.0;
                    }
                }
            }

            return Math.sqrt(cost);
        }

        // fallback: structural (degree difference)
        double degree1 = 0.0, degree2 = 0.0;
        for (double v : g1[i]) degree1 += v;
        for (double v : g2[j]) degree2 += v;
        return Math.abs(degree1 - degree2);
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
}

