package distance.graph;

import java.io.Serializable;
import java.util.Arrays;

import distance.api.DistanceFunction;

public class GraphEditDistance implements DistanceFunction, Serializable {
    private static final double NODE_INSERTION_DELETION_COST = 1.0;
    private static final double EDGE_INSERTION_DELETION_COST = 1.0;
    private static final double EDGE_SUBSTITUTION_COST_WEIGHTED = 1.0;

    /**
     * Compute the graph edit distance between two graphs represented as adjacency matrices.
     * @param a adjacency matrix of graph 1 (double[][])
     * @param b adjacency matrix of graph 2 (double[][])
     * @return graph edit distance
     */
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

        return graphEditDistance(adjA, adjB, featuresA, featuresB);
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




    /**
     * Compute node substitution cost.
     * Simple example: 0 if nodes are considered identical, else 1.
     * Extend with node labels or attributes if available.
     */
    private double nodeSubstitutionCost(
        double[][] g1,
        double[][] g2,
        Object[][] features1,
        Object[][] features2,
        int node1,
        int node2
    ) {
        if (features1 != null && features2 != null) {
            int len = Math.min(features1[node1].length, features2[node2].length);
            double cost = 0.0;

            for (int i = 0; i < len; i++) {
                Object f1 = features1[node1][i];
                Object f2 = features2[node2][i];

                if (f1 instanceof Number && f2 instanceof Number) {
                    double v1 = ((Number) f1).doubleValue();
                    double v2 = ((Number) f2).doubleValue();
                    cost += Math.pow(v1 - v2, 2);
                } else {
                    // Categorical comparison: 1 if unequal, 0 if equal
                    if (f1 == null || f2 == null || !f1.equals(f2)) {
                        cost += 1.0;
                    }
                }
            }

            return Math.sqrt(cost);
        } else {
            // Fallback to structural proxy: degree difference
            double degree1 = 0.0, degree2 = 0.0;
            for (int i = 0; i < g1.length; i++) degree1 += g1[node1][i];
            for (int i = 0; i < g2.length; i++) degree2 += g2[node2][i];
            return Math.abs(degree1 - degree2);
        }
    }



    /**
     * Compute the graph edit distance.
     */
    public double graphEditDistance(
        double[][] g1,
        double[][] g2,
        Object[][] features1,
        Object[][] features2
    ) {
        // Hungarian cost matrix construction
        int n1 = g1.length;
        int n2 = g2.length;
        double[][] costMatrix = new double[n1][n2];

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                costMatrix[i][j] = nodeSubstitutionCost(g1, g2, features1, features2, i, j);
            }
        }

        int[] nodeMapping = HungarianAlgorithm.hgAlgorithm(costMatrix, "min");
        double edgeCost = computeEdgeEditCost(g1, g2, nodeMapping);
        double nodeEditCost = Math.abs(n1 - n2) * NODE_INSERTION_DELETION_COST;

        return edgeCost + nodeEditCost;
    }


    /**
     * Count unmatched nodes on both sides given the node mapping.
     */
    private int countUnmatchedNodes(int[] nodeMapping, int n1, int n2) {
        int unmatched = 0;

        // Count unmatched nodes in g1 (mapped to -1)
        for (int i = 0; i < n1; i++) {
            if (nodeMapping[i] == -1) unmatched++;
        }

        // Count unmatched nodes in g2 (not mapped from g1)
        boolean[] matchedInG2 = new boolean[n2];
        for (int j : nodeMapping) {
            if (j != -1) matchedInG2[j] = true;
        }
        for (int j = 0; j < n2; j++) {
            if (!matchedInG2[j]) unmatched++;
        }

        return unmatched;
    }

    /**
     * Compute edge edit cost between two graphs given node mapping.
     */
    private double computeEdgeEditCost(double[][] g1, double[][] g2, int[] nodeMapping) {
        int n1 = g1.length;
        int n2 = g2.length;
        double cost = 0.0;

        // Build reverse mapping: node in g2 -> node in g1
        int[] reverseMapping = new int[n2];
        Arrays.fill(reverseMapping, -1);
        for (int i = 0; i < nodeMapping.length; i++) {
            int j = nodeMapping[i];
            if (j != -1) reverseMapping[j] = i;
        }

        // Iterate over matched nodes to compare edges
        for (int i = 0; i < n1; i++) {
            int j = nodeMapping[i];
            if (j == -1) continue;

            for (int k = 0; k < n1; k++) {
                int l = nodeMapping[k];
                if (l == -1) continue;

                boolean edgeG1 = g1[i][k] != 0;
                boolean edgeG2 = g2[j][l] != 0;

                if (edgeG1 && edgeG2) {
                    // Edge exists in both: substitution cost based on weight difference
                    cost += EDGE_SUBSTITUTION_COST_WEIGHTED * Math.abs(g1[i][k] - g2[j][l]);
                } else if (edgeG1 && !edgeG2) {
                    // Edge deleted
                    cost += EDGE_INSERTION_DELETION_COST;
                } else if (!edgeG1 && edgeG2) {
                    // Edge inserted
                    cost += EDGE_INSERTION_DELETION_COST;
                }
            }
        }

        // Optionally: Count edges connected to unmatched nodes as insertions/deletions
        // This is a refinement you may add based on application.

        return cost;
    }
}

