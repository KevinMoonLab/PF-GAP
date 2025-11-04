package distance.multiTS;

import util.Pair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class DTW_D implements Serializable {

    public DTW_D() {}

    public synchronized double distance(Object Series1, Object Series2, double bsf, int windowSize) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        int dims1 = series1.length;
        int dims2 = series2.length;

        if (dims1 != dims2) {
            throw new IllegalArgumentException("Both series must have the same number of dimensions (rows).");
        }

        int len1 = series1[0].length;
        int len2 = series2[0].length;

        // Validate all rows have consistent time lengths
        for (int d = 0; d < dims1; d++) {
            if (series1[d].length != len1 || series2[d].length != len2) {
                throw new IllegalArgumentException("All dimensions must have consistent time lengths.");
            }
        }

        if (windowSize == -1) {
            windowSize = Math.max(len1, len2);
        }

        double[][] cost = new double[len1][len2];

        for (int i = 0; i < len1; i++) {
            int jStart = Math.max(0, i - windowSize);
            int jStop = Math.min(len2 - 1, i + windowSize);

            for (int j = jStart; j <= jStop; j++) {
                double dist = squaredDistanceAt(series1, series2, i, j);

                if (i == 0 && j == 0) {
                    cost[i][j] = dist;
                } else {
                    double minPrev = Double.POSITIVE_INFINITY;
                    if (i > 0 && j > 0) minPrev = Math.min(minPrev, cost[i - 1][j - 1]);
                    if (i > 0) minPrev = Math.min(minPrev, cost[i - 1][j]);
                    if (j > 0) minPrev = Math.min(minPrev, cost[i][j - 1]);

                    cost[i][j] = dist + minPrev;
                }

                if (cost[i][j] > bsf * bsf) return Double.POSITIVE_INFINITY;
            }
        }

        double finalDist = Math.sqrt(cost[len1 - 1][len2 - 1]);
        return finalDist > bsf ? Double.POSITIVE_INFINITY : finalDist;
    }

    private double squaredDistanceAt(double[][] s1, double[][] s2, int t1, int t2) {
        double sum = 0.0;
        for (int d = 0; d < s1.length; d++) {
            double diff = s1[d][t1] - s2[d][t2];
            sum += diff * diff;
        }
        return sum;
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(double[][] series1, double[][] series2, int windowSize) {
        int len1 = series1[0].length;
        int len2 = series2[0].length;

        if (windowSize == -1) {
            windowSize = Math.max(len1, len2);
        }

        double[][] cost = new double[len1][len2];
        int[][][] backtrack = new int[len1][len2][2];

        for (int i = 0; i < len1; i++) {
            int jStart = Math.max(0, i - windowSize);
            int jStop = Math.min(len2 - 1, i + windowSize);

            for (int j = jStart; j <= jStop; j++) {
                double dist = squaredDistanceAt(series1, series2, i, j);

                if (i == 0 && j == 0) {
                    cost[i][j] = dist;
                    backtrack[i][j][0] = -1;
                    backtrack[i][j][1] = -1;
                } else {
                    double minPrev = Double.POSITIVE_INFINITY;
                    int pi = -1, pj = -1;

                    if (i > 0 && j > 0 && cost[i - 1][j - 1] < minPrev) {
                        minPrev = cost[i - 1][j - 1];
                        pi = i - 1; pj = j - 1;
                    }
                    if (i > 0 && cost[i - 1][j] < minPrev) {
                        minPrev = cost[i - 1][j];
                        pi = i - 1; pj = j;
                    }
                    if (j > 0 && cost[i][j - 1] < minPrev) {
                        minPrev = cost[i][j - 1];
                        pi = i; pj = j - 1;
                    }

                    cost[i][j] = dist + minPrev;
                    backtrack[i][j][0] = pi;
                    backtrack[i][j][1] = pj;
                }
            }
        }

        // Backtrack to extract path
        List<Pair<Integer, Integer>> path = new ArrayList<>();
        int i = len1 - 1, j = len2 - 1;
        while (i >= 0 && j >= 0) {
            path.add(0, new Pair<>(i, j));
            int ni = backtrack[i][j][0];
            int nj = backtrack[i][j][1];
            if (ni == -1 || nj == -1) break;
            i = ni;
            j = nj;
        }

        return path;
    }

}
