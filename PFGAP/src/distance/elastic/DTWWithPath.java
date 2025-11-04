package distance.elastic;

import util.Pair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class DTWWithPath implements Serializable {

    public DTWWithPath() {}

    public double distance(double[] series1, double[] series2, int windowSize) {
        return computeDTW(series1, series2, windowSize, false, null);
    }

    public List<Pair<Integer, Integer>> getAlignmentPath(double[] series1, double[] series2, int windowSize) {
        List<Pair<Integer, Integer>> path = new ArrayList<>();
        computeDTW(series1, series2, windowSize, true, path);
        return path;
    }

    private double computeDTW(double[] s1, double[] s2, int windowSize, boolean trackPath, List<Pair<Integer, Integer>> pathOut) {
        int n = s1.length;
        int m = s2.length;
        if (windowSize == -1) windowSize = Math.max(n, m);

        double[][] cost = new double[n][m];
        int[][][] backtrack = trackPath ? new int[n][m][2] : null;

        for (int i = 0; i < n; i++) {
            int jStart = Math.max(0, i - windowSize);
            int jEnd = Math.min(m, i + windowSize + 1);
            for (int j = jStart; j < jEnd; j++) {
                double dist = squaredDistance(s1[i], s2[j]);
                if (i == 0 && j == 0) {
                    cost[i][j] = dist;
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
                    if (trackPath) {
                        backtrack[i][j][0] = pi;
                        backtrack[i][j][1] = pj;
                    }
                }
            }
        }

        if (trackPath && pathOut != null) {
            int i = n - 1, j = m - 1;
            while (i >= 0 && j >= 0) {
                pathOut.add(0, new Pair<>(i, j));
                int ni = backtrack[i][j][0];
                int nj = backtrack[i][j][1];
                if (ni == i && nj == j) break;
                i = ni;
                j = nj;
            }
        }

        return Math.sqrt(cost[n - 1][m - 1]);
    }

    private double squaredDistance(double a, double b) {
        double diff = a - b;
        return diff * diff;
    }


}