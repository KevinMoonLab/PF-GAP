package distance.multiTS;

import transformation.FirstOrderDifference;

import java.io.Serializable;

public class WDDTW_D implements Serializable {

    private double g;
    private double[] weightVector;

    public WDDTW_D() {}

    public synchronized double distance(Object Series1, Object Series2, double bsf, double g) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        int dims1 = series1.length;
        int dims2 = series2.length;

        if (dims1 != dims2) {
            throw new IllegalArgumentException("Both series must have the same number of dimensions (rows).");
        }

        int len1 = series1[0].length;
        int len2 = series2[0].length;

        for (int d = 0; d < dims1; d++) {
            if (series1[d].length != len1 || series2[d].length != len2) {
                throw new IllegalArgumentException("All dimensions must have consistent time lengths.");
            }
        }

        // Apply first-order difference transformation
        double[][] deriv1 = FirstOrderDifference.computeFirstOrderDifference(series1);
        double[][] deriv2 = FirstOrderDifference.computeFirstOrderDifference(series2);

        int lenDeriv1 = deriv1[0].length;
        int lenDeriv2 = deriv2[0].length;

        int maxLength = Math.max(lenDeriv1, lenDeriv2);
        initWeights(g, maxLength);

        double[][] cost = new double[lenDeriv1][lenDeriv2];

        for (int i = 0; i < lenDeriv1; i++) {
            for (int j = 0; j < lenDeriv2; j++) {
                double dist = squaredDistanceAt(deriv1, deriv2, i, j);
                double weight = weightVector[Math.abs(i - j)];

                if (i == 0 && j == 0) {
                    cost[i][j] = weight * dist;
                } else {
                    double minPrev = Double.POSITIVE_INFINITY;
                    if (i > 0 && j > 0) minPrev = Math.min(minPrev, cost[i - 1][j - 1]);
                    if (i > 0) minPrev = Math.min(minPrev, cost[i - 1][j]);
                    if (j > 0) minPrev = Math.min(minPrev, cost[i][j - 1]);

                    cost[i][j] = weight * dist + minPrev;
                }

                if (cost[i][j] > bsf * bsf) return Double.POSITIVE_INFINITY;
            }
        }

        double finalDist = Math.sqrt(cost[lenDeriv1 - 1][lenDeriv2 - 1]);
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

    private void initWeights(double g, int seriesLength) {
        if (this.g == g && this.weightVector != null && this.weightVector.length == seriesLength) return;

        this.g = g;
        this.weightVector = new double[seriesLength];
        double halfLength = (double) seriesLength / 2;

        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = 1.0 / (1 + Math.exp(-g * (i - halfLength)));
        }
    }
}