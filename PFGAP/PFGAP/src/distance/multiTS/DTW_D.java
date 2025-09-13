package distance.multiTS;

import java.io.Serializable;

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

        if (windowSize == -1) {
            windowSize = Math.max(len1, len2);
        }

        double[] prevRow = new double[len2];
        double[] currentRow = new double[len2];

        // Initialize first row
        prevRow[0] = squaredDistanceAt(series1, series2, 0, 0);
        if (prevRow[0] > bsf * bsf) return Double.POSITIVE_INFINITY;

        for (int j = 1; j < Math.min(len2, 1 + windowSize); j++) {
            prevRow[j] = prevRow[j - 1] + squaredDistanceAt(series1, series2, 0, j);
            if (prevRow[j] > bsf * bsf) return Double.POSITIVE_INFINITY;
        }

        // Second row
        if (len1 >= 2) {
            currentRow[0] = prevRow[0] + squaredDistanceAt(series1, series2, 1, 0);
            if (currentRow[0] > bsf * bsf) return Double.POSITIVE_INFINITY;

            currentRow[1] = prevRow[0] + squaredDistanceAt(series1, series2, 1, 1);
            if (currentRow[1] > bsf * bsf) return Double.POSITIVE_INFINITY;

            for (int j = 2; j < Math.min(len2, windowSize + 2); j++) {
                currentRow[j] = Math.min(currentRow[j - 1], prevRow[j - 1]) + squaredDistanceAt(series1, series2, 1, j);
                if (currentRow[j] > bsf * bsf) return Double.POSITIVE_INFINITY;
            }
        }

        // Remaining rows
        for (int i = 2; i < len1; i++) {
            int jStart = Math.max(0, i - windowSize);
            int jStop = Math.min(len2 - 1, i + windowSize);

            double[] tmp = prevRow;
            prevRow = currentRow;
            currentRow = tmp;

            currentRow[jStart] = Math.min(prevRow[jStart], prevRow[Math.max(0, jStart - 1)]) + squaredDistanceAt(series1, series2, i, jStart);
            if (currentRow[jStart] > bsf * bsf) return Double.POSITIVE_INFINITY;

            for (int j = jStart + 1; j <= jStop; j++) {
                currentRow[j] = min(prevRow[j - 1], currentRow[j - 1], prevRow[j]) + squaredDistanceAt(series1, series2, i, j);
                if (currentRow[j] > bsf * bsf) return Double.POSITIVE_INFINITY;
            }
        }

        double finalDist = Math.sqrt(currentRow[len2 - 1]);
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

    private double min(double a, double b, double c) {
        return Math.min(a, Math.min(b, c));
    }
}

