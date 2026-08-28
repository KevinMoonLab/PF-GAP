package purity.isolation;

import java.util.List;

public class IsolationPathLength {

    private IsolationPathLength() {
        // Utility class
    }

    public static double compute(Object[] labels) {

        if (labels == null || labels.length <= 1) {
            return 0.0;
        }

        return correction(labels.length);
    }

    public static double compute(List<Object> labels) {

        if (labels == null || labels.size() <= 1) {
            return 0.0;
        }

        return correction(labels.size());
    }

    public static double compute(int nodeSize) {

        if (nodeSize <= 1) {
            return 0.0;
        }

        return correction(nodeSize);
    }

    public static double computeFromChildSizes(int[] childSizes) {

        if (childSizes == null || childSizes.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        int parentSize = 0;

        for (int size : childSizes) {
            if (size <= 0) {
                return Double.POSITIVE_INFINITY;
            }

            parentSize += size;
        }

        if (parentSize <= 0) {
            return Double.POSITIVE_INFINITY;
        }

        double weightedCorrection = 0.0;

        for (int size : childSizes) {
            double weight = (double) size / parentSize;
            weightedCorrection += weight * correction(size);
        }

        return weightedCorrection;
    }

    public static double correction(int n) {

        if (n <= 1) {
            return 0.0;
        }

        if (n == 2) {
            return 1.0;
        }

        return 2.0 * harmonic(n - 1) - (2.0 * (n - 1) / n);
    }

    private static double harmonic(int n) {

        double sum = 0.0;

        for (int i = 1; i <= n; i++) {
            sum += 1.0 / i;
        }

        return sum;
    }
}