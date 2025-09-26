package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.Euclidean;

import java.io.Serializable;

public class Euclidean_I implements Serializable {

    private final Euclidean euclidean;

    public Euclidean_I() {
        this.euclidean = new Euclidean();
    }

    /**
     * Computes the average Euclidean distance across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * Early abandoning is supported via the bsf threshold.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @return Average Euclidean distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += euclidean.distance(series1[i], series2[i], bsf);
        }

        return totalDistance / series1.length;
    }
}