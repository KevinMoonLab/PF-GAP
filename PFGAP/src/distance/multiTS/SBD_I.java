package distance.multiTS;

import distance.elastic.SBD;

import java.io.Serializable;

public class SBD_I implements Serializable {

    private final SBD sbd;

    public SBD_I() {
        this.sbd = new SBD();
    }

    /**
     * Computes the average SBD (Shape-Based Distance) across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * SBD uses normalized cross-correlation and returns 1 - max correlation.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @return Average SBD distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += sbd.distance(series1[i], series2[i]);
        }

        return totalDistance / series1.length;
    }
}