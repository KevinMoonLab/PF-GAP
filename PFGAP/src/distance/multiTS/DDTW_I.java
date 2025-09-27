package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.DDTW;

import java.io.Serializable;

public class DDTW_I implements Serializable {

    private final DDTW ddtw;

    public DDTW_I() {
        this.ddtw = new DDTW();
    }

    /**
     * Computes the average DDTW distance across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     *
     * @param Series1    Object expected to be double[][]
     * @param Series2    Object expected to be double[][]
     * @param bsf        Early abandoning threshold
     * @param windowSize Sakoe-Chiba window size
     * @return Average DDTW distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf, int windowSize) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += ddtw.distance(series1[i], series2[i], bsf, windowSize);
        }

        return totalDistance / series1.length;
    }

    public int get_random_window(ObjectDataset d, java.util.Random r) {
        return ddtw.get_random_window(d, r);
    }
}

/*
package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.DDTW;

import java.io.Serializable;

public class DDTW_I implements Serializable {

    private final DDTW ddtw;

    public DDTW_I() {
        this.ddtw = new DDTW();
    }

    public synchronized double distance(Object Series1, Object Series2, double bsf, int windowSize) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length)
            throw new IllegalArgumentException("Both series must have the same number of rows.");

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += ddtw.distance(series1[i], series2[i], bsf, windowSize);
        }

        return totalDistance / series1.length;
    }

    public int get_random_window(ObjectDataset d, java.util.Random r) {
        return ddtw.get_random_window(d, r);
    }
}
 */