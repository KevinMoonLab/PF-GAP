package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.ERP;

import java.io.Serializable;

public class ERP_I implements Serializable {

    private final ERP erp;

    public ERP_I() {
        this.erp = new ERP();
    }

    /**
     * Computes the average ERP distance across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * ERP uses a gap penalty (gValue) and a Sakoe-Chiba window.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @param windowSize Sakoe-Chiba window size
     * @param gValue Gap penalty value
     * @return Average ERP distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf, int windowSize, double gValue) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += erp.distance(series1[i], series2[i], bsf, windowSize, gValue);
        }

        return totalDistance / series1.length;
    }

    public int get_random_window(ObjectDataset d, java.util.Random r) {
        return erp.get_random_window(d, r);
    }

    public double get_random_g(ObjectDataset d, java.util.Random r) {
        return erp.get_random_g(d, r);
    }
}
