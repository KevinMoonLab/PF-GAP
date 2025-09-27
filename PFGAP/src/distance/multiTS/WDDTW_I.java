package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.WDDTW;

import java.io.Serializable;

public class WDDTW_I implements Serializable {

    private final WDDTW wddtw;

    public WDDTW_I() {
        this.wddtw = new WDDTW();
    }

    /**
     * Computes the average WDDTW distance across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * WDDTW applies derivative transformation before computing weighted DTW.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @param g Weighting parameter for logistic curve
     * @return Average WDDTW distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf, double g) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += wddtw.distance(series1[i], series2[i], bsf, g);
        }

        return totalDistance / series1.length;
    }

    public double get_random_g(ObjectDataset d, java.util.Random r) {
        return wddtw.get_random_g(d, r);
    }
}