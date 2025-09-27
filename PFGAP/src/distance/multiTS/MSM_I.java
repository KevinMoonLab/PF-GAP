package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.MSM;

import java.io.Serializable;

public class MSM_I implements Serializable {

    private final MSM msm;

    public MSM_I() {
        this.msm = new MSM();
    }

    /**
     * Computes the average MSM distance across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * MSM uses a cost parameter to penalize move/split/merge operations.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @param cost Cost parameter for MSM operations
     * @return Average MSM distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf, double cost) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += msm.distance(series1[i], series2[i], bsf, cost);
        }

        return totalDistance / series1.length;
    }

    public double get_random_cost(ObjectDataset d, java.util.Random r) {
        return msm.get_random_cost(d, r);
    }
}