package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.TWE;

import java.io.Serializable;

public class TWE_I implements Serializable {

    private final TWE twe;

    public TWE_I() {
        this.twe = new TWE();
    }

    /**
     * Computes the average TWE distance across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * TWE uses edit-based operations with stiffness (nu) and penalty (lambda).
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @param nu Stiffness parameter
     * @param lambda Penalty parameter
     * @return Average TWE distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf, double nu, double lambda) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += twe.distance(series1[i], series2[i], bsf, nu, lambda);
        }

        return totalDistance / series1.length;
    }

    public double get_random_nu(ObjectDataset d, java.util.Random r) {
        return twe.get_random_nu(d, r);
    }

    public double get_random_lambda(ObjectDataset d, java.util.Random r) {
        return twe.get_random_lambda(d, r);
    }
}