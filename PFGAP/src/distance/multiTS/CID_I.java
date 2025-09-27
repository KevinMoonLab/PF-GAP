package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.CID;

import java.io.Serializable;

public class CID_I implements Serializable {

    private final CID cid;

    public CID_I() {
        this.cid = new CID();
    }

    /**
     * Computes the average CID (Complexity-Invariant Distance) across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * CID scales Euclidean distance by a complexity correction factor.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @return Average CID distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += cid.distance(series1[i], series2[i], bsf);
        }

        return totalDistance / series1.length;
    }
}