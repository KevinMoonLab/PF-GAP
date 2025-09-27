package distance.multiTS;

import core.contracts.ObjectDataset;
import distance.elastic.ShapeHoG1dDTW;

import java.io.Serializable;

public class ShapeHoG1dDTW_I implements Serializable {

    private final ShapeHoG1dDTW hogdtw; //this might need to be fixed...

    public ShapeHoG1dDTW_I() {
        this.hogdtw = new ShapeHoG1dDTW();
    }

    /**
     * Computes the average ShapeHoG1dDTW distance across all rows of the input matrices.
     * Each row in series1 is compared to the corresponding row in series2.
     * This method applies first-order differencing and histogram of gradients before DTW.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @param windowSize Sakoe-Chiba window size
     * @return Average ShapeHoG1dDTW distance across all rows
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf, int windowSize) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Both series must have the same number of rows.");
        }

        double totalDistance = 0.0;
        for (int i = 0; i < series1.length; i++) {
            totalDistance += hogdtw.distance(series1[i], series2[i], bsf, windowSize);
        }

        return totalDistance / series1.length;
    }

    public int get_random_window(ObjectDataset d, java.util.Random r) {
        return hogdtw.get_random_window(d, r); // if needed, otherwise remove
    }
}