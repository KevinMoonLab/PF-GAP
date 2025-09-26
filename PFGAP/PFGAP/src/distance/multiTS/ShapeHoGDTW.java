package distance.multiTS;

import transformation.FirstOrderDifference;
import transformation.MultivariateHistogramOfGradients;
import transformation.MultivariateHistogramOfGradients.Strategy;

import java.io.Serializable;

public class ShapeHoGDTW implements Serializable {

    public ShapeHoGDTW() {}

    /**
     * Computes the DTW distance between two multivariate time series after applying
     * first-order differencing and multivariate histogram of gradients.
     *
     * @param Series1 Object expected to be double[][]
     * @param Series2 Object expected to be double[][]
     * @param bsf Early abandoning threshold
     * @param windowSize Sakoe-Chiba window size
     * @return DTW distance between histogram representations
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf, int windowSize) {
        double[][] series1 = (double[][]) Series1;
        double[][] series2 = (double[][]) Series2;

        // Apply first-order difference
        double[][] diff1 = FirstOrderDifference.computeFirstOrderDifference(series1);
        double[][] diff2 = FirstOrderDifference.computeFirstOrderDifference(series2);

        // Compute multivariate histograms using default strategy
        double[] hist1 = MultivariateHistogramOfGradients.computeHistogram(diff1, Strategy.CONCATENATE_GRADIENTS);
        double[] hist2 = MultivariateHistogramOfGradients.computeHistogram(diff2, Strategy.CONCATENATE_GRADIENTS);

        // Apply univariate DTW to the histogram vectors
        return new distance.elastic.DTW().distance(hist1, hist2, bsf, windowSize);
    }
}