package transformation;

import java.util.ArrayList;
import java.util.List;

public class MultivariateHistogramOfGradients {

    public enum Strategy {
        CONCATENATE_GRADIENTS,
        PER_DIMENSION,
        PCA_BEFORE_HISTOGRAM,
        AVERAGE_GRADIENTS
    }

    public static double[] computeHistogram(double[][] input, int binsPerDim, Strategy strategy) {
        if (input == null || input.length == 0) {
            return new double[0];
        }

        switch (strategy) {
            case CONCATENATE_GRADIENTS:
                return histogramFromConcatenatedGradients(input, binsPerDim);
            case PER_DIMENSION:
                return histogramFromPerDimension(input, binsPerDim);
            case PCA_BEFORE_HISTOGRAM:
                double[][] projected = PCA.transform(input, 1); // reduce to 1D
                return HistogramOfGradients.computeHistogram(projected[0], binsPerDim);
            case AVERAGE_GRADIENTS:
                return histogramFromAveragedGradients(input, binsPerDim);
            default:
                throw new IllegalArgumentException("Unknown strategy: " + strategy);
        }
    }

    public static double[] computeHistogram(double[][] input, Strategy strategy) {
        int binsPerDim = (int) Math.sqrt(input[0].length);
        return computeHistogram(input, binsPerDim, strategy);
    }

    private static double[] histogramFromConcatenatedGradients(double[][] input, int bins) {
        List<Double> allGradients = new ArrayList<>();
        for (double[] dim : input) {
            double[] grad = FirstOrderDifference.computeFirstOrderDifference(dim);
            for (double g : grad) allGradients.add(g);
        }
        double[] flat = allGradients.stream().mapToDouble(Double::doubleValue).toArray();
        return HistogramOfGradients.computeHistogram(flat, bins);
    }

    private static double[] histogramFromPerDimension(double[][] input, int bins) {
        List<double[]> histograms = new ArrayList<>();
        for (double[] dim : input) {
            double[] grad = FirstOrderDifference.computeFirstOrderDifference(dim);
            double[] hist = HistogramOfGradients.computeHistogram(grad, bins);
            histograms.add(hist);
        }

        int totalLength = histograms.stream().mapToInt(h -> h.length).sum();
        double[] combined = new double[totalLength];
        int pos = 0;
        for (double[] h : histograms) {
            System.arraycopy(h, 0, combined, pos, h.length);
            pos += h.length;
        }
        return combined;
    }

    private static double[] histogramFromAveragedGradients(double[][] input, int bins) {
        int len = input[0].length - 1;
        double[] avgGrad = new double[len];
        for (double[] dim : input) {
            double[] grad = FirstOrderDifference.computeFirstOrderDifference(dim);
            for (int i = 0; i < len; i++) {
                avgGrad[i] += grad[i];
            }
        }
        for (int i = 0; i < len; i++) {
            avgGrad[i] /= input.length;
        }
        return HistogramOfGradients.computeHistogram(avgGrad, bins);
    }
}