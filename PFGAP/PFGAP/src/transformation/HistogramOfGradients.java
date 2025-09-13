package transformation;

import java.util.Arrays;

public class HistogramOfGradients {

    // Default version with the square root number of bins.
    public static double[] computeHistogram(double[] input) {
        int binNum = (int) Math.sqrt(input.length);
        return computeHistogram(input, binNum);
    }

    public static double[] computeHistogram(double[] input, int numBins) {
        double[] gradients = FirstOrderDifference.computeFirstOrderDifference(input);
        //double[] gradients = (double[]) gradientObj;

        if (gradients.length == 0 || numBins <= 0) {
            return new double[0];
        }

        double min = Arrays.stream(gradients).min().orElse(0);
        double max = Arrays.stream(gradients).max().orElse(1);
        double binSize = (max - min) / numBins;

        int[] histogram = new int[numBins];
        for (double g : gradients) {
            int bin = (int) ((g - min) / binSize);
            if (bin == numBins) bin--; // edge case
            histogram[bin]++;
        }

        double[] histogramAsDouble = Arrays.stream(histogram).asDoubleStream().toArray();
        return histogramAsDouble;
    }
}
