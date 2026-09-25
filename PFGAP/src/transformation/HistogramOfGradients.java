package transformation;

import java.util.Objects;

/**
 * Histogram-of-gradients transformation for one-dimensional numeric series.
 *
 * <p>The source series is differentiated exactly once with
 * {@link DerivativeTransform}. Gradient values are then assigned to equal-width
 * bins spanning the observed derivative range. The histogram contains raw
 * counts as {@code double} values.</p>
 *
 * <p>The default bin count is {@code max(1, floor(sqrt(length)))}. Constant
 * derivative series are assigned entirely to the first bin, avoiding division
 * by zero and invalid indices. The implementation uses primitive loops rather
 * than streams and supports both {@code double[]} and {@code float[]} input.</p>
 */
public final class HistogramOfGradients {

    private HistogramOfGradients() {
    }

    public static double[] computeHistogram(double[] input) {
        Objects.requireNonNull(input, "Histogram input cannot be null.");
        return computeHistogram(input, defaultBinCount(input.length));
    }

    public static double[] computeHistogram(float[] input) {
        Objects.requireNonNull(input, "Histogram input cannot be null.");
        return computeHistogram(input, defaultBinCount(input.length));
    }

    public static double[] computeHistogram(
            double[] input,
            int numberOfBins
    ) {
        double[] gradients = DerivativeTransform.transform(input);
        return computeFromGradients(gradients, numberOfBins);
    }

    public static double[] computeHistogram(
            float[] input,
            int numberOfBins
    ) {
        double[] gradients = DerivativeTransform.transform(input);
        return computeFromGradients(gradients, numberOfBins);
    }

    /**
     * Computes a histogram from values that have already been transformed into
     * gradients. This entry point avoids applying the derivative twice in
     * callers that deliberately own derivative preprocessing.
     */
    public static double[] computeGradientHistogram(
            double[] gradients,
            int numberOfBins
    ) {
        Objects.requireNonNull(gradients, "Gradient input cannot be null.");
        if (gradients.length == 0) {
            throw new IllegalArgumentException(
                    "Gradient input cannot be empty."
            );
        }
        return computeFromGradients(gradients, numberOfBins);
    }

    public static double[] computeGradientHistogram(double[] gradients) {
        Objects.requireNonNull(gradients, "Gradient input cannot be null.");
        return computeGradientHistogram(
                gradients,
                defaultBinCount(gradients.length)
        );
    }

    private static double[] computeFromGradients(
            double[] gradients,
            int numberOfBins
    ) {
        if (numberOfBins <= 0) {
            throw new IllegalArgumentException(
                    "Histogram bin count must be positive."
            );
        }

        double minimum = gradients[0];
        double maximum = gradients[0];
        for (int index = 1; index < gradients.length; index++) {
            double value = gradients[index];
            if (value < minimum) {
                minimum = value;
            } else if (value > maximum) {
                maximum = value;
            }
        }

        double[] histogram = new double[numberOfBins];
        double range = maximum - minimum;
        if (range == 0.0) {
            histogram[0] = gradients.length;
            return histogram;
        }

        double scale = numberOfBins / range;
        for (double gradient : gradients) {
            int bin = (int) ((gradient - minimum) * scale);
            if (bin >= numberOfBins) {
                bin = numberOfBins - 1;
            }
            histogram[bin] += 1.0;
        }

        return histogram;
    }

    private static int defaultBinCount(int inputLength) {
        if (inputLength <= 0) {
            throw new IllegalArgumentException(
                    "Histogram input cannot be empty."
            );
        }
        return Math.max(1, (int) Math.sqrt(inputLength));
    }
}
