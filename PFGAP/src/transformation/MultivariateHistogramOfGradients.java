package transformation;

import java.util.Objects;

/**
 * Histogram construction for multivariate gradient matrices.
 *
 * <p>Input is expected to be an already-derived matrix with orientation
 * {@code [dimension][time]}. This class never applies
 * {@link DerivativeTransform}; callers own derivative preprocessing. This
 * contract prevents the accidental double differentiation that previously
 * occurred in ShapeHoG distance pipelines.</p>
 *
 * <p>All strategies use primitive arrays and loops. Histogram values are raw
 * counts represented as {@code double}.</p>
 */
public final class MultivariateHistogramOfGradients {

    public enum Strategy {
        CONCATENATE_GRADIENTS,
        PER_DIMENSION,
        PCA_BEFORE_HISTOGRAM,
        AVERAGE_GRADIENTS
    }

    private MultivariateHistogramOfGradients() {
    }

    public static double[] computeHistogram(
            double[][] gradients,
            int binsPerDimension,
            Strategy strategy
    ) {
        requireGradientMatrix(gradients);
        requirePositiveBins(binsPerDimension);
        Objects.requireNonNull(strategy, "Histogram strategy cannot be null.");

        return switch (strategy) {
            case CONCATENATE_GRADIENTS ->
                    histogramFromConcatenatedGradients(
                            gradients,
                            binsPerDimension
                    );
            case PER_DIMENSION ->
                    histogramFromPerDimension(
                            gradients,
                            binsPerDimension
                    );
            case PCA_BEFORE_HISTOGRAM ->
                    histogramAfterPca(
                            gradients,
                            binsPerDimension
                    );
            case AVERAGE_GRADIENTS ->
                    histogramFromAveragedGradients(
                            gradients,
                            binsPerDimension
                    );
        };
    }

    public static double[] computeHistogram(
            double[][] gradients,
            Strategy strategy
    ) {
        requireGradientMatrix(gradients);
        return computeHistogram(
                gradients,
                defaultBinCount(gradients[0].length),
                strategy
        );
    }

    private static double[] histogramFromConcatenatedGradients(
            double[][] gradients,
            int bins
    ) {
        int totalLength = 0;
        for (double[] dimension : gradients) {
            totalLength = Math.addExact(totalLength, dimension.length);
        }

        double[] flattened = new double[totalLength];
        int output = 0;
        for (double[] dimension : gradients) {
            System.arraycopy(
                    dimension,
                    0,
                    flattened,
                    output,
                    dimension.length
            );
            output += dimension.length;
        }

        return HistogramOfGradients.computeGradientHistogram(
                flattened,
                bins
        );
    }

    private static double[] histogramFromPerDimension(
            double[][] gradients,
            int bins
    ) {
        double[] combined = new double[
                Math.multiplyExact(gradients.length, bins)
        ];
        int output = 0;

        for (double[] dimension : gradients) {
            double[] histogram =
                    HistogramOfGradients.computeGradientHistogram(
                            dimension,
                            bins
                    );
            System.arraycopy(
                    histogram,
                    0,
                    combined,
                    output,
                    bins
            );
            output += bins;
        }

        return combined;
    }

    private static double[] histogramAfterPca(
            double[][] gradients,
            int bins
    ) {
        double[][] projected = PCA.transform(gradients, 1);
        return HistogramOfGradients.computeGradientHistogram(
                projected[0],
                bins
        );
    }

    private static double[] histogramFromAveragedGradients(
            double[][] gradients,
            int bins
    ) {
        int timeLength = gradients[0].length;
        double[] average = new double[timeLength];

        for (double[] dimension : gradients) {
            for (int time = 0; time < timeLength; time++) {
                average[time] += dimension[time];
            }
        }

        double reciprocalDimensionCount = 1.0 / gradients.length;
        for (int time = 0; time < timeLength; time++) {
            average[time] *= reciprocalDimensionCount;
        }

        return HistogramOfGradients.computeGradientHistogram(
                average,
                bins
        );
    }

    private static int defaultBinCount(int timeLength) {
        return Math.max(1, (int) Math.sqrt(timeLength));
    }

    private static void requireGradientMatrix(double[][] gradients) {
        Objects.requireNonNull(
                gradients,
                "Gradient matrix cannot be null."
        );
        if (gradients.length == 0) {
            throw new IllegalArgumentException(
                    "Gradient matrix cannot have zero dimensions."
            );
        }

        int timeLength = -1;
        for (int dimension = 0;
             dimension < gradients.length;
             dimension++) {
            double[] row = Objects.requireNonNull(
                    gradients[dimension],
                    "Gradient row cannot be null at dimension "
                            + dimension
                            + "."
            );
            if (row.length == 0) {
                throw new IllegalArgumentException(
                        "Gradient rows cannot be empty."
                );
            }
            if (timeLength < 0) {
                timeLength = row.length;
            } else if (row.length != timeLength) {
                throw new IllegalArgumentException(
                        "All gradient rows must have equal time length."
                );
            }
        }
    }

    private static void requirePositiveBins(int bins) {
        if (bins <= 0) {
            throw new IllegalArgumentException(
                    "Histogram bin count must be positive."
            );
        }
    }
}
