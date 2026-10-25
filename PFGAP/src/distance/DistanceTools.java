package distance;

import core.contracts.ObjectDataset;

import java.util.List;

/** Utility methods used by distance measures. */
public class DistanceTools {

    private static final double NEGATIVE_VARIANCE_TOLERANCE = 1.0e-15;

    public static int[] getInclusive10(int min, int max) {
        int[] output = new int[10];
        double difference = (double) (max - min) / 9.0;
        double current = min;
        output[0] = min;
        for (int index = 1; index < 9; index++) {
            current += difference;
            output[index] = (int) Math.round(current);
        }
        output[9] = max;
        return output;
    }

    public static double[] getInclusive10(double min, double max) {
        double[] output = new double[10];
        double difference = (max - min) / 9.0;
        output[0] = min;
        for (int index = 1; index < 9; index++) {
            output[index] = output[index - 1] + difference;
        }
        output[9] = max;
        return output;
    }

    public static int sim(double first, double second, double epsilon) {
        return Math.abs(first - second) <= epsilon ? 1 : 0;
    }

    /**
     * Computes population standard deviation over every value in the dataset.
     * Supports double[], float[], double[][], and float[][] series. For
     * multivariate series, values are pooled across dimensions and time points.
     * Unequal-length series and dimensions are supported.
     */
    public static double stdv_p(ObjectDataset train) {
        if (train == null) {
            throw new IllegalArgumentException(
                    "Cannot compute standard deviation from a null dataset."
            );
        }
        if (train.size() == 0) {
            throw new IllegalArgumentException(
                    "Cannot compute standard deviation from an empty dataset."
            );
        }

        PopulationVarianceAccumulator accumulator =
                new PopulationVarianceAccumulator();
        for (int caseIndex = 0; caseIndex < train.size(); caseIndex++) {
            addSeries(train.get_series(caseIndex), caseIndex, accumulator);
        }
        return accumulator.populationStandardDeviation();
    }

    /**
     * Computes population standard deviation over every value in a list of
     * univariate double series. Unequal-length series are supported.
     */
    public static double stdv_p(List<double[]> train) {
        if (train == null) {
            throw new IllegalArgumentException(
                    "Cannot compute standard deviation from a null list."
            );
        }
        if (train.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot compute standard deviation from an empty list."
            );
        }

        PopulationVarianceAccumulator accumulator =
                new PopulationVarianceAccumulator();
        for (int caseIndex = 0; caseIndex < train.size(); caseIndex++) {
            double[] series = train.get(caseIndex);
            if (series == null) {
                throw new IllegalArgumentException(
                        "Series at case " + caseIndex + " is null."
                );
            }
            addValues(series, caseIndex, -1, accumulator);
        }
        return accumulator.populationStandardDeviation();
    }

    private static void addSeries(
            Object series,
            int caseIndex,
            PopulationVarianceAccumulator accumulator
    ) {
        if (series == null) {
            throw new IllegalArgumentException(
                    "Series at case " + caseIndex + " is null."
            );
        }

        if (series instanceof double[] values) {
            addValues(values, caseIndex, -1, accumulator);
            return;
        }
        if (series instanceof float[] values) {
            addValues(values, caseIndex, -1, accumulator);
            return;
        }
        if (series instanceof double[][] dimensions) {
            for (int dimension = 0;
                 dimension < dimensions.length;
                 dimension++) {
                if (dimensions[dimension] == null) {
                    throw nullDimension(caseIndex, dimension);
                }
                addValues(
                        dimensions[dimension],
                        caseIndex,
                        dimension,
                        accumulator
                );
            }
            return;
        }
        if (series instanceof float[][] dimensions) {
            for (int dimension = 0;
                 dimension < dimensions.length;
                 dimension++) {
                if (dimensions[dimension] == null) {
                    throw nullDimension(caseIndex, dimension);
                }
                addValues(
                        dimensions[dimension],
                        caseIndex,
                        dimension,
                        accumulator
                );
            }
            return;
        }

        throw new IllegalArgumentException(
                "DistanceTools.stdv_p requires double[], float[], double[][], "
                        + "or float[][] series, but case " + caseIndex
                        + " contains " + series.getClass().getTypeName() + "."
        );
    }

    private static IllegalArgumentException nullDimension(
            int caseIndex,
            int dimension
    ) {
        return new IllegalArgumentException(
                "Case " + caseIndex + ", dimension " + dimension
                        + " is null."
        );
    }

    private static void addValues(
            double[] values,
            int caseIndex,
            int dimension,
            PopulationVarianceAccumulator accumulator
    ) {
        for (int timeIndex = 0; timeIndex < values.length; timeIndex++) {
            accumulator.add(
                    values[timeIndex],
                    caseIndex,
                    dimension,
                    timeIndex
            );
        }
    }

    private static void addValues(
            float[] values,
            int caseIndex,
            int dimension,
            PopulationVarianceAccumulator accumulator
    ) {
        for (int timeIndex = 0; timeIndex < values.length; timeIndex++) {
            accumulator.add(
                    values[timeIndex],
                    caseIndex,
                    dimension,
                    timeIndex
            );
        }
    }

    public static double Min3(
            final double first,
            final double second,
            final double third
    ) {
        return first <= second
                ? (first <= third ? first : third)
                : (second <= third ? second : third);
    }

    public static int ArgMin3(
            final double first,
            final double second,
            final double third
    ) {
        return first <= second
                ? (first <= third ? 0 : 2)
                : (second <= third ? 1 : 2);
    }

    public static double sum(final double... values) {
        double result = 0.0;
        for (double value : values) {
            result += value;
        }
        return result;
    }

    public static double max(final double... values) {
        double maximum = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (maximum < value) {
                maximum = value;
            }
        }
        return maximum;
    }

    public static double min(final double... values) {
        double minimum = Double.POSITIVE_INFINITY;
        for (double value : values) {
            if (value < minimum) {
                minimum = value;
            }
        }
        return minimum;
    }

    /** Numerically stable Welford population-variance accumulator. */
    private static final class PopulationVarianceAccumulator {
        private long count;
        private double mean;
        private double m2;

        private void add(
                double value,
                int caseIndex,
                int dimension,
                int timeIndex
        ) {
            if (!Double.isFinite(value)) {
                StringBuilder message = new StringBuilder()
                        .append("Cannot compute standard deviation from value ")
                        .append(value)
                        .append(" at case ")
                        .append(caseIndex);
                if (dimension >= 0) {
                    message.append(", dimension ").append(dimension);
                }
                message.append(", time index ")
                        .append(timeIndex)
                        .append('.');
                throw new IllegalArgumentException(message.toString());
            }

            count++;
            double delta = value - mean;
            mean += delta / count;
            double deltaFromUpdatedMean = value - mean;
            m2 += delta * deltaFromUpdatedMean;
        }

        private double populationStandardDeviation() {
            if (count == 0L) {
                throw new IllegalArgumentException(
                        "Cannot compute standard deviation because the input "
                                + "contains no numeric values."
                );
            }

            double variance = m2 / count;
            if (variance < 0.0
                    && variance >= -NEGATIVE_VARIANCE_TOLERANCE) {
                variance = 0.0;
            }
            if (!Double.isFinite(variance) || variance < 0.0) {
                throw new IllegalStateException(
                        "Computed invalid population variance " + variance
                                + " from " + count + " values."
                );
            }

            double standardDeviation = Math.sqrt(variance);
            if (!Double.isFinite(standardDeviation)) {
                throw new IllegalStateException(
                        "Computed non-finite population standard deviation "
                                + standardDeviation + " from " + count
                                + " values."
                );
            }
            return standardDeviation;
        }
    }
}
