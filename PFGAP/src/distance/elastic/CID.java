package distance.elastic;

import core.AppContext;

import java.io.Serial;
import java.io.Serializable;

/**
 * Complexity-Invariant Distance for primitive univariate series.
 *
 * <p>The original CID definition is Euclidean distance multiplied by the
 * ratio of the larger complexity estimate to the smaller. To remain
 * consistent with PFGAP's squared-distance kernels, this implementation
 * returns the algebraically equivalent squared value:</p>
 *
 * <pre>
 * squaredCID = squaredEuclidean
 *         * max(squaredComplexity1, squaredComplexity2)
 *         / min(squaredComplexity1, squaredComplexity2)
 * </pre>
 *
 * <p>Complexity is the sum of squared first differences. Two constant series
 * use correction factor one. If exactly one series is constant, the
 * correction is infinite, matching the limiting behavior of the CID ratio.</p>
 *
 * <p>A finite {@code bestSoFar} is interpreted in squared-CID units and is
 * converted to a squared-Euclidean cutoff after both complexities are known.
 * Contiguous full-series calculations optionally use the JDK Vector API when
 * {@code AppContext.useVectorApi} is true.</p>
 */
public final class CID implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public CID() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return AppContext.useVectorApi
                    ? VectorKernels.distance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    )
                    : scalarDistance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return AppContext.useVectorApi
                    ? VectorKernels.distance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    )
                    : scalarDistance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    );
        }

        throw unsupportedPair(first, second);
    }

    public double distance(Object first, Object second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    private static double scalarDistance(
            double[] first,
            double[] second,
            double bestSoFar
    ) {
        double firstComplexity = squaredComplexity(first);
        double secondComplexity = squaredComplexity(second);
        double correction = squaredCorrection(
                firstComplexity,
                secondComplexity
        );

        if (correction == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }

        double euclideanCutoff = euclideanCutoff(bestSoFar, correction);
        double squaredEuclidean = 0.0;
        for (int index = 0; index < first.length; index++) {
            double difference = first[index] - second[index];
            squaredEuclidean += difference * difference;
            if (squaredEuclidean > euclideanCutoff) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return squaredEuclidean * correction;
    }

    private static double scalarDistance(
            float[] first,
            float[] second,
            double bestSoFar
    ) {
        double firstComplexity = squaredComplexity(first);
        double secondComplexity = squaredComplexity(second);
        double correction = squaredCorrection(
                firstComplexity,
                secondComplexity
        );

        if (correction == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }

        double euclideanCutoff = euclideanCutoff(bestSoFar, correction);
        double squaredEuclidean = 0.0;
        for (int index = 0; index < first.length; index++) {
            double difference = (double) first[index] - (double) second[index];
            squaredEuclidean += difference * difference;
            if (squaredEuclidean > euclideanCutoff) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return squaredEuclidean * correction;
    }

    private static double squaredComplexity(double[] series) {
        double total = 0.0;
        for (int index = 1; index < series.length; index++) {
            double difference = series[index] - series[index - 1];
            total += difference * difference;
        }
        return total;
    }

    private static double squaredComplexity(float[] series) {
        double total = 0.0;
        for (int index = 1; index < series.length; index++) {
            double difference =
                    (double) series[index] - (double) series[index - 1];
            total += difference * difference;
        }
        return total;
    }

    private static double squaredCorrection(
            double firstComplexity,
            double secondComplexity
    ) {
        if (firstComplexity == 0.0) {
            return secondComplexity == 0.0
                    ? 1.0
                    : Double.POSITIVE_INFINITY;
        }
        if (secondComplexity == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.max(firstComplexity, secondComplexity)
                / Math.min(firstComplexity, secondComplexity);
    }

    private static double euclideanCutoff(
            double bestSoFar,
            double correction
    ) {
        if (Double.isNaN(bestSoFar)) {
            throw new IllegalArgumentException(
                    "CID bestSoFar cannot be NaN."
            );
        }
        if (bestSoFar == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }
        return Math.max(0.0, bestSoFar) / correction;
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "CID requires matching double[] or float[] inputs. Received "
                        + typeName(first)
                        + " and "
                        + typeName(second)
                        + "."
        );
    }

    private static String typeName(Object value) {
        return value == null
                ? "null"
                : value.getClass().getTypeName();
    }

    /** Isolates all incubator-module references in a lazily loaded class. */
    private static final class VectorKernels {

        private static final jdk.incubator.vector.VectorSpecies<Double>
                DOUBLE_SPECIES =
                jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED;

        private static final jdk.incubator.vector.VectorSpecies<Float>
                FLOAT_SPECIES =
                jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;

        private VectorKernels() {
        }

        private static double distance(
                double[] first,
                double[] second,
                double bestSoFar
        ) {
            double firstComplexity = squaredComplexity(first);
            double secondComplexity = squaredComplexity(second);
            double correction = squaredCorrection(
                    firstComplexity,
                    secondComplexity
            );
            if (correction == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            double cutoff = euclideanCutoff(bestSoFar, correction);
            return squaredEuclidean(first, second, cutoff) * correction;
        }

        private static double distance(
                float[] first,
                float[] second,
                double bestSoFar
        ) {
            double firstComplexity = squaredComplexity(first);
            double secondComplexity = squaredComplexity(second);
            double correction = squaredCorrection(
                    firstComplexity,
                    secondComplexity
            );
            if (correction == Double.POSITIVE_INFINITY) {
                return Double.POSITIVE_INFINITY;
            }

            double cutoff = euclideanCutoff(bestSoFar, correction);
            return squaredEuclidean(first, second, cutoff) * correction;
        }

        private static double squaredComplexity(double[] series) {
            int differenceCount = series.length - 1;
            int upperBound = DOUBLE_SPECIES.loopBound(differenceCount);
            int offset = 0;
            jdk.incubator.vector.DoubleVector accumulated =
                    jdk.incubator.vector.DoubleVector.zero(DOUBLE_SPECIES);

            for (; offset < upperBound;
                 offset += DOUBLE_SPECIES.length()) {
                jdk.incubator.vector.DoubleVector current =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES,
                                series,
                                offset + 1
                        );
                jdk.incubator.vector.DoubleVector previous =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES,
                                series,
                                offset
                        );
                jdk.incubator.vector.DoubleVector difference =
                        current.sub(previous);
                accumulated = accumulated.add(
                        difference.mul(difference)
                );
            }

            double total = accumulated.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );
            for (int index = offset + 1;
                 index < series.length;
                 index++) {
                double difference = series[index] - series[index - 1];
                total += difference * difference;
            }
            return total;
        }

        private static double squaredComplexity(float[] series) {
            int differenceCount = series.length - 1;
            int upperBound = FLOAT_SPECIES.loopBound(differenceCount);
            int offset = 0;
            double total = 0.0;

            for (; offset < upperBound;
                 offset += FLOAT_SPECIES.length()) {
                jdk.incubator.vector.FloatVector current =
                        jdk.incubator.vector.FloatVector.fromArray(
                                FLOAT_SPECIES,
                                series,
                                offset + 1
                        );
                jdk.incubator.vector.FloatVector previous =
                        jdk.incubator.vector.FloatVector.fromArray(
                                FLOAT_SPECIES,
                                series,
                                offset
                        );
                jdk.incubator.vector.FloatVector difference =
                        current.sub(previous);
                total += (double) difference.mul(difference).reduceLanes(
                        jdk.incubator.vector.VectorOperators.ADD
                );
            }

            for (int index = offset + 1;
                 index < series.length;
                 index++) {
                double difference =
                        (double) series[index] - (double) series[index - 1];
                total += difference * difference;
            }
            return total;
        }

        private static double squaredEuclidean(
                double[] first,
                double[] second,
                double cutoff
        ) {
            int upperBound = DOUBLE_SPECIES.loopBound(first.length);
            int index = 0;
            double total = 0.0;

            for (; index < upperBound;
                 index += DOUBLE_SPECIES.length()) {
                jdk.incubator.vector.DoubleVector difference =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES,
                                first,
                                index
                        ).sub(
                                jdk.incubator.vector.DoubleVector.fromArray(
                                        DOUBLE_SPECIES,
                                        second,
                                        index
                                )
                        );
                total += difference.mul(difference).reduceLanes(
                        jdk.incubator.vector.VectorOperators.ADD
                );
                if (total > cutoff) {
                    return Double.POSITIVE_INFINITY;
                }
            }

            for (; index < first.length; index++) {
                double difference = first[index] - second[index];
                total += difference * difference;
                if (total > cutoff) {
                    return Double.POSITIVE_INFINITY;
                }
            }
            return total;
        }

        private static double squaredEuclidean(
                float[] first,
                float[] second,
                double cutoff
        ) {
            int upperBound = FLOAT_SPECIES.loopBound(first.length);
            int index = 0;
            double total = 0.0;

            for (; index < upperBound;
                 index += FLOAT_SPECIES.length()) {
                jdk.incubator.vector.FloatVector difference =
                        jdk.incubator.vector.FloatVector.fromArray(
                                FLOAT_SPECIES,
                                first,
                                index
                        ).sub(
                                jdk.incubator.vector.FloatVector.fromArray(
                                        FLOAT_SPECIES,
                                        second,
                                        index
                                )
                        );
                total += (double) difference.mul(difference).reduceLanes(
                        jdk.incubator.vector.VectorOperators.ADD
                );
                if (total > cutoff) {
                    return Double.POSITIVE_INFINITY;
                }
            }

            for (; index < first.length; index++) {
                double difference =
                        (double) first[index] - (double) second[index];
                total += difference * difference;
                if (total > cutoff) {
                    return Double.POSITIVE_INFINITY;
                }
            }
            return total;
        }
    }
}
