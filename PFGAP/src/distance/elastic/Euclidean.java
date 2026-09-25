package distance.elastic;

import core.AppContext;

import java.io.Serial;
import java.io.Serializable;

/**
 * Squared Euclidean distance kernels for primitive double and float vectors.
 *
 * <p>Every overload returns squared Euclidean distance. Float inputs remain
 * stored as {@code float[]} and are widened only for scalar arithmetic; no
 * temporary {@code double[]} is allocated.</p>
 *
 * <p>When {@code AppContext.useVectorApi} is true, contiguous all-feature
 * calculations use the JDK Vector API. Selected-feature calculations remain
 * scalar because their indexed access pattern is not normally a good SIMD
 * fit. A finite {@code bestSoFar} is checked after each vector block and each
 * scalar tail element. Consequently, vector early abandoning may evaluate up
 * to one complete SIMD block beyond the first element that crosses the bound,
 * while preserving the returned squared-distance semantics.</p>
 *
 * <p>This class is stateless and thread-safe. Input shape and selected-index
 * validity are trusted preconditions established by the dataset and splitter
 * layers rather than revalidated in the pairwise hot path.</p>
 */
public final class Euclidean implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public Euclidean() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return AppContext.useVectorApi
                    ? VectorKernels.squaredDistance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    )
                    : squaredDistance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return AppContext.useVectorApi
                    ? VectorKernels.squaredDistance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    )
                    : squaredDistance(
                            firstValues,
                            secondValues,
                            bestSoFar
                    );
        }

        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        if (selectedDimensions == null) {
            return distance(first, second, bestSoFar);
        }

        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return squaredDistance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    selectedDimensions
            );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return squaredDistance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    selectedDimensions
            );
        }

        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY
        );
    }

    public double distance(
            Object first,
            Object second,
            int[] selectedDimensions
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                selectedDimensions
        );
    }

    private static double squaredDistance(
            double[] first,
            double[] second,
            double bestSoFar
    ) {
        double total = 0.0;

        for (int index = 0; index < first.length; index++) {
            double difference = first[index] - second[index];
            total += difference * difference;

            if (total > bestSoFar) {
                return total;
            }
        }

        return total;
    }

    private static double squaredDistance(
            float[] first,
            float[] second,
            double bestSoFar
    ) {
        double total = 0.0;

        for (int index = 0; index < first.length; index++) {
            double difference =
                    (double) first[index] - (double) second[index];
            total += difference * difference;

            if (total > bestSoFar) {
                return total;
            }
        }

        return total;
    }

    private static double squaredDistance(
            double[] first,
            double[] second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        double total = 0.0;

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int feature = selectedDimensions[position];
            double difference = first[feature] - second[feature];
            total += difference * difference;

            if (total > bestSoFar) {
                return total;
            }
        }

        return total;
    }

    private static double squaredDistance(
            float[] first,
            float[] second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        double total = 0.0;

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int feature = selectedDimensions[position];
            double difference =
                    (double) first[feature]
                            - (double) second[feature];
            total += difference * difference;

            if (total > bestSoFar) {
                return total;
            }
        }

        return total;
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Euclidean distance requires matching double[] or float[] "
                        + "inputs. Received "
                        + typeName(first)
                        + " and "
                        + typeName(second)
                        + ". Mixed float/double pairs and boxed numeric "
                        + "arrays are not supported."
        );
    }

    private static String typeName(Object value) {
        return value == null
                ? "null"
                : value.getClass().getTypeName();
    }

    /**
     * Isolates all incubator-module references in a lazily initialized nested
     * class. The scalar path can therefore run without loading this class.
     */
    private static final class VectorKernels {

        private static final jdk.incubator.vector.VectorSpecies<Double>
                DOUBLE_SPECIES =
                jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED;

        private static final jdk.incubator.vector.VectorSpecies<Float>
                FLOAT_SPECIES =
                jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;

        private VectorKernels() {
        }

        private static double squaredDistance(
                double[] first,
                double[] second,
                double bestSoFar
        ) {
            int upperBound = DOUBLE_SPECIES.loopBound(first.length);
            int index = 0;
            double total = 0.0;

            if (Double.isInfinite(bestSoFar)) {
                jdk.incubator.vector.DoubleVector accumulated =
                        jdk.incubator.vector.DoubleVector.zero(DOUBLE_SPECIES);

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
                    accumulated = accumulated.add(
                            difference.mul(difference)
                    );
                }

                total = accumulated.reduceLanes(
                        jdk.incubator.vector.VectorOperators.ADD
                );
            } else {
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

                    if (total > bestSoFar) {
                        return total;
                    }
                }
            }

            for (; index < first.length; index++) {
                double difference = first[index] - second[index];
                total += difference * difference;

                if (total > bestSoFar) {
                    return total;
                }
            }

            return total;
        }

        private static double squaredDistance(
                float[] first,
                float[] second,
                double bestSoFar
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

                if (total > bestSoFar) {
                    return total;
                }
            }

            for (; index < first.length; index++) {
                double difference =
                        (double) first[index] - (double) second[index];
                total += difference * difference;

                if (total > bestSoFar) {
                    return total;
                }
            }

            return total;
        }
    }
}
