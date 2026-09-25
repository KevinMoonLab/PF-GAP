package distance.elastic;

import core.AppContext;

import java.io.Serial;
import java.io.Serializable;

/**
 * Manhattan distance kernels for primitive double and float vectors.
 *
 * <p>Every overload returns the sum of absolute coordinate differences.
 * Float inputs remain stored as {@code float[]} and no temporary
 * {@code double[]} is allocated.</p>
 *
 * <p>When {@code AppContext.useVectorApi} is true, contiguous all-feature
 * calculations use the JDK Vector API. Selected-feature calculations remain
 * scalar because they use indirect indexed access. A finite
 * {@code bestSoFar} is checked after every vector block and every scalar tail
 * element. Consequently, vector early abandoning may evaluate up to one SIMD
 * block beyond the first coordinate that crosses the bound.</p>
 *
 * <p>This class is stateless and thread-safe. Equal vector lengths and valid
 * selected indices are trusted preconditions established by the dataset and
 * splitter layers rather than revalidated in the pairwise hot path.</p>
 */
public final class Manhattan implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public Manhattan() {
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
            return scalarDistance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    selectedDimensions
            );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return scalarDistance(
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

    private static double scalarDistance(
            double[] first,
            double[] second,
            double bestSoFar
    ) {
        double total = 0.0;

        for (int index = 0; index < first.length; index++) {
            total += Math.abs(first[index] - second[index]);

            if (total > bestSoFar) {
                return total;
            }
        }

        return total;
    }

    private static double scalarDistance(
            float[] first,
            float[] second,
            double bestSoFar
    ) {
        double total = 0.0;

        for (int index = 0; index < first.length; index++) {
            total += Math.abs(
                    (double) first[index] - (double) second[index]
            );

            if (total > bestSoFar) {
                return total;
            }
        }

        return total;
    }

    private static double scalarDistance(
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
            total += Math.abs(first[feature] - second[feature]);

            if (total > bestSoFar) {
                return total;
            }
        }

        return total;
    }

    private static double scalarDistance(
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
            total += Math.abs(
                    (double) first[feature] - (double) second[feature]
            );

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
                "Manhattan distance requires matching double[] or float[] "
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
     * class. The scalar path can therefore run without initializing the vector
     * implementation.
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

        private static double distance(
                double[] first,
                double[] second,
                double bestSoFar
        ) {
            int upperBound = DOUBLE_SPECIES.loopBound(first.length);
            int index = 0;
            double total = 0.0;

            if (bestSoFar == Double.POSITIVE_INFINITY) {
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
                            ).abs();
                    accumulated = accumulated.add(difference);
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
                            ).abs();
                    total += difference.reduceLanes(
                            jdk.incubator.vector.VectorOperators.ADD
                    );

                    if (total > bestSoFar) {
                        return total;
                    }
                }
            }

            for (; index < first.length; index++) {
                total += Math.abs(first[index] - second[index]);

                if (total > bestSoFar) {
                    return total;
                }
            }

            return total;
        }

        private static double distance(
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
                        ).abs();

                total += (double) difference.reduceLanes(
                        jdk.incubator.vector.VectorOperators.ADD
                );

                if (total > bestSoFar) {
                    return total;
                }
            }

            for (; index < first.length; index++) {
                total += Math.abs(
                        (double) first[index] - (double) second[index]
                );

                if (total > bestSoFar) {
                    return total;
                }
            }

            return total;
        }
    }
}
