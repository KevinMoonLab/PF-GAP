package distance.elastic;

import core.AppContext;

import java.io.Serial;
import java.io.Serializable;

/**
 * Cosine distance kernels for primitive double and float vectors.
 *
 * <p>Cosine distance is defined as {@code 1.0 - cosineSimilarity}. The
 * {@code bestSoFar} argument is accepted for compatibility with the common
 * distance dispatch contract, but is intentionally not used for early
 * abandoning. Partial dot products and cosine norms do not provide a generally
 * valid monotone lower bound on the final cosine distance.</p>
 *
 * <p>When {@code AppContext.useVectorApi} is true, contiguous all-feature
 * calculations use the JDK Vector API. Selected-feature calculations remain
 * scalar because they use indirect indexed access. Equal vector lengths and
 * valid selected indices are trusted preconditions established by the dataset
 * and splitter layers.</p>
 *
 * <p>Two zero vectors have distance {@code 0.0}. A zero vector and a nonzero
 * vector have distance {@code 1.0}. This keeps the distance finite for routing
 * and treats identical zero vectors as identical observations.</p>
 */
public final class Cosine implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public Cosine() {
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return AppContext.useVectorApi
                    ? VectorKernels.distance(firstValues, secondValues)
                    : scalarDistance(firstValues, secondValues);
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return AppContext.useVectorApi
                    ? VectorKernels.distance(firstValues, secondValues)
                    : scalarDistance(firstValues, secondValues);
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
                    selectedDimensions
            );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return scalarDistance(
                    firstValues,
                    secondValues,
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
            double[] second
    ) {
        double dotProduct = 0.0;
        double firstSquaredNorm = 0.0;
        double secondSquaredNorm = 0.0;

        for (int index = 0; index < first.length; index++) {
            double firstValue = first[index];
            double secondValue = second[index];
            dotProduct += firstValue * secondValue;
            firstSquaredNorm += firstValue * firstValue;
            secondSquaredNorm += secondValue * secondValue;
        }

        return finishDistance(
                dotProduct,
                firstSquaredNorm,
                secondSquaredNorm
        );
    }

    private static double scalarDistance(
            float[] first,
            float[] second
    ) {
        double dotProduct = 0.0;
        double firstSquaredNorm = 0.0;
        double secondSquaredNorm = 0.0;

        for (int index = 0; index < first.length; index++) {
            double firstValue = first[index];
            double secondValue = second[index];
            dotProduct += firstValue * secondValue;
            firstSquaredNorm += firstValue * firstValue;
            secondSquaredNorm += secondValue * secondValue;
        }

        return finishDistance(
                dotProduct,
                firstSquaredNorm,
                secondSquaredNorm
        );
    }

    private static double scalarDistance(
            double[] first,
            double[] second,
            int[] selectedDimensions
    ) {
        double dotProduct = 0.0;
        double firstSquaredNorm = 0.0;
        double secondSquaredNorm = 0.0;

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int feature = selectedDimensions[position];
            double firstValue = first[feature];
            double secondValue = second[feature];
            dotProduct += firstValue * secondValue;
            firstSquaredNorm += firstValue * firstValue;
            secondSquaredNorm += secondValue * secondValue;
        }

        return finishDistance(
                dotProduct,
                firstSquaredNorm,
                secondSquaredNorm
        );
    }

    private static double scalarDistance(
            float[] first,
            float[] second,
            int[] selectedDimensions
    ) {
        double dotProduct = 0.0;
        double firstSquaredNorm = 0.0;
        double secondSquaredNorm = 0.0;

        for (int position = 0;
             position < selectedDimensions.length;
             position++) {
            int feature = selectedDimensions[position];
            double firstValue = first[feature];
            double secondValue = second[feature];
            dotProduct += firstValue * secondValue;
            firstSquaredNorm += firstValue * firstValue;
            secondSquaredNorm += secondValue * secondValue;
        }

        return finishDistance(
                dotProduct,
                firstSquaredNorm,
                secondSquaredNorm
        );
    }

    private static double finishDistance(
            double dotProduct,
            double firstSquaredNorm,
            double secondSquaredNorm
    ) {
        if (firstSquaredNorm == 0.0) {
            return secondSquaredNorm == 0.0 ? 0.0 : 1.0;
        }
        if (secondSquaredNorm == 0.0) {
            return 1.0;
        }

        double similarity = dotProduct
                / Math.sqrt(firstSquaredNorm * secondSquaredNorm);

        // Protect the theoretical [-1, 1] range from roundoff.
        if (similarity > 1.0) {
            similarity = 1.0;
        } else if (similarity < -1.0) {
            similarity = -1.0;
        }

        return 1.0 - similarity;
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Cosine distance requires matching double[] or float[] "
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

    /** Isolates incubator-module references in a lazily initialized class. */
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
                double[] second
        ) {
            int upperBound = DOUBLE_SPECIES.loopBound(first.length);
            int index = 0;

            jdk.incubator.vector.DoubleVector dotProduct =
                    jdk.incubator.vector.DoubleVector.zero(DOUBLE_SPECIES);
            jdk.incubator.vector.DoubleVector firstSquaredNorm =
                    jdk.incubator.vector.DoubleVector.zero(DOUBLE_SPECIES);
            jdk.incubator.vector.DoubleVector secondSquaredNorm =
                    jdk.incubator.vector.DoubleVector.zero(DOUBLE_SPECIES);

            for (; index < upperBound;
                 index += DOUBLE_SPECIES.length()) {
                jdk.incubator.vector.DoubleVector firstVector =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES,
                                first,
                                index
                        );
                jdk.incubator.vector.DoubleVector secondVector =
                        jdk.incubator.vector.DoubleVector.fromArray(
                                DOUBLE_SPECIES,
                                second,
                                index
                        );

                dotProduct = dotProduct.add(
                        firstVector.mul(secondVector)
                );
                firstSquaredNorm = firstSquaredNorm.add(
                        firstVector.mul(firstVector)
                );
                secondSquaredNorm = secondSquaredNorm.add(
                        secondVector.mul(secondVector)
                );
            }

            double dot = dotProduct.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );
            double firstNorm = firstSquaredNorm.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );
            double secondNorm = secondSquaredNorm.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );

            for (; index < first.length; index++) {
                double firstValue = first[index];
                double secondValue = second[index];
                dot += firstValue * secondValue;
                firstNorm += firstValue * firstValue;
                secondNorm += secondValue * secondValue;
            }

            return finishDistance(dot, firstNorm, secondNorm);
        }

        private static double distance(
                float[] first,
                float[] second
        ) {
            int upperBound = FLOAT_SPECIES.loopBound(first.length);
            int index = 0;

            jdk.incubator.vector.FloatVector dotProduct =
                    jdk.incubator.vector.FloatVector.zero(FLOAT_SPECIES);
            jdk.incubator.vector.FloatVector firstSquaredNorm =
                    jdk.incubator.vector.FloatVector.zero(FLOAT_SPECIES);
            jdk.incubator.vector.FloatVector secondSquaredNorm =
                    jdk.incubator.vector.FloatVector.zero(FLOAT_SPECIES);

            for (; index < upperBound;
                 index += FLOAT_SPECIES.length()) {
                jdk.incubator.vector.FloatVector firstVector =
                        jdk.incubator.vector.FloatVector.fromArray(
                                FLOAT_SPECIES,
                                first,
                                index
                        );
                jdk.incubator.vector.FloatVector secondVector =
                        jdk.incubator.vector.FloatVector.fromArray(
                                FLOAT_SPECIES,
                                second,
                                index
                        );

                dotProduct = dotProduct.add(
                        firstVector.mul(secondVector)
                );
                firstSquaredNorm = firstSquaredNorm.add(
                        firstVector.mul(firstVector)
                );
                secondSquaredNorm = secondSquaredNorm.add(
                        secondVector.mul(secondVector)
                );
            }

            double dot = dotProduct.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );
            double firstNorm = firstSquaredNorm.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );
            double secondNorm = secondSquaredNorm.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );

            for (; index < first.length; index++) {
                double firstValue = first[index];
                double secondValue = second[index];
                dot += firstValue * secondValue;
                firstNorm += firstValue * firstValue;
                secondNorm += secondValue * secondValue;
            }

            return finishDistance(dot, firstNorm, secondNorm);
        }
    }
}
