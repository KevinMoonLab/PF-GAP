package distance.missing;

import core.AppContext;

import java.io.Serial;
import java.io.Serializable;

/**
 * Missing-value-aware Euclidean distance for primitive univariate series.
 *
 * <p>The squared distance is</p>
 *
 * <pre>
 * (fullLength / jointlyObservedCount)
 *         * sum((first[i] - second[i])^2)
 * </pre>
 *
 * <p>where the sum includes only positions observed in both series. If no
 * position is jointly observed, the distance is unavailable and positive
 * infinity is returned. The ordinary distance is the square root of this
 * scaled squared distance.</p>
 *
 * <p>Matching {@code double[]} and matching {@code float[]} inputs are
 * supported. Missing values are represented by NaN. Float values are widened
 * individually in the scalar path. When {@code AppContext.useVectorApi} is
 * enabled, a masked Vector API reduction processes contiguous primitive data
 * without constructing filtered or converted arrays.</p>
 *
 * <p>The final scale depends on the final jointly observed count. Consequently,
 * a one-pass exact implementation cannot safely abandon from an unscaled
 * partial sum. A finite {@code bestSoFar} is therefore applied after the exact
 * scaled result is available.</p>
 */
public final class NaNEuclidean implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    public NaNEuclidean() {
    }

    public double distance(Object first, Object second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar
    ) {
        validateBestSoFar(bestSoFar);
        double squared = squaredDistance(first, second);
        if (Double.isInfinite(squared)) {
            return Double.POSITIVE_INFINITY;
        }

        double result = Math.sqrt(squared);
        return result > bestSoFar
                ? Double.POSITIVE_INFINITY
                : result;
    }

    public double squaredDistance(Object first, Object second) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            MissingDistanceTools.validateSameLength(
                    firstValues,
                    secondValues
            );
            return AppContext.useVectorApi
                    ? VectorKernels.squaredDistance(
                            firstValues,
                            secondValues
                    )
                    : scalarSquaredDistance(
                            firstValues,
                            secondValues
                    );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            MissingDistanceTools.validateSameLength(
                    firstValues,
                    secondValues
            );
            return AppContext.useVectorApi
                    ? VectorKernels.squaredDistance(
                            firstValues,
                            secondValues
                    )
                    : scalarSquaredDistance(
                            firstValues,
                            secondValues
                    );
        }

        throw unsupportedPair(first, second);
    }

    public boolean isComputable(Object first, Object second) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            MissingDistanceTools.validateSameLength(
                    firstValues,
                    secondValues
            );
            return MissingDistanceTools.countJointlyObserved(
                    firstValues,
                    secondValues
            ) > 0;
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            MissingDistanceTools.validateSameLength(
                    firstValues,
                    secondValues
            );
            return MissingDistanceTools.countJointlyObserved(
                    firstValues,
                    secondValues
            ) > 0;
        }

        throw unsupportedPair(first, second);
    }

    private static double scalarSquaredDistance(
            double[] first,
            double[] second
    ) {
        double squaredSum = 0.0;
        int jointlyObserved = 0;

        for (int index = 0; index < first.length; index++) {
            double firstValue = first[index];
            double secondValue = second[index];
            if (Double.isNaN(firstValue)
                    || Double.isNaN(secondValue)) {
                continue;
            }

            double difference = firstValue - secondValue;
            squaredSum += difference * difference;
            jointlyObserved++;
        }

        return scale(first.length, jointlyObserved, squaredSum);
    }

    private static double scalarSquaredDistance(
            float[] first,
            float[] second
    ) {
        double squaredSum = 0.0;
        int jointlyObserved = 0;

        for (int index = 0; index < first.length; index++) {
            float firstValue = first[index];
            float secondValue = second[index];
            if (Float.isNaN(firstValue)
                    || Float.isNaN(secondValue)) {
                continue;
            }

            double difference =
                    (double) firstValue - (double) secondValue;
            squaredSum += difference * difference;
            jointlyObserved++;
        }

        return scale(first.length, jointlyObserved, squaredSum);
    }

    private static double scale(
            int fullLength,
            int jointlyObserved,
            double squaredSum
    ) {
        if (jointlyObserved == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return (double) fullLength / jointlyObserved * squaredSum;
    }

    private static void validateBestSoFar(double bestSoFar) {
        if (Double.isNaN(bestSoFar) || bestSoFar < 0.0) {
            throw new IllegalArgumentException(
                    "NaNEuclidean bestSoFar must be nonnegative and not NaN. "
                            + "Received: " + bestSoFar + "."
            );
        }
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "NaNEuclidean requires matching double[] or float[] inputs. "
                        + "Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }

    /** Isolates incubator-module references in a lazily loaded class. */
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
                double[] second
        ) {
            int upperBound = DOUBLE_SPECIES.loopBound(first.length);
            int index = 0;
            int jointlyObserved = 0;
            jdk.incubator.vector.DoubleVector accumulated =
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
                jdk.incubator.vector.VectorMask<Double> observed =
                        firstVector.test(
                                jdk.incubator.vector.VectorOperators.IS_NAN
                        ).or(
                                secondVector.test(
                                        jdk.incubator.vector.VectorOperators.IS_NAN
                                )
                        ).not();
                jdk.incubator.vector.DoubleVector difference =
                        firstVector.sub(secondVector);
                accumulated = accumulated.add(
                        difference.mul(difference),
                        observed
                );
                jointlyObserved += observed.trueCount();
            }

            double squaredSum = accumulated.reduceLanes(
                    jdk.incubator.vector.VectorOperators.ADD
            );
            for (; index < first.length; index++) {
                double firstValue = first[index];
                double secondValue = second[index];
                if (Double.isNaN(firstValue)
                        || Double.isNaN(secondValue)) {
                    continue;
                }
                double difference = firstValue - secondValue;
                squaredSum += difference * difference;
                jointlyObserved++;
            }

            return scale(first.length, jointlyObserved, squaredSum);
        }

        private static double squaredDistance(
                float[] first,
                float[] second
        ) {
            int upperBound = FLOAT_SPECIES.loopBound(first.length);
            int index = 0;
            int jointlyObserved = 0;
            double squaredSum = 0.0;

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
                jdk.incubator.vector.VectorMask<Float> observed =
                        firstVector.test(
                                jdk.incubator.vector.VectorOperators.IS_NAN
                        ).or(
                                secondVector.test(
                                        jdk.incubator.vector.VectorOperators.IS_NAN
                                )
                        ).not();
                jdk.incubator.vector.FloatVector difference =
                        firstVector.sub(secondVector);
                squaredSum += difference.mul(difference).reduceLanes(
                        jdk.incubator.vector.VectorOperators.ADD,
                        observed
                );
                jointlyObserved += observed.trueCount();
            }

            for (; index < first.length; index++) {
                float firstValue = first[index];
                float secondValue = second[index];
                if (Float.isNaN(firstValue)
                        || Float.isNaN(secondValue)) {
                    continue;
                }
                double difference =
                        (double) firstValue - (double) secondValue;
                squaredSum += difference * difference;
                jointlyObserved++;
            }

            return scale(first.length, jointlyObserved, squaredSum);
        }
    }
}
