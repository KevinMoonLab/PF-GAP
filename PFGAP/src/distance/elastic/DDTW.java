package distance.elastic;

import core.contracts.ObjectDataset;
import transformation.DerivativeTransform;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Derivative Dynamic Time Warping with squared Euclidean local costs.
 *
 * <p>Both inputs are transformed with the unified
 * {@link DerivativeTransform}, then evaluated by the optimized squared-cost
 * {@link DTW} kernel. Every overload returns accumulated squared derivative-DTW
 * cost, and a finite {@code bestSoFar} is interpreted in the same units.</p>
 */
public final class DDTW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DTW dtw;

    public DDTW() {
        dtw = new DTW();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize
            );
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize
            );
        }

        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            int windowSize
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                windowSize
        );
    }

    private double distance(
            double[] first,
            double[] second,
            double bestSoFar,
            int windowSize
    ) {
        double[] firstDerivative = DerivativeTransform.transform(first);
        double[] secondDerivative = DerivativeTransform.transform(second);

        return dtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                windowSize
        );
    }

    private double distance(
            float[] first,
            float[] second,
            double bestSoFar,
            int windowSize
    ) {
        double[] firstDerivative = DerivativeTransform.transform(first);
        double[] secondDerivative = DerivativeTransform.transform(second);

        return dtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                windowSize
        );
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "DDTW requires matching double[] or float[] inputs. Received "
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

    public int get_random_window(ObjectDataset dataset, Random random) {
        return dtw.get_random_window(dataset, random);
    }
}
