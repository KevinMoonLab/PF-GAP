package distance.elastic;

import core.contracts.ObjectDataset;
import transformation.DerivativeTransform;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Weighted Derivative Dynamic Time Warping.
 *
 * <p>Both inputs are transformed with the unified
 * {@link DerivativeTransform}, then evaluated by the optimized weighted,
 * squared-cost {@link WDTW} kernel. Every overload returns accumulated
 * weighted squared derivative cost, and a finite {@code bestSoFar} is
 * interpreted in the same units.</p>
 *
 * <p>Both {@code double[]} and {@code float[]} inputs are supported. Float
 * derivatives are formed in double precision. The class contains no mutable
 * scratch state and is safe for concurrent use.</p>
 */
public final class WDDTW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final WDTW wdtw;

    public WDDTW() {
        wdtw = new WDTW();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g
    ) {
        if (first instanceof double[] firstValues
                && second instanceof double[] secondValues) {
            return distance(firstValues, secondValues, bestSoFar, g);
        }

        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return distance(firstValues, secondValues, bestSoFar, g);
        }

        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            double g
    ) {
        return distance(
                first,
                second,
                Double.POSITIVE_INFINITY,
                g
        );
    }

    private double distance(
            double[] first,
            double[] second,
            double bestSoFar,
            double g
    ) {
        double[] firstDerivative = DerivativeTransform.transform(first);
        double[] secondDerivative = DerivativeTransform.transform(second);

        return wdtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                g
        );
    }

    private double distance(
            float[] first,
            float[] second,
            double bestSoFar,
            double g
    ) {
        double[] firstDerivative = DerivativeTransform.transform(first);
        double[] secondDerivative = DerivativeTransform.transform(second);

        return wdtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                g
        );
    }

    public double get_random_g(ObjectDataset dataset, Random random) {
        return wdtw.get_random_g(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "WDDTW requires matching double[] or float[] inputs. Received "
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
}
