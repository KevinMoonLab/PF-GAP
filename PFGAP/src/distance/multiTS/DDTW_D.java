package distance.multiTS;

import transformation.DerivativeTransform;

import java.io.Serial;
import java.io.Serializable;

/**
 * Dependent multivariate Derivative Dynamic Time Warping.
 *
 * <p>Inputs are transformed with the unified derivative convention, after
 * which one shared dependent-DTW alignment is evaluated across all selected
 * channels. Every overload returns accumulated squared cost, and a finite
 * {@code bestSoFar} is interpreted in the same units.</p>
 *
 * <p>When dimensions are selected, only those channels are transformed into a
 * compact derivative matrix. This is deliberate: derivative storage is needed
 * throughout every dynamic-programming cell, and compacting selected channels
 * avoids both transforming unselected channels and repeatedly consulting the
 * selection array inside the dependent local-cost loop.</p>
 *
 * <p>Both {@code double[][]} and {@code float[][]} dimension-major inputs are
 * supported. Derivatives are stored as {@code double[][]}. The class contains
 * no mutable scratch state and is safe for concurrent use.</p>
 */
public final class DDTW_D implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DTW_D dtw;

    public DDTW_D() {
        dtw = new DTW_D();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceAll(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize
            );
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceAll(
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
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        if (selectedDimensions == null) {
            return distance(first, second, bestSoFar, windowSize);
        }

        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceSelected(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }

        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceSelected(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    windowSize,
                    selectedDimensions
            );
        }

        throw unsupportedPair(first, second);
    }

    private double distanceAll(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize
    ) {
        double[][] firstDerivative = DerivativeTransform.transform(first);
        double[][] secondDerivative = DerivativeTransform.transform(second);

        return dtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                windowSize
        );
    }

    private double distanceAll(
            float[][] first,
            float[][] second,
            double bestSoFar,
            int windowSize
    ) {
        double[][] firstDerivative = DerivativeTransform.transform(first);
        double[][] secondDerivative = DerivativeTransform.transform(second);

        return dtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                windowSize
        );
    }

    private double distanceSelected(
            double[][] first,
            double[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        double[][] firstDerivative =
                DerivativeTransform.transformSelected(
                        first,
                        selectedDimensions
                );
        double[][] secondDerivative =
                DerivativeTransform.transformSelected(
                        second,
                        selectedDimensions
                );

        return dtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                windowSize
        );
    }

    private double distanceSelected(
            float[][] first,
            float[][] second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        double[][] firstDerivative =
                DerivativeTransform.transformSelected(
                        first,
                        selectedDimensions
                );
        double[][] secondDerivative =
                DerivativeTransform.transformSelected(
                        second,
                        selectedDimensions
                );

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
                "Dependent DDTW requires matching double[][] or float[][] "
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
}
