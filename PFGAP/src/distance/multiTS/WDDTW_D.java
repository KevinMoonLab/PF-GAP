package distance.multiTS;

import transformation.DerivativeTransform;

import java.io.Serial;
import java.io.Serializable;

/**
 * Dependent multivariate Weighted Derivative Dynamic Time Warping.
 *
 * <p>Inputs are transformed with the unified derivative convention, after
 * which one shared dependent WDTW alignment is evaluated across all selected
 * channels. Every overload returns accumulated weighted squared cost, and a
 * finite {@code bestSoFar} is interpreted in the same units.</p>
 *
 * <p>When dimensions are selected, only those channels are transformed into
 * compact derivative matrices. This avoids transforming unselected channels
 * and avoids selected-index lookup inside every dynamic-programming cell.</p>
 */
public final class WDDTW_D implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final WDTW_D wdtw;

    public WDDTW_D() {
        wdtw = new WDTW_D();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceAll(firstValues, secondValues, bestSoFar, g);
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceAll(firstValues, secondValues, bestSoFar, g);
        }
        throw unsupportedPair(first, second);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            double g,
            int[] selectedDimensions
    ) {
        if (selectedDimensions == null) {
            return distance(first, second, bestSoFar, g);
        }
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            return distanceSelected(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    g,
                    selectedDimensions
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            return distanceSelected(
                    firstValues,
                    secondValues,
                    bestSoFar,
                    g,
                    selectedDimensions
            );
        }
        throw unsupportedPair(first, second);
    }

    private double distanceAll(
            double[][] first,
            double[][] second,
            double bestSoFar,
            double g
    ) {
        double[][] firstDerivative = DerivativeTransform.transform(first);
        double[][] secondDerivative = DerivativeTransform.transform(second);
        return wdtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                g
        );
    }

    private double distanceAll(
            float[][] first,
            float[][] second,
            double bestSoFar,
            double g
    ) {
        double[][] firstDerivative = DerivativeTransform.transform(first);
        double[][] secondDerivative = DerivativeTransform.transform(second);
        return wdtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                g
        );
    }

    private double distanceSelected(
            double[][] first,
            double[][] second,
            double bestSoFar,
            double g,
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
        return wdtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                g
        );
    }

    private double distanceSelected(
            float[][] first,
            float[][] second,
            double bestSoFar,
            double g,
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
        return wdtw.distance(
                firstDerivative,
                secondDerivative,
                bestSoFar,
                g
        );
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "Dependent WDDTW requires matching double[][] or float[][] "
                        + "inputs. Received "
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
}
