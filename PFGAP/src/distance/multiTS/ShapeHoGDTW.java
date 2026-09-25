package distance.multiTS;

import distance.elastic.DTW;
import transformation.DerivativeTransform;
import transformation.MultivariateHistogramOfGradients;
import transformation.MultivariateHistogramOfGradients.Strategy;

import java.io.Serial;
import java.io.Serializable;

/**
 * DTW over a multivariate histogram-of-gradient representation.
 *
 * <p>The unified derivative transform is applied exactly once. Selected
 * channels are compacted during derivative transformation before the
 * multivariate histogram is built.</p>
 */
public final class ShapeHoGDTW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DTW dtw;

    public ShapeHoGDTW() {
        dtw = new DTW();
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize
    ) {
        return distance(first, second, bestSoFar, windowSize, null);
    }

    public double distance(
            Object first,
            Object second,
            double bestSoFar,
            int windowSize,
            int[] selectedDimensions
    ) {
        if (first instanceof double[][] firstValues
                && second instanceof double[][] secondValues) {
            double[][] firstDerivative = selectedDimensions == null
                    ? DerivativeTransform.transform(firstValues)
                    : DerivativeTransform.transformSelected(
                            firstValues, selectedDimensions
                    );
            double[][] secondDerivative = selectedDimensions == null
                    ? DerivativeTransform.transform(secondValues)
                    : DerivativeTransform.transformSelected(
                            secondValues, selectedDimensions
                    );
            return histogramDistance(
                    firstDerivative, secondDerivative,
                    bestSoFar, windowSize
            );
        }
        if (first instanceof float[][] firstValues
                && second instanceof float[][] secondValues) {
            double[][] firstDerivative = selectedDimensions == null
                    ? DerivativeTransform.transform(firstValues)
                    : DerivativeTransform.transformSelected(
                            firstValues, selectedDimensions
                    );
            double[][] secondDerivative = selectedDimensions == null
                    ? DerivativeTransform.transform(secondValues)
                    : DerivativeTransform.transformSelected(
                            secondValues, selectedDimensions
                    );
            return histogramDistance(
                    firstDerivative, secondDerivative,
                    bestSoFar, windowSize
            );
        }
        throw unsupportedPair(first, second);
    }

    private double histogramDistance(
            double[][] firstDerivative,
            double[][] secondDerivative,
            double bestSoFar,
            int windowSize
    ) {
        double[] firstHistogram =
                MultivariateHistogramOfGradients.computeHistogram(
                        firstDerivative,
                        Strategy.CONCATENATE_GRADIENTS
                );
        double[] secondHistogram =
                MultivariateHistogramOfGradients.computeHistogram(
                        secondDerivative,
                        Strategy.CONCATENATE_GRADIENTS
                );
        return dtw.distance(
                firstHistogram,
                secondHistogram,
                bestSoFar,
                windowSize
        );
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "ShapeHoGDTW requires matching double[][] or float[][] inputs. "
                        + "Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
