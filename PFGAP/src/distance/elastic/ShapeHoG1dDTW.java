package distance.elastic;

import core.contracts.ObjectDataset;
import transformation.HistogramOfGradients;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * DTW between one-dimensional histogram-of-gradient representations.
 *
 * <p>{@link HistogramOfGradients} owns the single derivative transformation;
 * this class does not differentiate the input beforehand. The resulting
 * histogram vectors are compared with squared-cost DTW.</p>
 */
public final class ShapeHoG1dDTW implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final DTW dtw;

    public ShapeHoG1dDTW() {
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
            return dtw.distance(
                    HistogramOfGradients.computeHistogram(firstValues),
                    HistogramOfGradients.computeHistogram(secondValues),
                    bestSoFar,
                    windowSize
            );
        }
        if (first instanceof float[] firstValues
                && second instanceof float[] secondValues) {
            return dtw.distance(
                    HistogramOfGradients.computeHistogram(firstValues),
                    HistogramOfGradients.computeHistogram(secondValues),
                    bestSoFar,
                    windowSize
            );
        }
        throw unsupportedPair(first, second);
    }

    public int get_random_window(ObjectDataset dataset, Random random) {
        return dtw.get_random_window(dataset, random);
    }

    private static IllegalArgumentException unsupportedPair(
            Object first,
            Object second
    ) {
        return new IllegalArgumentException(
                "ShapeHoG1dDTW requires matching double[] or float[] inputs. "
                        + "Received " + typeName(first) + " and "
                        + typeName(second) + "."
        );
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getTypeName();
    }
}
