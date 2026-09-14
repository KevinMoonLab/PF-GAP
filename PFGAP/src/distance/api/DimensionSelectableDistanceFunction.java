package distance.api;

/**
 * Custom Java distance that can evaluate a selected subset of realized
 * dimensions without requiring PFGAP to copy or slice the input data.
 */
public interface DimensionSelectableDistanceFunction
        extends DistanceFunction {

    /**
     * Computes a distance using only the supplied selected dimensions.
     *
     * @param first materialized first input
     * @param second materialized second input
     * @param selectedDimensions distinct selected dimension indices
     * @return distance value
     */
    double compute(
            Object first,
            Object second,
            int[] selectedDimensions
    );
}