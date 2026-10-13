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

    /**
     * Computes a distance using the supplied selected dimensions and an
     * optional early-abandoning cutoff.
     *
     * <p>The default implementation ignores {@code bestSoFar} and delegates
     * to the existing selected-dimension method. Implementations that support
     * early abandoning may override this method.</p>
     *
     * <p>An implementation may abandon only after proving that the final
     * distance cannot be less than or equal to {@code bestSoFar}. Returning
     * an exact value equal to the cutoff is necessary to preserve ties.</p>
     *
     * @param first materialized first input
     * @param second materialized second input
     * @param bestSoFar current smallest competing distance
     * @param selectedDimensions distinct selected dimension indices
     * @return exact distance when it is at most {@code bestSoFar}; otherwise,
     *         either the exact distance or positive infinity
     */
    default double compute(
            Object first,
            Object second,
            double bestSoFar,
            int[] selectedDimensions
    ) {
        return compute(
                first,
                second,
                selectedDimensions
        );
    }
}