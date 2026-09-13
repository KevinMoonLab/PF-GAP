package proximity;

import java.util.OptionalDouble;

/**
 * Immutable mathematical and computational properties of a proximity measure.
 *
 * <p>The proximity-matrix engine uses these properties only when an
 * implementation explicitly declares them. It must not infer symmetry,
 * self-similarity, or diagonal values from a proximity name.</p>
 *
 * <p>Train/train symmetry means that, for every valid pair of training
 * indices {@code i} and {@code j}, the measure guarantees:</p>
 *
 * <pre>
 * proximity(i, j) == proximity(j, i)
 * </pre>
 *
 * <p>A constant train diagonal means that every valid train/train
 * self-proximity has one implementation-declared value. When no constant is
 * declared, the matrix engine must compute every diagonal entry normally.</p>
 *
 * <p>This metadata describes guarantees of the implemented formula, including
 * its treatment of bootstrap membership, out-of-bag membership, missing leaf
 * assignments, and normalization. A property must not be declared merely
 * because it is commonly associated with a similarly named proximity.</p>
 *
 * @param trainSymmetric whether train/train proximity is guaranteed symmetric
 * @param constantTrainDiagonal optional constant value for every train/train
 *                              diagonal entry
 */
public record ProximityProperties(
        boolean trainSymmetric,
        OptionalDouble constantTrainDiagonal
) {

    /**
     * Creates and validates a property descriptor.
     */
    public ProximityProperties {
        if (constantTrainDiagonal == null) {
            throw new IllegalArgumentException(
                    "constantTrainDiagonal cannot be null. "
                            + "Use OptionalDouble.empty() when the diagonal "
                            + "must be computed."
            );
        }

        if (constantTrainDiagonal.isPresent()) {
            double value =
                    constantTrainDiagonal.getAsDouble();

            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                        "A constant train diagonal must be finite. Received: "
                                + value
                                + "."
                );
            }
        }
    }

    /**
     * Creates properties for a measure whose train/train matrix is not
     * guaranteed symmetric and whose diagonal must be computed.
     */
    public static ProximityProperties general() {
        return new ProximityProperties(
                false,
                OptionalDouble.empty()
        );
    }

    /**
     * Creates properties for a symmetric train/train measure whose diagonal
     * must still be computed normally.
     */
    public static ProximityProperties symmetric() {
        return new ProximityProperties(
                true,
                OptionalDouble.empty()
        );
    }

    /**
     * Creates properties for a measure with a known constant train diagonal.
     *
     * @param trainSymmetric whether train/train proximity is symmetric
     * @param diagonalValue constant finite self-proximity
     */
    public static ProximityProperties withConstantTrainDiagonal(
            boolean trainSymmetric,
            double diagonalValue
    ) {
        return new ProximityProperties(
                trainSymmetric,
                OptionalDouble.of(diagonalValue)
        );
    }

    /**
     * Returns whether the matrix engine may fill train/train diagonal entries
     * without invoking the measure.
     */
    public boolean hasConstantTrainDiagonal() {
        return constantTrainDiagonal.isPresent();
    }

    /**
     * Returns the declared constant train diagonal.
     *
     * @throws IllegalStateException when the diagonal must be computed
     */
    public double requireConstantTrainDiagonal() {
        if (constantTrainDiagonal.isEmpty()) {
            throw new IllegalStateException(
                    "This proximity measure does not declare a constant "
                            + "train diagonal."
            );
        }

        return constantTrainDiagonal.getAsDouble();
    }
}
