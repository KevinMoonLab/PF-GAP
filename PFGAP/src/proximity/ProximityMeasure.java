package proximity;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Mathematical contract for one proximity measure.
 *
 * <p>Implementations provide scalar train/train and test/train calculations as
 * universal fallbacks. Measures that can construct a row by visiting only
 * possible nonzero targets may additionally implement the row-accumulation
 * methods. This allows the sparse matrix engine to avoid an O(N^2) scan when
 * the formula admits a leaf-oriented, tree-oriented, or source-row algorithm.</p>
 *
 * <p>Train/train and test/train calculations declare requirements separately.
 * This distinction prevents a train-only calculation from allocating test
 * indexes and prevents a test/train calculation from preparing train-only data
 * it does not need. The distinction is especially important for very large
 * datasets.</p>
 *
 * <p>Matrix allocation, task scheduling, sparse thresholding, symmetry
 * optimization, diagonal policy, and publication to application state belong
 * to the matrix-computation layer. Implementations must be stateless, safe for
 * concurrent invocation, and must not create executors or mutate the forest,
 * context, or shared application state.</p>
 */
public interface ProximityMeasure {

    /**
     * Computes one directed train/train proximity.
     */
    double computeTrainTrain(
            int sourceTrainIndex,
            int targetTrainIndex,
            ProximityContext context
    );

    /**
     * Computes one directed test/train proximity.
     */
    double computeTestTrain(
            int testIndex,
            int trainIndex,
            ProximityContext context
    );

    /**
     * Returns whether this measure can accumulate a complete train/train row
     * without requiring the matrix engine to visit every target column.
     */
    default boolean supportsTrainRowAccumulation() {
        return false;
    }

    /**
     * Accumulates one final, normalized train/train row.
     *
     * <p>The accumulator is empty on entry and belongs exclusively to the
     * calling task. The implementation must preserve its declared diagonal
     * semantics and must not apply the matrix output's sparsity threshold.</p>
     */
    default void accumulateTrainRow(
            int sourceTrainIndex,
            ProximityContext context,
            ProximityRowAccumulator accumulator
    ) {
        throw new UnsupportedOperationException(
                "Proximity measure "
                        + id()
                        + " does not support direct train-row accumulation."
        );
    }

    /**
     * Returns whether this measure can accumulate a complete test/train row
     * without requiring the matrix engine to visit every training column.
     */
    default boolean supportsTestTrainRowAccumulation() {
        return false;
    }

    /**
     * Accumulates one final, normalized test/train row.
     *
     * <p>The accumulator is empty on entry and belongs exclusively to the
     * calling task. The implementation must not apply the matrix output's
     * sparsity threshold.</p>
     */
    default void accumulateTestTrainRow(
            int testIndex,
            ProximityContext context,
            ProximityRowAccumulator accumulator
    ) {
        throw new UnsupportedOperationException(
                "Proximity measure "
                        + id()
                        + " does not support direct test/train-row accumulation."
        );
    }

    /**
     * Returns mathematical guarantees that the matrix engine may exploit.
     */
    ProximityProperties properties();

    /**
     * Returns the minimal immutable requirements for train/train calculation.
     *
     * <p>The set must include data needed by both scalar train/train evaluation
     * and direct train-row accumulation.</p>
     */
    Set<ProximityDataRequirement> trainRequirements();

    /**
     * Returns the minimal immutable requirements for test/train calculation.
     *
     * <p>The set must include data needed by both scalar test/train evaluation
     * and direct test/train-row accumulation.</p>
     */
    Set<ProximityDataRequirement> testTrainRequirements();

    /**
     * Returns the immutable union of train/train and test/train requirements.
     *
     * <p>This convenience method is intended for diagnostics and complete
     * precomputation. Performance-sensitive builders should call the
     * domain-specific requirement method instead.</p>
     */
    default Set<ProximityDataRequirement> allRequirements() {
        Set<ProximityDataRequirement> train =
                trainRequirements();

        Set<ProximityDataRequirement> testTrain =
                testTrainRequirements();

        if (train == null || testTrain == null) {
            throw new IllegalStateException(
                    "Proximity measure "
                            + id()
                            + " returned null requirements."
            );
        }

        if (train.isEmpty() && testTrain.isEmpty()) {
            return Set.of();
        }

        EnumSet<ProximityDataRequirement> combined =
                EnumSet.noneOf(
                        ProximityDataRequirement.class
                );

        combined.addAll(train);
        combined.addAll(testTrain);

        return Collections.unmodifiableSet(
                combined
        );
    }

    /**
     * Returns a stable implementation identifier for diagnostics and result
     * metadata.
     */
    default String id() {
        return getClass().getSimpleName();
    }

    /**
     * Validates implementation metadata and domain-specific requirements.
     *
     * <p>The resolver or context builder should call this once per selected
     * measure, not once per matrix entry.</p>
     */
    default void validateContract() {
        String identifier =
                id();

        if (identifier == null || identifier.isBlank()) {
            throw new IllegalStateException(
                    "A proximity measure must declare a non-blank identifier."
            );
        }

        if (properties() == null) {
            throw new IllegalStateException(
                    "Proximity measure "
                            + identifier
                            + " returned null properties."
            );
        }

        validateRequirements(
                identifier,
                "train/train",
                trainRequirements()
        );

        validateRequirements(
                identifier,
                "test/train",
                testTrainRequirements()
        );
    }

    /**
     * Validates one domain-specific requirement set.
     */
    private static void validateRequirements(
            String identifier,
            String domain,
            Set<ProximityDataRequirement> requirements
    ) {
        if (requirements == null) {
            throw new IllegalStateException(
                    "Proximity measure "
                            + identifier
                            + " returned null "
                            + domain
                            + " requirements."
            );
        }

        for (ProximityDataRequirement requirement : requirements) {
            if (requirement == null) {
                throw new IllegalStateException(
                        "Proximity measure "
                                + identifier
                                + " declared a null "
                                + domain
                                + " data requirement."
                );
            }
        }
    }
}
