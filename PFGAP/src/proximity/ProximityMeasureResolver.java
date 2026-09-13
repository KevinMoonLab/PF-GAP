package proximity;

import proximity.proximities.BreimanProximity;
import proximity.proximities.DepthWeightedProximity;
import proximity.proximities.PFGAPProximity;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves configured proximity types to stateless measure implementations.
 *
 * <p>This class centralizes built-in registration so callers do not duplicate
 * switches across scalar, dense-matrix, sparse-matrix, training, and
 * test/train workflows. Every registered implementation is validated once
 * when the resolver class is initialized.</p>
 *
 * <p>To add a built-in proximity type:</p>
 * <ol>
 *     <li>Add the identifier to {@link ProximityType}.</li>
 *     <li>Implement {@link ProximityMeasure} as a stateless singleton.</li>
 *     <li>Declare exact properties and domain-specific data requirements.</li>
 *     <li>Add one entry to {@link #createBuiltInMeasures()}.</li>
 *     <li>Add scalar, row-accumulation, dense/sparse, and worker-count tests.</li>
 * </ol>
 *
 * <p>Runtime plugin registration is intentionally not supported by this
 * initial resolver. If custom proximity measures are introduced later, they
 * should use a separate immutable registry assembled during application
 * configuration rather than mutating this global built-in map.</p>
 */
public final class ProximityMeasureResolver {

    private static final Map<ProximityType, ProximityMeasure> BUILT_IN_MEASURES =
            createBuiltInMeasures();

    private ProximityMeasureResolver() {
    }

    /**
     * Resolves one non-null built-in proximity type.
     *
     * @param type configured proximity type
     * @return validated stateless implementation
     * @throws NullPointerException when {@code type} is null
     * @throws IllegalArgumentException when no implementation is registered
     */
    public static ProximityMeasure resolve(
            ProximityType type
    ) {
        Objects.requireNonNull(
                type,
                "ProximityType cannot be null."
        );

        ProximityMeasure measure =
                BUILT_IN_MEASURES.get(
                        type
                );

        if (measure == null) {
            throw new IllegalArgumentException(
                    "No proximity measure is registered for type "
                            + type
                            + "."
            );
        }

        return measure;
    }

    /**
     * Returns whether a built-in implementation is registered for a type.
     */
    public static boolean supports(
            ProximityType type
    ) {
        return type != null
                && BUILT_IN_MEASURES.containsKey(type);
    }

    /**
     * Returns an immutable view of all built-in registrations.
     *
     * <p>This method is intended for diagnostics, documentation, and tests.</p>
     */
    public static Map<ProximityType, ProximityMeasure> builtInMeasures() {
        return BUILT_IN_MEASURES;
    }

    private static Map<ProximityType, ProximityMeasure>
    createBuiltInMeasures() {
        EnumMap<ProximityType, ProximityMeasure> measures =
                new EnumMap<>(
                        ProximityType.class
                );

        register(
                measures,
                ProximityType.PFGAP,
                PFGAPProximity.INSTANCE
        );

        register(
                measures,
                ProximityType.BREIMAN,
                BreimanProximity.INSTANCE
        );

        register(
                measures,
                ProximityType.DEPTH_WEIGHTED,
                DepthWeightedProximity.INSTANCE
        );

        if (measures.size() != ProximityType.values().length) {
            throw new ExceptionInInitializerError(
                    "Built-in proximity registration is incomplete. Registered "
                            + measures.size()
                            + " of "
                            + ProximityType.values().length
                            + " ProximityType values."
            );
        }

        return Map.copyOf(
                measures
        );
    }

    private static void register(
            EnumMap<ProximityType, ProximityMeasure> measures,
            ProximityType type,
            ProximityMeasure measure
    ) {
        Objects.requireNonNull(
                type,
                "Registered ProximityType cannot be null."
        );

        Objects.requireNonNull(
                measure,
                "Registered ProximityMeasure cannot be null."
        );

        measure.validateContract();

        ProximityMeasure prior =
                measures.putIfAbsent(
                        type,
                        measure
                );

        if (prior != null) {
            throw new ExceptionInInitializerError(
                    "Duplicate proximity registration for "
                            + type
                            + ": "
                            + prior.id()
                            + " and "
                            + measure.id()
                            + "."
            );
        }
    }
}
