package trees;

import core.AppContext;
import proximity.ProximityDataRequirement;
import proximity.ProximityMeasure;
import proximity.ProximityMeasureResolver;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Immutable fit-time plan describing whether recursive OOB routing is needed.
 *
 * <p>The plan is derived once from requested work. Sampling membership remains
 * independent: a bootstrap or subsample can be constructed without routing its
 * non-sampled complement through every trained node.</p>
 */
public final class OobRoutingPlan {
    private final Set<OobRoutingReason> reasons;

    private OobRoutingPlan(EnumSet<OobRoutingReason> reasons) {
        this.reasons = reasons.isEmpty()
                ? Set.of()
                : Collections.unmodifiableSet(EnumSet.copyOf(reasons));
    }

    /** Returns a plan with no routing requirements. */
    public static OobRoutingPlan none() {
        return new OobRoutingPlan(
                EnumSet.noneOf(OobRoutingReason.class)
        );
    }

    /** Resolves the current training invocation's OOB-routing requirements. */
    public static OobRoutingPlan resolve() {
        EnumSet<OobRoutingReason> reasons =
                EnumSet.noneOf(OobRoutingReason.class);

        if (AppContext.oob_scores) {
            reasons.add(OobRoutingReason.OOB_SCORING);
        }

        ProximityMeasure measure = ProximityMeasureResolver.resolve(
                AppContext.proximityType
        );
        boolean trainProximityNeedsOob = requiresOobRouting(
                measure.trainRequirements()
        );
        boolean testTrainProximityNeedsOob = requiresOobRouting(
                measure.testTrainRequirements()
        );

        if (AppContext.getprox && trainProximityNeedsOob) {
            reasons.add(OobRoutingReason.REQUESTED_TRAINING_PROXIMITIES);
        }
        if (AppContext.perform_train_imputation && trainProximityNeedsOob) {
            reasons.add(OobRoutingReason.TRAINING_IMPUTATION);
        }
        if (AppContext.perform_test_imputation && testTrainProximityNeedsOob) {
            reasons.add(OobRoutingReason.TESTING_IMPUTATION);
        }
        if (AppContext.get_training_outlier_scores
                && !AppContext.isIsolationMode()
                && !AppContext.isRegressionMode()
                && trainProximityNeedsOob) {
            reasons.add(
                    OobRoutingReason.TRAINING_PROXIMITY_OUTLIER_SCORES
            );
        }
        return new OobRoutingPlan(reasons);
    }

    /** Returns whether any requested task needs recursive OOB routing. */
    public boolean routingRequired() {
        return !reasons.isEmpty();
    }

    /** Returns the immutable reasons that enabled routing. */
    public Set<OobRoutingReason> reasons() {
        return reasons;
    }

    /** Validates combinations whose requested OOB semantics cannot exist. */
    public void validateConfiguration(SamplingStrategy samplingStrategy) {
        if (samplingStrategy == null) {
            throw new IllegalArgumentException(
                    "Sampling strategy cannot be null."
            );
        }
        if (reasons.contains(OobRoutingReason.OOB_SCORING)
                && samplingStrategy == SamplingStrategy.FULL) {
            throw new IllegalArgumentException(
                    "oob_scores=True requires sampling='bootstrap' or "
                            + "sampling='subsample'; sampling='full' has no "
                            + "out-of-bag observations."
            );
        }
        if (AppContext.return_oob_predictions && !AppContext.oob_scores) {
            throw new IllegalArgumentException(
                    "return_oob_predictions=True requires oob_scores=True so "
                            + "OOB instances are routed during fitting."
            );
        }
        if (AppContext.return_oob_predictions && AppContext.isIsolationMode()) {
            throw new IllegalArgumentException(
                    "OOB predictions are not currently supported in isolation mode."
            );
        }
    }

    private static boolean requiresOobRouting(
            Set<ProximityDataRequirement> requirements
    ) {
        if (requirements == null) {
            throw new IllegalStateException(
                    "Proximity requirements cannot be null."
            );
        }
        return requirements.contains(
                ProximityDataRequirement.TRAIN_OUT_OF_BAG_LEAF_ASSIGNMENTS
        ) || requirements.contains(
                ProximityDataRequirement.LEAF_ALL_TRAIN_MEMBERS
        );
    }

    @Override
    public String toString() {
        return "OobRoutingPlan{" + "reasons=" + reasons + '}';
    }
}