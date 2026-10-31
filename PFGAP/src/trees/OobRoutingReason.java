package trees;

/** Fit-time task that requires non-sampled training observations to be routed. */
public enum OobRoutingReason {
    /** Explicit fit-time OOB routing requested by oob_scores. */
    OOB_SCORING,
    REQUESTED_TRAINING_PROXIMITIES,
    TRAINING_IMPUTATION,
    TESTING_IMPUTATION,
    TRAINING_PROXIMITY_OUTLIER_SCORES
}