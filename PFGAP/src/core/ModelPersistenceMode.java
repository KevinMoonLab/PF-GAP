package core;

import java.io.Serializable;

/**
 * Defines how a trained PFGAP model retains its training data when saved.
 */
public enum ModelPersistenceMode implements Serializable {

    /**
     * Saves the complete training dataset together with the trained forest.
     *
     * <p>This preserves the current persistence behavior and retains operations
     * that require arbitrary training values, including imputation.</p>
     */
    EMBEDDED,

    /**
     * Saves the trained forest without the complete training dataset.
     *
     * <p>For v1, this mode is supported only for models trained from eagerly
     * loaded data, so that the split exemplars retained by the forest are
     * already materialized. Prediction, OOD scoring, and supported proximity
     * calculations remain available, while operations requiring arbitrary
     * training values, including imputation, are unavailable.</p>
     */
    PREDICTION_ONLY
}
