package core;

import datasets.ListObjectDataset;
import trees.ProximityForest;

import java.io.Serializable;
import java.util.Objects;

/**
 * Versioned serialization envelope for a saved PFGAP model.
 *
 * <p>This class is an internal persistence representation. It is distinct from
 * {@link ModelIO.LoadedModel}, which is the runtime value returned to callers
 * after a model has been read.</p>
 */
final class PersistedModel implements Serializable {

    private static final long serialVersionUID = 1L;

    static final int CURRENT_FORMAT_VERSION = 1;

    private final int formatVersion;
    private final ModelPersistenceMode persistenceMode;
    private final ProximityForest forest;
    private final ListObjectDataset trainingData;
    private final int trainingSize;
    private final AppContextSnapshot snapshot;

    private PersistedModel(
            int formatVersion,
            ModelPersistenceMode persistenceMode,
            ProximityForest forest,
            ListObjectDataset trainingData,
            int trainingSize,
            AppContextSnapshot snapshot
    ) {
        if (formatVersion <= 0) {
            throw new IllegalArgumentException(
                    "formatVersion must be greater than zero."
            );
        }
        if (trainingSize < 0) {
            throw new IllegalArgumentException(
                    "trainingSize cannot be negative."
            );
        }

        this.formatVersion = formatVersion;
        this.persistenceMode = Objects.requireNonNull(
                persistenceMode,
                "persistenceMode cannot be null."
        );
        this.forest = Objects.requireNonNull(
                forest,
                "forest cannot be null."
        );
        this.snapshot = Objects.requireNonNull(
                snapshot,
                "snapshot cannot be null."
        );

        this.trainingSize = trainingSize;

        switch (persistenceMode) {
            case EMBEDDED:
                this.trainingData = Objects.requireNonNull(
                        trainingData,
                        "EMBEDDED persistence requires trainingData."
                );
                if (trainingData.size() != trainingSize) {
                    throw new IllegalArgumentException(
                            "trainingSize must match trainingData.size() " +
                            "for EMBEDDED persistence."
                    );
                }
                break;

            case PREDICTION_ONLY:
                if (trainingData != null) {
                    throw new IllegalArgumentException(
                            "PREDICTION_ONLY persistence must not contain " +
                            "trainingData."
                    );
                }
                this.trainingData = null;
                break;

            default:
                throw new IllegalArgumentException(
                        "Unsupported persistence mode: " + persistenceMode
                );
        }
    }

    static PersistedModel embedded(
            ProximityForest forest,
            ListObjectDataset trainingData,
            AppContextSnapshot snapshot
    ) {
        Objects.requireNonNull(
                trainingData,
                "trainingData cannot be null."
        );
        return new PersistedModel(
                CURRENT_FORMAT_VERSION,
                ModelPersistenceMode.EMBEDDED,
                forest,
                trainingData,
                trainingData.size(),
                snapshot
        );
    }

    static PersistedModel predictionOnly(
            ProximityForest forest,
            int trainingSize,
            AppContextSnapshot snapshot
    ) {
        return new PersistedModel(
                CURRENT_FORMAT_VERSION,
                ModelPersistenceMode.PREDICTION_ONLY,
                forest,
                null,
                trainingSize,
                snapshot
        );
    }

    int getFormatVersion() {
        return formatVersion;
    }

    ModelPersistenceMode getPersistenceMode() {
        return persistenceMode;
    }

    ProximityForest getForest() {
        return forest;
    }

    ListObjectDataset getTrainingData() {
        return trainingData;
    }

    int getTrainingSize() {
        return trainingSize;
    }

    AppContextSnapshot getSnapshot() {
        return snapshot;
    }
}
