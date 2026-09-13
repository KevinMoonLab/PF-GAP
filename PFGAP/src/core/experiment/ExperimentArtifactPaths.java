package core.experiment;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Resolves output paths and repetition-specific artifact names for one PFGAP
 * experiment run.
 *
 * <p>The configured output directory is normalized once when this object is
 * constructed. All returned artifact paths are absolute and normalized.</p>
 *
 * <p>This class performs no file I/O. It only owns path and filename policy.</p>
 */
public final class ExperimentArtifactPaths {

    public static final String TRAINING_OUTLIER_SCORES =
            "training_outlier_scores.csv";

    public static final String VALIDATION_OUTLIER_SCORES =
            "validation_outlier_scores.csv";

    public static final String TEST_OUTLIER_SCORES =
            "test_outlier_scores.csv";

    public static final String VALIDATION_PREDICTIONS =
            "validation_predictions.csv";

    public static final String TEST_PREDICTIONS =
            "test_predictions.csv";

    public static final String TRAINING_PROXIMITIES_SPARSE =
            "training_proximities.mtx";

    public static final String TRAINING_PROXIMITIES_DENSE =
            "training_proximities.csv";

    public static final String TEST_TRAIN_PROXIMITIES_SPARSE =
            "test_train_proximities.mtx";

    public static final String TEST_TRAIN_PROXIMITIES_DENSE =
            "test_train_proximities.csv";

    private final Path outputDirectory;
    private final int repetitionCount;

    /**
     * Creates experiment artifact path policy.
     *
     * @param outputDirectory configured output directory
     * @param repetitionCount total number of experiment repetitions
     */
    public ExperimentArtifactPaths(
            String outputDirectory,
            int repetitionCount
    ) {
        this(
                requireOutputDirectory(outputDirectory),
                repetitionCount
        );
    }

    /**
     * Creates experiment artifact path policy.
     *
     * @param outputDirectory configured output directory
     * @param repetitionCount total number of experiment repetitions
     */
    public ExperimentArtifactPaths(
            Path outputDirectory,
            int repetitionCount
    ) {
        if (outputDirectory == null) {
            throw new IllegalArgumentException(
                    "Output directory cannot be null."
            );
        }
        if (repetitionCount < 1) {
            throw new IllegalArgumentException(
                    "Repetition count must be positive. Received: "
                            + repetitionCount
                            + "."
            );
        }

        this.outputDirectory =
                outputDirectory.toAbsolutePath()
                        .normalize();

        this.repetitionCount =
                repetitionCount;
    }

    public Path getOutputDirectory() {
        return outputDirectory;
    }

    public int getRepetitionCount() {
        return repetitionCount;
    }

    /**
     * Resolves an ordinary artifact filename in the configured output
     * directory.
     */
    public Path resolve(String fileName) {
        return outputDirectory.resolve(
                        requireFileName(fileName)
                )
                .normalize();
    }

    /**
     * Resolves an artifact filename after applying repetition suffix policy.
     */
    public Path resolveRepeated(
            String baseFileName,
            int repetition
    ) {
        return resolve(
                repeatedFileName(
                        baseFileName,
                        repetition
                )
        );
    }

    /**
     * Adds a one-based repetition suffix when the run contains multiple
     * repetitions.
     *
     * <p>Examples:</p>
     *
     * <pre>
     * predictions.csv  -> predictions_repeat_2.csv
     * model             -> model_repeat_2
     * </pre>
     */
    public String repeatedFileName(
            String baseFileName,
            int repetition
    ) {
        String validatedBaseName =
                requireFileName(baseFileName);

        validateRepetition(repetition);

        if (repetitionCount <= 1) {
            return validatedBaseName;
        }

        int extensionSeparator =
                validatedBaseName.lastIndexOf('.');

        String suffix =
                "_repeat_" + (repetition + 1);

        if (extensionSeparator <= 0) {
            return validatedBaseName + suffix;
        }

        return validatedBaseName.substring(
                0,
                extensionSeparator
        )
                + suffix
                + validatedBaseName.substring(
                extensionSeparator
        );
    }

    /**
     * Returns a path relative to the output directory when possible.
     *
     * <p>If the supplied path is on another filesystem root, the normalized
     * absolute path is returned instead.</p>
     */
    public String relativeArtifactPath(Path artifactPath) {
        if (artifactPath == null) {
            return null;
        }

        Path normalizedArtifact =
                artifactPath.toAbsolutePath()
                        .normalize();

        try {
            return outputDirectory.relativize(
                    normalizedArtifact
            ).toString();
        } catch (IllegalArgumentException exception) {
            return normalizedArtifact.toString();
        }
    }

    public Path trainingOutlierScores(int repetition) {
        return resolveRepeated(
                TRAINING_OUTLIER_SCORES,
                repetition
        );
    }

    public Path validationOutlierScores(int repetition) {
        return resolveRepeated(
                VALIDATION_OUTLIER_SCORES,
                repetition
        );
    }

    public Path testOutlierScores(int repetition) {
        return resolveRepeated(
                TEST_OUTLIER_SCORES,
                repetition
        );
    }

    public Path validationPredictions(int repetition) {
        return resolveRepeated(
                VALIDATION_PREDICTIONS,
                repetition
        );
    }

    public Path testPredictions(int repetition) {
        return resolveRepeated(
                TEST_PREDICTIONS,
                repetition
        );
    }

    public Path trainingProximitiesSparse(int repetition) {
        return resolveRepeated(
                TRAINING_PROXIMITIES_SPARSE,
                repetition
        );
    }

    public Path trainingProximitiesDense(int repetition) {
        return resolveRepeated(
                TRAINING_PROXIMITIES_DENSE,
                repetition
        );
    }

    public Path testTrainProximitiesSparse(int repetition) {
        return resolveRepeated(
                TEST_TRAIN_PROXIMITIES_SPARSE,
                repetition
        );
    }

    public Path testTrainProximitiesDense(int repetition) {
        return resolveRepeated(
                TEST_TRAIN_PROXIMITIES_DENSE,
                repetition
        );
    }

    private void validateRepetition(int repetition) {
        if (repetition < 0 || repetition >= repetitionCount) {
            throw new IllegalArgumentException(
                    "Repetition index must be within [0, "
                            + repetitionCount
                            + "). Received: "
                            + repetition
                            + "."
            );
        }
    }

    private static Path requireOutputDirectory(String outputDirectory) {
        if (outputDirectory == null || outputDirectory.isBlank()) {
            throw new IllegalArgumentException(
                    "Output directory cannot be null or blank."
            );
        }

        return Paths.get(
                outputDirectory.trim()
        );
    }

    private static String requireFileName(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            throw new IllegalArgumentException(
                    "Artifact filename cannot be null or blank."
            );
        }

        Path candidate =
                Paths.get(
                        fileName.trim()
                );

        if (candidate.isAbsolute()
                || candidate.getNameCount() != 1) {
            throw new IllegalArgumentException(
                    "Artifact filename must be a single relative filename. "
                            + "Received: "
                            + fileName
                            + "."
            );
        }

        String normalized =
                candidate.toString();

        if (normalized.equals(".")
                || normalized.equals("..")) {
            throw new IllegalArgumentException(
                    "Artifact filename cannot be '.' or '..'."
            );
        }

        return normalized;
    }
}