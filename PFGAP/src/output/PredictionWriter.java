package output;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Writes classification or regression predictions as indexed CSV records.
 *
 * This writer is deliberately independent of ProximityForestResult,
 * AppContext, and ListObjectDataset. The calling orchestration layer is
 * responsible for:
 *
 *     1. Selecting the output path.
 *
 *     2. Converting internal classification labels back to their original
 *        user-facing values.
 *
 *     3. Supplying actual labels or targets when they are available.
 *
 * Classification output without actual labels:
 *
 *     instance_index,prediction
 *     0,class_a
 *     1,class_b
 *
 * Classification output with actual labels:
 *
 *     instance_index,prediction,actual,correct
 *     0,class_a,class_a,true
 *     1,class_b,class_a,false
 *
 * Regression output without actual targets:
 *
 *     instance_index,prediction
 *     0,1.25
 *     1,4.81
 *
 * Regression output with actual targets:
 *
 *     instance_index,prediction,actual,residual,absolute_error
 *     0,1.25,1.30,-0.05,0.05
 *     1,4.81,4.50,0.31,0.31
 *
 * Regression residuals use:
 *
 *     residual = prediction - actual
 *
 * Actual-value columns are omitted when the supplied actual-value list is
 * null, empty, or contains only null values.
 *
 * CSV quoting, UTF-8 output, directory creation, and non-finite numeric
 * rendering are delegated to {@link CsvUtils}.
 */
public final class PredictionWriter {

    private PredictionWriter() {
    }

    /**
     * Writes classification predictions.
     *
     * @param path output CSV path
     * @param predictions user-facing predicted labels
     * @param actualLabels actual labels, or null when unavailable
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeClassification(
            Path path,
            List<?> predictions,
            List<?> actualLabels
    ) throws IOException {
        validateCommonArguments(
                path,
                predictions,
                actualLabels
        );

        boolean includeActual =
                hasActualValues(actualLabels);

        Path outputPath =
                normalizePath(path);

        try (BufferedWriter writer =
                     CsvUtils.newWriter(outputPath)) {

            if (includeActual) {
                CsvUtils.writeHeader(
                        writer,
                        "instance_index",
                        "prediction",
                        "actual",
                        "correct"
                );
            } else {
                CsvUtils.writeHeader(
                        writer,
                        "instance_index",
                        "prediction"
                );
            }

            for (int instanceIndex = 0;
                 instanceIndex < predictions.size();
                 instanceIndex++) {

                Object prediction =
                        predictions.get(instanceIndex);

                if (!includeActual) {
                    CsvUtils.writeRow(
                            writer,
                            instanceIndex,
                            prediction
                    );

                    continue;
                }

                Object actual =
                        actualLabels.get(instanceIndex);

                Object correct =
                        actual == null
                                ? null
                                : Objects.equals(
                                prediction,
                                actual
                        );

                CsvUtils.writeRow(
                        writer,
                        instanceIndex,
                        prediction,
                        actual,
                        correct
                );
            }
        }

        return outputPath;
    }

    /**
     * Writes classification predictions when actual labels are unavailable.
     *
     * @param path output CSV path
     * @param predictions user-facing predicted labels
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeClassification(
            Path path,
            List<?> predictions
    ) throws IOException {
        return writeClassification(
                path,
                predictions,
                null
        );
    }

    /**
     * Writes regression predictions.
     *
     * Predictions must be numeric. Non-null actual targets must also be
     * numeric.
     *
     * Null predictions are rejected because a regression prediction should
     * always contain a numeric result. Null actual targets are permitted and
     * produce blank actual, residual, and absolute-error fields.
     *
     * @param path output CSV path
     * @param predictions predicted numeric targets
     * @param actualTargets actual numeric targets, or null when unavailable
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeRegression(
            Path path,
            List<?> predictions,
            List<?> actualTargets
    ) throws IOException {
        validateCommonArguments(
                path,
                predictions,
                actualTargets
        );

        validateRegressionPredictions(
                predictions
        );

        boolean includeActual =
                hasActualValues(actualTargets);

        if (includeActual) {
            validateRegressionActualTargets(
                    actualTargets
            );
        }

        Path outputPath =
                normalizePath(path);

        try (BufferedWriter writer =
                     CsvUtils.newWriter(outputPath)) {

            if (includeActual) {
                CsvUtils.writeHeader(
                        writer,
                        "instance_index",
                        "prediction",
                        "actual",
                        "residual",
                        "absolute_error"
                );
            } else {
                CsvUtils.writeHeader(
                        writer,
                        "instance_index",
                        "prediction"
                );
            }

            for (int instanceIndex = 0;
                 instanceIndex < predictions.size();
                 instanceIndex++) {

                Number predictionNumber =
                        (Number) predictions.get(
                                instanceIndex
                        );

                double prediction =
                        predictionNumber.doubleValue();

                if (!includeActual) {
                    CsvUtils.writeRow(
                            writer,
                            instanceIndex,
                            CsvUtils.numericField(
                                    prediction
                            )
                    );

                    continue;
                }

                Object actualValue =
                        actualTargets.get(
                                instanceIndex
                        );

                if (actualValue == null) {
                    CsvUtils.writeRow(
                            writer,
                            instanceIndex,
                            CsvUtils.numericField(
                                    prediction
                            ),
                            null,
                            null,
                            null
                    );

                    continue;
                }

                double actual =
                        ((Number) actualValue)
                                .doubleValue();

                double residual =
                        prediction - actual;

                double absoluteError =
                        Math.abs(residual);

                CsvUtils.writeRow(
                        writer,
                        instanceIndex,
                        CsvUtils.numericField(
                                prediction
                        ),
                        CsvUtils.numericField(
                                actual
                        ),
                        CsvUtils.numericField(
                                residual
                        ),
                        CsvUtils.numericField(
                                absoluteError
                        )
                );
            }
        }

        return outputPath;
    }

    /**
     * Writes regression predictions when actual targets are unavailable.
     *
     * @param path output CSV path
     * @param predictions predicted numeric targets
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeRegression(
            Path path,
            List<?> predictions
    ) throws IOException {
        return writeRegression(
                path,
                predictions,
                null
        );
    }

    /**
     * Selects classification or regression output through one convenience
     * entry point.
     *
     * @param path output CSV path
     * @param predictions predicted labels or numeric targets
     * @param actualValues actual labels or targets, possibly null
     * @param regression whether predictions represent regression targets
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path write(
            Path path,
            List<?> predictions,
            List<?> actualValues,
            boolean regression
    ) throws IOException {
        if (regression) {
            return writeRegression(
                    path,
                    predictions,
                    actualValues
            );
        }

        return writeClassification(
                path,
                predictions,
                actualValues
        );
    }

    /**
     * Selects classification or regression output when actual values are
     * unavailable.
     *
     * @param path output CSV path
     * @param predictions predicted labels or numeric targets
     * @param regression whether predictions represent regression targets
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path write(
            Path path,
            List<?> predictions,
            boolean regression
    ) throws IOException {
        return write(
                path,
                predictions,
                null,
                regression
        );
    }

    private static void validateCommonArguments(
            Path path,
            List<?> predictions,
            List<?> actualValues
    ) {
        Objects.requireNonNull(
                path,
                "Prediction output path cannot be null."
        );

        Objects.requireNonNull(
                predictions,
                "Prediction list cannot be null."
        );

        if (actualValues != null
                && !actualValues.isEmpty()
                && actualValues.size()
                != predictions.size()) {

            throw new IllegalArgumentException(
                    "Prediction and actual-value counts differ. "
                            + "Predictions="
                            + predictions.size()
                            + ", actual values="
                            + actualValues.size()
                            + "."
            );
        }
    }

    /**
     * Returns whether actual labels or targets contain at least one
     * non-null value.
     *
     * ListObjectDataset may maintain a label list containing one null entry
     * per instance when labels are unavailable. Treating such a list as
     * labeled would create unnecessary empty actual-value columns.
     */
    private static boolean hasActualValues(
            List<?> actualValues
    ) {
        if (actualValues == null
                || actualValues.isEmpty()) {

            return false;
        }

        for (Object value : actualValues) {
            if (value != null) {
                return true;
            }
        }

        return false;
    }

    private static void validateRegressionPredictions(
            List<?> predictions
    ) {
        for (int instanceIndex = 0;
             instanceIndex < predictions.size();
             instanceIndex++) {

            Object prediction =
                    predictions.get(instanceIndex);

            if (!(prediction instanceof Number)) {
                throw new IllegalArgumentException(
                        "Regression prediction at instance "
                                + instanceIndex
                                + " must be numeric, but received "
                                + describeValue(prediction)
                                + "."
                );
            }
        }
    }

    private static void validateRegressionActualTargets(
            List<?> actualTargets
    ) {
        for (int instanceIndex = 0;
             instanceIndex < actualTargets.size();
             instanceIndex++) {

            Object actual =
                    actualTargets.get(instanceIndex);

            if (actual == null) {
                continue;
            }

            if (!(actual instanceof Number)) {
                throw new IllegalArgumentException(
                        "Regression actual target at instance "
                                + instanceIndex
                                + " must be numeric or null, but received "
                                + describeValue(actual)
                                + "."
                );
            }
        }
    }

    private static Path normalizePath(
            Path path
    ) {
        return path.toAbsolutePath()
                .normalize();
    }

    private static String describeValue(
            Object value
    ) {
        if (value == null) {
            return "null";
        }

        return "'"
                + value
                + "' of type "
                + value.getClass()
                .getName();
    }
}