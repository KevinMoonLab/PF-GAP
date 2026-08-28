package output;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Writes outlier scores as indexed CSV records.
 *
 * Basic output:
 *
 *     instance_index,outlier_score
 *     0,0.481251
 *     1,0.720004
 *
 * Output with labels:
 *
 *     instance_index,outlier_score,label
 *     0,0.481251,class_a
 *     1,0.720004,class_b
 *
 * Output with an additional diagnostic:
 *
 *     instance_index,outlier_score,mean_path_length
 *     0,0.481251,7.42
 *     1,0.720004,4.11
 *
 * Output with both labels and diagnostics:
 *
 *     instance_index,outlier_score,label,mean_path_length
 *     0,0.481251,class_a,7.42
 *     1,0.720004,class_b,4.11
 *
 * The diagnostic column name is supplied by the caller so this writer can
 * support isolation path lengths or future outlier-scoring diagnostics
 * without becoming coupled to a particular scoring implementation.
 *
 * Labels and diagnostic columns are omitted if the supplied collection is
 * null, empty, or contains only null values.
 *
 * CSV quoting, UTF-8 output, directory creation, and non-finite numeric
 * rendering are delegated to {@link CsvUtils}.
 */
public final class OutlierScoreWriter {

    private static final String DEFAULT_DIAGNOSTIC_COLUMN =
            "diagnostic_value";

    private OutlierScoreWriter() {
    }

    /**
     * Writes outlier scores without labels or diagnostics.
     *
     * @param path output CSV path
     * @param scores outlier score array
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path write(
            Path path,
            double[] scores
    ) throws IOException {
        return write(
                path,
                scores,
                null,
                null,
                null
        );
    }

    /**
     * Writes outlier scores with optional labels.
     *
     * @param path output CSV path
     * @param scores outlier score array
     * @param labels labels aligned with the score array, or null
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path write(
            Path path,
            double[] scores,
            List<?> labels
    ) throws IOException {
        return write(
                path,
                scores,
                labels,
                null,
                null
        );
    }

    /**
     * Writes outlier scores with optional diagnostic values.
     *
     * The supplied diagnostic column name is used only when at least one
     * diagnostic value is present.
     *
     * @param path output CSV path
     * @param scores outlier score array
     * @param diagnosticColumnName diagnostic column heading
     * @param diagnosticValues diagnostic values aligned with scores
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeWithDiagnostics(
            Path path,
            double[] scores,
            String diagnosticColumnName,
            double[] diagnosticValues
    ) throws IOException {
        return write(
                path,
                scores,
                null,
                diagnosticColumnName,
                diagnosticValues
        );
    }

    /**
     * Writes outlier scores with optional labels and diagnostic values.
     *
     * @param path output CSV path
     * @param scores outlier score array
     * @param labels labels aligned with scores, or null
     * @param diagnosticColumnName diagnostic column heading
     * @param diagnosticValues diagnostic values aligned with scores, or null
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path write(
            Path path,
            double[] scores,
            List<?> labels,
            String diagnosticColumnName,
            double[] diagnosticValues
    ) throws IOException {
        validateArguments(
                path,
                scores,
                labels,
                diagnosticColumnName,
                diagnosticValues
        );

        boolean includeLabels =
                hasValues(labels);

        boolean includeDiagnostics =
                diagnosticValues != null;

        String normalizedDiagnosticName =
                includeDiagnostics
                        ? normalizeDiagnosticColumnName(
                        diagnosticColumnName
                )
                        : null;

        Path outputPath =
                normalizePath(path);

        try (BufferedWriter writer =
                     CsvUtils.newWriter(outputPath)) {

            writeHeader(
                    writer,
                    includeLabels,
                    includeDiagnostics,
                    normalizedDiagnosticName
            );

            for (int instanceIndex = 0;
                 instanceIndex < scores.length;
                 instanceIndex++) {

                Object scoreField =
                        CsvUtils.numericField(
                                scores[instanceIndex]
                        );

                Object label =
                        includeLabels
                                ? labels.get(instanceIndex)
                                : null;

                Object diagnosticField =
                        includeDiagnostics
                                ? CsvUtils.numericField(
                                diagnosticValues[instanceIndex]
                        )
                                : null;

                writeRow(
                        writer,
                        instanceIndex,
                        scoreField,
                        label,
                        diagnosticField,
                        includeLabels,
                        includeDiagnostics
                );
            }
        }

        return outputPath;
    }

    /**
     * Writes boxed outlier scores without labels or diagnostics.
     *
     * Null score values are represented as blank CSV fields.
     *
     * @param path output CSV path
     * @param scores boxed outlier scores
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path write(
            Path path,
            List<? extends Number> scores
    ) throws IOException {
        return writeBoxed(
                path,
                scores,
                null
        );
    }

    /**
     * Writes boxed outlier scores with optional labels.
     *
     * Null score values are represented as blank CSV fields.
     *
     * @param path output CSV path
     * @param scores boxed outlier scores
     * @param labels labels aligned with scores, or null
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path writeBoxed(
            Path path,
            List<? extends Number> scores,
            List<?> labels
    ) throws IOException {
        Objects.requireNonNull(
                path,
                "Outlier-score output path cannot be null."
        );

        Objects.requireNonNull(
                scores,
                "Outlier-score list cannot be null."
        );

        validateAlignedList(
                labels,
                scores.size(),
                "label"
        );

        boolean includeLabels =
                hasValues(labels);

        Path outputPath =
                normalizePath(path);

        try (BufferedWriter writer =
                     CsvUtils.newWriter(outputPath)) {

            if (includeLabels) {
                CsvUtils.writeHeader(
                        writer,
                        "instance_index",
                        "outlier_score",
                        "label"
                );
            } else {
                CsvUtils.writeHeader(
                        writer,
                        "instance_index",
                        "outlier_score"
                );
            }

            for (int instanceIndex = 0;
                 instanceIndex < scores.size();
                 instanceIndex++) {

                Number score =
                        scores.get(instanceIndex);

                Object scoreField =
                        score == null
                                ? null
                                : CsvUtils.numericField(
                                score.doubleValue()
                        );

                if (includeLabels) {
                    CsvUtils.writeRow(
                            writer,
                            instanceIndex,
                            scoreField,
                            labels.get(instanceIndex)
                    );
                } else {
                    CsvUtils.writeRow(
                            writer,
                            instanceIndex,
                            scoreField
                    );
                }
            }
        }

        return outputPath;
    }

    private static void writeHeader(
            BufferedWriter writer,
            boolean includeLabels,
            boolean includeDiagnostics,
            String diagnosticColumnName
    ) throws IOException {
        if (includeLabels && includeDiagnostics) {
            CsvUtils.writeHeader(
                    writer,
                    "instance_index",
                    "outlier_score",
                    "label",
                    diagnosticColumnName
            );

            return;
        }

        if (includeLabels) {
            CsvUtils.writeHeader(
                    writer,
                    "instance_index",
                    "outlier_score",
                    "label"
            );

            return;
        }

        if (includeDiagnostics) {
            CsvUtils.writeHeader(
                    writer,
                    "instance_index",
                    "outlier_score",
                    diagnosticColumnName
            );

            return;
        }

        CsvUtils.writeHeader(
                writer,
                "instance_index",
                "outlier_score"
        );
    }

    private static void writeRow(
            BufferedWriter writer,
            int instanceIndex,
            Object score,
            Object label,
            Object diagnostic,
            boolean includeLabels,
            boolean includeDiagnostics
    ) throws IOException {
        if (includeLabels && includeDiagnostics) {
            CsvUtils.writeRow(
                    writer,
                    instanceIndex,
                    score,
                    label,
                    diagnostic
            );

            return;
        }

        if (includeLabels) {
            CsvUtils.writeRow(
                    writer,
                    instanceIndex,
                    score,
                    label
            );

            return;
        }

        if (includeDiagnostics) {
            CsvUtils.writeRow(
                    writer,
                    instanceIndex,
                    score,
                    diagnostic
            );

            return;
        }

        CsvUtils.writeRow(
                writer,
                instanceIndex,
                score
        );
    }

    private static void validateArguments(
            Path path,
            double[] scores,
            List<?> labels,
            String diagnosticColumnName,
            double[] diagnosticValues
    ) {
        Objects.requireNonNull(
                path,
                "Outlier-score output path cannot be null."
        );

        Objects.requireNonNull(
                scores,
                "Outlier-score array cannot be null."
        );

        validateAlignedList(
                labels,
                scores.length,
                "label"
        );

        if (diagnosticValues != null
                && diagnosticValues.length != scores.length) {

            throw new IllegalArgumentException(
                    "Outlier-score and diagnostic-value counts differ. "
                            + "Scores="
                            + scores.length
                            + ", diagnostic values="
                            + diagnosticValues.length
                            + "."
            );
        }

        if (diagnosticValues == null
                && diagnosticColumnName != null
                && !diagnosticColumnName.isBlank()) {

            throw new IllegalArgumentException(
                    "A diagnostic column name was supplied, but no "
                            + "diagnostic values were provided."
            );
        }
    }

    private static void validateAlignedList(
            List<?> values,
            int expectedSize,
            String valueDescription
    ) {
        if (values == null || values.isEmpty()) {
            return;
        }

        if (values.size() != expectedSize) {
            throw new IllegalArgumentException(
                    "Outlier-score and "
                            + valueDescription
                            + " counts differ. Scores="
                            + expectedSize
                            + ", "
                            + valueDescription
                            + " values="
                            + values.size()
                            + "."
            );
        }
    }

    /**
     * Returns whether a collection contains at least one non-null value.
     *
     * A dataset may maintain a label list containing one null entry for
     * every instance when labels are unavailable. Such a list should not
     * cause an empty label column to be written.
     */
    private static boolean hasValues(
            List<?> values
    ) {
        if (values == null || values.isEmpty()) {
            return false;
        }

        for (Object value : values) {
            if (value != null) {
                return true;
            }
        }

        return false;
    }

    private static String normalizeDiagnosticColumnName(
            String diagnosticColumnName
    ) {
        if (diagnosticColumnName == null
                || diagnosticColumnName.isBlank()) {

            return DEFAULT_DIAGNOSTIC_COLUMN;
        }

        return diagnosticColumnName.trim();
    }

    private static Path normalizePath(
            Path path
    ) {
        return path.toAbsolutePath()
                .normalize();
    }
}