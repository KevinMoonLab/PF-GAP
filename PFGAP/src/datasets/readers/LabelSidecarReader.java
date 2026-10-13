package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRecord;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Reads positional scalar or multi-output labels from a delimited sidecar.
 *
 * <p>Sidecar row {@code i} belongs to per-file observation {@code i} after the
 * dataset reader has applied its deterministic file ordering. One sidecar
 * field materializes as a scalar label; multiple fields materialize as an
 * immutable ordered list.</p>
 *
 * <p>The sidecar deliberately uses the same field separator and header setting
 * as its associated instance files. Labels are loaded eagerly even when the
 * feature series remain lazy.</p>
 */
public final class LabelSidecarReader {

    private LabelSidecarReader() {
    }

    /** Supported supervised forest modes for label parsing. */
    public enum LabelMode {
        CLASSIFICATION,
        REGRESSION;

        /**
         * Parses the ordinary forest-mode names used by PFGAP.
         * Isolation has no supervised labels and is therefore rejected.
         */
        public static LabelMode fromForestMode(String forestMode) {
            String normalized = Objects.requireNonNull(
                    forestMode,
                    "forestMode cannot be null."
            ).trim().toLowerCase(Locale.ROOT);

            return switch (normalized) {
                case "classification" -> CLASSIFICATION;
                case "regression" -> REGRESSION;
                case "isolation" -> throw new IllegalArgumentException(
                        "Isolation mode does not use a label sidecar."
                );
                default -> throw new IllegalArgumentException(
                        "Unsupported forest mode for label sidecar: "
                                + forestMode
                );
            };
        }
    }

    /**
     * Reads a positional label sidecar with FastCSV.
     *
     * @param labelPath sidecar path
     * @param hasHeader whether the first record is a header
     * @param fieldSeparator one-character field separator
     * @param forestMode classification or regression
     * @return ordered scalar or list-valued labels
     * @throws IOException if the sidecar cannot be read
     */
    public static List<Object> read(
            String labelPath,
            boolean hasHeader,
            String fieldSeparator,
            String forestMode
    ) throws IOException {
        return read(
                Path.of(requireNonblank(labelPath, "labelPath")),
                hasHeader,
                requireSingleCharacter(
                        fieldSeparator,
                        "fieldSeparator"
                ),
                LabelMode.fromForestMode(forestMode)
        );
    }

    /** Reads a positional sidecar using an already resolved mode. */
    public static List<Object> read(
            Path labelPath,
            boolean hasHeader,
            char fieldSeparator,
            LabelMode mode
    ) throws IOException {
        Path path = validateFile(labelPath);
        LabelMode nonnullMode = Objects.requireNonNull(
                mode,
                "Label mode cannot be null."
        );

        ArrayList<Object> labels = new ArrayList<>();
        int expectedOutputCount = -1;
        int recordIndex = 0;

        try (CsvReader<CsvRecord> csvReader = CsvReader.builder()
                .fieldSeparator(fieldSeparator)
                .skipEmptyLines(false)
                .detectBomHeader(true)
                .ofCsvRecord(path)) {
            var iterator = csvReader.iterator();

            if (hasHeader) {
                if (!iterator.hasNext()) {
                    throw new IOException(
                            "Label sidecar contains no header or data: " + path
                    );
                }
                List<String> header = iterator.next().getFields();
                validateRecordWidth(
                        header,
                        -1,
                        path,
                        "header"
                );
                expectedOutputCount = header.size();
            }

            while (iterator.hasNext()) {
                List<String> fields = iterator.next().getFields();
                validateRecordWidth(
                        fields,
                        recordIndex,
                        path,
                        "label record"
                );

                if (expectedOutputCount < 0) {
                    expectedOutputCount = fields.size();
                } else if (fields.size() != expectedOutputCount) {
                    throw new IllegalArgumentException(
                            "Label sidecar output count changed at record "
                                    + recordIndex + " in " + path
                                    + ": expected " + expectedOutputCount
                                    + " fields, but found " + fields.size()
                                    + "."
                    );
                }

                labels.add(parseRecord(
                        fields,
                        nonnullMode,
                        recordIndex,
                        path
                ));
                recordIndex++;
            }
        } catch (IllegalArgumentException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            throw new IOException(
                    "Failed while parsing label sidecar: " + path,
                    exception
            );
        }

        if (labels.isEmpty()) {
            throw new IOException(
                    "Label sidecar contains no label records: " + path
            );
        }
        return List.copyOf(labels);
    }

    /** Ensures positional sidecar labels align with discovered instance files. */
    public static void validateCount(
            List<?> labels,
            int instanceFileCount,
            Path labelPath
    ) {
        Objects.requireNonNull(labels, "Labels cannot be null.");
        if (instanceFileCount < 0) {
            throw new IllegalArgumentException(
                    "Instance-file count cannot be negative."
            );
        }
        if (labels.size() != instanceFileCount) {
            throw new IllegalArgumentException(
                    "Label sidecar " + labelPath + " contains "
                            + labels.size() + " records, but "
                            + instanceFileCount
                            + " instance files were discovered."
            );
        }
    }

    private static Object parseRecord(
            List<String> fields,
            LabelMode mode,
            int recordIndex,
            Path path
    ) {
        if (fields.size() == 1) {
            return parseComponent(
                    fields.get(0),
                    mode,
                    recordIndex,
                    0,
                    path
            );
        }

        ArrayList<Object> outputs = new ArrayList<>(fields.size());
        for (int outputIndex = 0;
             outputIndex < fields.size();
             outputIndex++) {
            outputs.add(parseComponent(
                    fields.get(outputIndex),
                    mode,
                    recordIndex,
                    outputIndex,
                    path
            ));
        }
        return List.copyOf(outputs);
    }

    private static Object parseComponent(
            String field,
            LabelMode mode,
            int recordIndex,
            int outputIndex,
            Path path
    ) {
        String token = field == null ? "" : field.trim();
        if (token.isEmpty()) {
            throw invalidComponent(
                    "Label component cannot be empty",
                    recordIndex,
                    outputIndex,
                    path,
                    null
            );
        }

        if (mode == LabelMode.REGRESSION) {
            try {
                double value = JavaDoubleParser.parseDouble(token);
                if (!Double.isFinite(value)) {
                    throw invalidComponent(
                            "Regression target must be finite, but was " + value,
                            recordIndex,
                            outputIndex,
                            path,
                            null
                    );
                }
                return value;
            } catch (NumberFormatException exception) {
                throw invalidComponent(
                        "Regression target is not numeric: '" + token + "'",
                        recordIndex,
                        outputIndex,
                        path,
                        exception
                );
            }
        }

        try {
            return Integer.valueOf(token);
        } catch (NumberFormatException ignored) {
            return token;
        }
    }

    private static void validateRecordWidth(
            List<String> fields,
            int recordIndex,
            Path path,
            String role
    ) {
        if (fields == null || fields.isEmpty()) {
            String location = recordIndex < 0
                    ? ""
                    : " at record " + recordIndex;
            throw new IllegalArgumentException(
                    "Encountered an empty " + role + location
                            + " in label sidecar " + path + "."
            );
        }
    }

    private static IllegalArgumentException invalidComponent(
            String message,
            int recordIndex,
            int outputIndex,
            Path path,
            Throwable cause
    ) {
        String fullMessage = message
                + " at label record " + recordIndex
                + ", output " + outputIndex
                + " in " + path + ".";
        return cause == null
                ? new IllegalArgumentException(fullMessage)
                : new IllegalArgumentException(fullMessage, cause);
    }

    private static Path validateFile(Path labelPath) throws IOException {
        Path path = Objects.requireNonNull(
                labelPath,
                "Label path cannot be null."
        ).toAbsolutePath().normalize();
        if (!Files.exists(path)) {
            throw new IOException(
                    "Label sidecar does not exist: " + path
            );
        }
        if (!Files.isRegularFile(path)) {
            throw new IOException(
                    "Label sidecar path is not a regular file: " + path
            );
        }
        if (!Files.isReadable(path)) {
            throw new IOException(
                    "Label sidecar is not readable: " + path
            );
        }
        return path;
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    argumentName + " cannot be null or blank."
            );
        }
        return value.trim();
    }

    private static char requireSingleCharacter(
            String value,
            String argumentName
    ) {
        String normalized = requireNonblank(value, argumentName);
        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    argumentName + " must contain exactly one character."
            );
        }
        return normalized.charAt(0);
    }
}
