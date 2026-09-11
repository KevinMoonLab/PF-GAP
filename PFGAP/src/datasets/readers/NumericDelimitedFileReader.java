package datasets.readers;

import ch.randelshofer.fastdoubleparser.JavaDoubleParser;
import core.AppContext;
import datasets.ListObjectDataset;
import de.siegmar.fastcsv.reader.AbstractBaseCsvCallbackHandler;
import de.siegmar.fastcsv.reader.CsvReader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * High-throughput reader for row-oriented numeric delimited datasets.
 *
 * <p>Physical layout:</p>
 *
 * <pre>
 * one delimited record = one dataset instance
 * one feature field    = one entry in a primitive double[]
 * </pre>
 *
 * <p>The reader supports a single-character delimiter, including an actual tab
 * or the escaped-tab configuration value {@code "\\t"}. A label may be embedded
 * in the first or last field, supplied in a separate label file, or omitted for
 * isolation and unlabeled testing workflows.</p>
 *
 * <p>Numeric fields are parsed directly from FastCSV's character buffer with
 * {@link JavaDoubleParser}. Except for the first headerless record, this avoids
 * creating {@code CsvRecord}, {@code List<String>}, and one {@code String} per
 * numeric value. Each completed record is added directly to the final
 * {@link ListObjectDataset} as a primitive {@code double[]}.</p>
 *
 * <p>When {@code hasMissingValues=true}, configured missing tokens and blank
 * numeric fields are represented by {@link Double#NaN}. When it is false, the
 * same inputs produce a descriptive failure. This class never creates boxed
 * numeric arrays.</p>
 *
 * <p>This reader returns raw eager data. Standardization and other preprocessing
 * remain owned by PFGAP's preprocessing pipeline.</p>
 */
public class NumericDelimitedFileReader implements DatasetReader {

    private final String dataFileName;
    private final String labelFileName;
    private final String entrySeparator;
    private final char fieldSeparator;
    private final boolean hasHeader;
    private final boolean hasMissingValues;
    private final boolean targetColumnIsFirst;
    private final boolean isTest;
    private final boolean isRegression;
    private final Set<String> missingStrings;

    public NumericDelimitedFileReader(
            ReaderOptions options
    ) {
        this(
                Objects.requireNonNull(
                        options,
                        "ReaderOptions cannot be null."
                ).getDataPath(),
                options.getLabelPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.hasMissingValues(),
                options.targetColumnIsFirst(),
                options.isTest(),
                options.isRegression()
        );
    }

    public NumericDelimitedFileReader(
            String dataFileName,
            String labelFileName,
            String entrySeparator,
            boolean hasHeader,
            boolean hasMissingValues,
            boolean targetColumnIsFirst,
            boolean isTest,
            boolean isRegression
    ) {
        this.dataFileName = requireNonblank(dataFileName, "dataFileName");
        this.labelFileName = normalizeNullableString(labelFileName);
        this.entrySeparator = validateAndNormalizeSeparator(entrySeparator);
        this.fieldSeparator = this.entrySeparator.charAt(0);
        this.hasHeader = hasHeader;
        this.hasMissingValues = hasMissingValues;
        this.targetColumnIsFirst = targetColumnIsFirst;
        this.isTest = isTest;
        this.isRegression = isRegression;
        this.missingStrings = snapshotMissingStrings();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        long start = System.nanoTime();

        Path dataPath = Path.of(dataFileName);
        validateDataFile(dataPath);

        List<Object> labels = labelFileName == null
                ? List.of()
                : DelimitedFileReader.readGenericLabels(
                labelFileName,
                hasHeader,
                isRegression
        );

        ListObjectDataset dataset = new ListObjectDataset();
        NumericDatasetCallbackHandler handler = new NumericDatasetCallbackHandler(
                dataPath,
                dataset,
                labels,
                hasHeader,
                hasMissingValues,
                shouldParseEmbeddedLabel(),
                targetColumnIsFirst,
                isRegression,
                missingStrings
        );

        try (CsvReader<Boolean> csvReader = CsvReader.builder()
                .fieldSeparator(fieldSeparator)
                .skipEmptyLines(false)
                .detectBomHeader(true)
                .build(handler, dataPath)) {
            for (Boolean ignored : csvReader) {
                // Parsing and dataset construction occur in the callback.
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException(
                    "Failed while parsing numeric delimited dataset: " + dataPath,
                    e
            );
        }

        int instanceCount = handler.getInstanceCount();
        if (instanceCount == 0) {
            throw new IOException(
                    "Numeric delimited dataset contains no data records: " + dataPath
            );
        }

        validateSeparateLabelCount(labels, instanceCount);

        /*
         * Preserve the legacy AppContext metadata contract without incorrectly
         * reporting the final row's length for unequal-length data.
         */
        int commonLength = handler.hasCommonFeatureCount()
                ? handler.getCommonFeatureCount()
                : 0;

        dataset.setLength(commonLength);
        AppContext.length = commonLength;

        DelimitedFileReader.ProgressLogger.logDuration(
                start,
                System.nanoTime()
        );
        return dataset;
    }

    private boolean shouldParseEmbeddedLabel() {
        return labelFileName == null
                && !AppContext.isIsolationMode()
                && (!isTest || AppContext.exists_testlabels);
    }

    private void validateSeparateLabelCount(
            List<Object> labels,
            int instanceCount
    ) {
        if (labelFileName == null) {
            return;
        }
        if (labels.size() != instanceCount) {
            throw new IllegalArgumentException(
                    "Separate label count does not match the number of data "
                            + "instances. Labels=" + labels.size()
                            + ", instances=" + instanceCount + "."
            );
        }
    }

    private static void validateDataFile(Path dataPath) throws IOException {
        if (!Files.exists(dataPath)) {
            throw new IOException(
                    "Numeric delimited data file does not exist: " + dataPath
            );
        }
        if (!Files.isRegularFile(dataPath)) {
            throw new IOException(
                    "Numeric delimited data path is not a regular file: " + dataPath
            );
        }
        if (!Files.isReadable(dataPath)) {
            throw new IOException(
                    "Numeric delimited data file is not readable: " + dataPath
            );
        }
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

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        if (trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")) {
            return null;
        }
        return trimmed;
    }

    private static String validateAndNormalizeSeparator(String separator) {
        if (separator == null || separator.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericDelimitedFileReader requires a non-empty entry separator."
            );
        }

        String normalized = switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };

        if (normalized.length() != 1) {
            throw new IllegalArgumentException(
                    "NumericDelimitedFileReader requires a single-character "
                            + "entry separator. Received: '" + separator + "'."
            );
        }

        char delimiter = normalized.charAt(0);
        if (delimiter == '\n' || delimiter == '\r') {
            throw new IllegalArgumentException(
                    "A line-separator character cannot be used as the entry separator."
            );
        }
        return normalized;
    }

    private static Set<String> snapshotMissingStrings() {
        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {
            return Set.of();
        }

        Set<String> normalized = new HashSet<>();
        for (String value : AppContext.MissingStrings) {
            if (value == null) {
                continue;
            }
            String trimmed = value.trim();
            if (!trimmed.isEmpty()) {
                normalized.add(trimmed.toUpperCase(Locale.ROOT));
            }
        }

        return normalized.isEmpty()
                ? Set.of()
                : Collections.unmodifiableSet(normalized);
    }

    /**
     * FastCSV callback that constructs primitive rows directly in the dataset.
     */
    private static final class NumericDatasetCallbackHandler
            extends AbstractBaseCsvCallbackHandler<Boolean> {

        private final Path file;
        private final ListObjectDataset dataset;
        private final List<Object> separateLabels;
        private final boolean hasHeader;
        private final boolean hasMissingValues;
        private final boolean embeddedLabel;
        private final boolean targetColumnIsFirst;
        private final boolean isRegression;
        private final Set<String> missingStrings;

        private List<String> firstRecordFields = new ArrayList<>();
        private int expectedFieldCount = -1;
        private int expectedFeatureCount = -1;
        private int instanceCount;
        private int commonFeatureCount = -1;
        private boolean commonFeatureCountValid = true;
        private boolean schemaResolved;

        private double[] currentFeatures;
        private String currentLabelToken;
        private int currentOutputIndex;

        private NumericDatasetCallbackHandler(
                Path file,
                ListObjectDataset dataset,
                List<Object> separateLabels,
                boolean hasHeader,
                boolean hasMissingValues,
                boolean embeddedLabel,
                boolean targetColumnIsFirst,
                boolean isRegression,
                Set<String> missingStrings
        ) {
            this.file = file;
            this.dataset = dataset;
            this.separateLabels = separateLabels;
            this.hasHeader = hasHeader;
            this.hasMissingValues = hasMissingValues;
            this.embeddedLabel = embeddedLabel;
            this.targetColumnIsFirst = targetColumnIsFirst;
            this.isRegression = isRegression;
            this.missingStrings = missingStrings;
        }

        @Override
        public void handleField(
                int fieldIndex,
                char[] buffer,
                int offset,
                int length,
                boolean quoted
        ) {
            if (!schemaResolved) {
                firstRecordFields.add(new String(buffer, offset, length));
                return;
            }

            if (fieldIndex >= expectedFieldCount) {
                throw inconsistentColumnCount(fieldIndex + 1);
            }

            if (currentFeatures == null) {
                currentFeatures = new double[expectedFeatureCount];
                currentOutputIndex = 0;
                currentLabelToken = null;
            }

            if (embeddedLabel && isLabelField(fieldIndex)) {
                currentLabelToken = new String(buffer, offset, length).trim();
                return;
            }

            currentFeatures[currentOutputIndex++] = parseNumericField(
                    buffer,
                    offset,
                    length,
                    fieldIndex
            );
        }

        @Override
        protected Boolean buildRecord() {
            int actualFieldCount = getFieldCount();

            if (!schemaResolved) {
                resolveSchema(actualFieldCount);
                schemaResolved = true;

                if (hasHeader) {
                    firstRecordFields = null;
                    return null;
                }

                appendBufferedFirstDataRecord();
                firstRecordFields = null;
                return Boolean.TRUE;
            }

            if (actualFieldCount != expectedFieldCount) {
                throw inconsistentColumnCount(actualFieldCount);
            }
            if (currentFeatures == null) {
                throw new IllegalArgumentException(
                        "Encountered an empty numeric record in file " + file
                                + " at CSV line " + getStartingLineNumber() + "."
                );
            }
            if (currentOutputIndex != expectedFeatureCount) {
                throw new IllegalStateException(
                        "Numeric record in file " + file + " produced "
                                + currentOutputIndex + " features; expected "
                                + expectedFeatureCount + "."
                );
            }

            Object label = resolveCurrentLabel();
            addInstance(label, currentFeatures);

            currentFeatures = null;
            currentLabelToken = null;
            currentOutputIndex = 0;
            return Boolean.TRUE;
        }

        private void resolveSchema(int fieldCount) {
            if (fieldCount <= 0 || firstRecordFields.size() != fieldCount) {
                throw new IllegalArgumentException(
                        "Numeric delimited file has no valid columns: " + file
                );
            }
            if (embeddedLabel && fieldCount < 2) {
                throw new IllegalArgumentException(
                        "Numeric delimited records in file " + file
                                + " must contain at least one feature and one label."
                );
            }

            expectedFieldCount = fieldCount;
            expectedFeatureCount = embeddedLabel
                    ? fieldCount - 1
                    : fieldCount;

            if (expectedFeatureCount <= 0) {
                throw new IllegalArgumentException(
                        "No numeric feature columns are available in file: " + file
                );
            }
        }

        private void appendBufferedFirstDataRecord() {
            double[] features = new double[expectedFeatureCount];
            int outputIndex = 0;
            String labelToken = null;

            for (int fieldIndex = 0;
                 fieldIndex < expectedFieldCount;
                 fieldIndex++) {
                String token = firstRecordFields.get(fieldIndex);
                if (embeddedLabel && isLabelField(fieldIndex)) {
                    labelToken = token == null ? "" : token.trim();
                    continue;
                }
                features[outputIndex++] = parseNumericToken(token, fieldIndex);
            }

            Object label = embeddedLabel
                    ? parseEmbeddedLabel(labelToken)
                    : getSeparateLabel(instanceCount);
            addInstance(label, features);
        }

        private boolean isLabelField(int fieldIndex) {
            return targetColumnIsFirst
                    ? fieldIndex == 0
                    : fieldIndex == expectedFieldCount - 1;
        }

        private double parseNumericField(
                char[] buffer,
                int offset,
                int length,
                int fieldIndex
        ) {
            int start = offset;
            int end = offset + length;
            while (start < end && Character.isWhitespace(buffer[start])) {
                start++;
            }
            while (end > start && Character.isWhitespace(buffer[end - 1])) {
                end--;
            }

            if (isMissingToken(buffer, start, end - start)) {
                if (hasMissingValues) {
                    return Double.NaN;
                }
                throw missingValue(fieldIndex);
            }

            try {
                return JavaDoubleParser.parseDouble(
                        buffer,
                        start,
                        end - start
                );
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Could not parse numeric value in file " + file
                                + " at instance " + instanceCount
                                + ", field " + fieldIndex
                                + ", starting CSV line "
                                + getStartingLineNumber() + ".",
                        e
                );
            }
        }

        private double parseNumericToken(String token, int fieldIndex) {
            String trimmed = token == null ? "" : token.trim();
            if (isMissingToken(trimmed)) {
                if (hasMissingValues) {
                    return Double.NaN;
                }
                throw missingValue(fieldIndex);
            }

            try {
                return JavaDoubleParser.parseDouble(trimmed);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Could not parse numeric value '" + trimmed
                                + "' in file " + file
                                + " at instance 0, field " + fieldIndex + ".",
                        e
                );
            }
        }

        private boolean isMissingToken(
                char[] buffer,
                int offset,
                int length
        ) {
            if (length == 0) {
                return true;
            }
            if (missingStrings.isEmpty()) {
                return false;
            }

            for (String indicator : missingStrings) {
                if (indicator.length() != length) {
                    continue;
                }
                boolean matches = true;
                for (int index = 0; index < length; index++) {
                    char actual = Character.toUpperCase(buffer[offset + index]);
                    if (actual != indicator.charAt(index)) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    return true;
                }
            }
            return false;
        }

        private boolean isMissingToken(String token) {
            return token == null
                    || token.isEmpty()
                    || missingStrings.contains(token.toUpperCase(Locale.ROOT));
        }

        private IllegalArgumentException missingValue(int fieldIndex) {
            return new IllegalArgumentException(
                    "Encountered a missing numeric value in file " + file
                            + " at instance " + instanceCount
                            + ", field " + fieldIndex
                            + ", but hasMissingValues is false."
            );
        }

        private Object resolveCurrentLabel() {
            if (embeddedLabel) {
                return parseEmbeddedLabel(currentLabelToken);
            }
            return getSeparateLabel(instanceCount);
        }

        private Object parseEmbeddedLabel(String token) {
            if (token == null || token.isEmpty()) {
                throw new IllegalArgumentException(
                        "Embedded label cannot be missing in file " + file
                                + " at instance " + instanceCount + "."
                );
            }

            if (isRegression) {
                try {
                    return JavaDoubleParser.parseDouble(token);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "Could not parse regression label '" + token
                                    + "' in file " + file
                                    + " at instance " + instanceCount + ".",
                            e
                    );
                }
            }
            return DelimitedFileReader.RowParser.tryParseLabel(token);
        }

        private Object getSeparateLabel(int index) {
            if (separateLabels.isEmpty()) {
                return null;
            }
            if (index >= separateLabels.size()) {
                throw new IllegalArgumentException(
                        "The separate label file contains fewer labels than "
                                + "the data file. Missing label for instance "
                                + index + "."
                );
            }
            return separateLabels.get(index);
        }

        private void addInstance(Object label, double[] features) {
            dataset.add(
                    label,
                    features,
                    instanceCount
            );

            if (commonFeatureCount < 0) {
                commonFeatureCount = features.length;
            } else if (features.length != commonFeatureCount) {
                commonFeatureCountValid = false;
            }

            DelimitedFileReader.ProgressLogger.logProgress(instanceCount);
            instanceCount++;
        }

        private IllegalArgumentException inconsistentColumnCount(
                int actualFieldCount
        ) {
            return new IllegalArgumentException(
                    "Inconsistent column count in file " + file
                            + " at CSV record beginning on line "
                            + getStartingLineNumber()
                            + ". Expected " + expectedFieldCount
                            + " columns but found " + actualFieldCount + "."
            );
        }

        private int getInstanceCount() {
            return instanceCount;
        }

        private boolean hasCommonFeatureCount() {
            return commonFeatureCountValid && commonFeatureCount >= 0;
        }

        private int getCommonFeatureCount() {
            return commonFeatureCount;
        }
    }
}